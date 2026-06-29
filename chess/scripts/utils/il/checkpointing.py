"""IL checkpoint/SWA helper functions."""

import math

import torch
from tqdm import tqdm

from src.model import save_checkpoint
from utils.shared.elo_estimator import estimate_model_elo


def build_runtime_state(
    scheduler,
    scaler,
    best_val_loss,
    patience_counter,
    estimated_elo=None,
    estimated_elo_epoch=None,
):
    """Build runtime state dictionary for checkpoint extra_state."""
    state = {
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }
    if estimated_elo is not None:
        try:
            state["estimated_elo"] = float(estimated_elo)
        except (TypeError, ValueError):
            pass
    if estimated_elo_epoch is not None:
        try:
            state["estimated_elo_epoch"] = int(estimated_elo_epoch)
        except (TypeError, ValueError):
            pass
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        state["scaler_state_dict"] = scaler.state_dict()
    return state


def save_swa_snapshot_checkpoint(
    epoch_idx,
    fallback_loss,
    ref_val_losses,
    ref_val_metrics,
    use_swa,
    swa_model,
    swa_start,
    il_dir,
    history_positions,
    expected_input_planes,
    model_version,
    start_mode,
    use_bfloat16,
    model_file_tag,
    model_architecture,
    training_batch_size=None,
):
    """Save SWA snapshot checkpoint (raw SWA weights, without BN refresh)."""
    if not use_swa or swa_model is None:
        return
    if (epoch_idx + 1) < swa_start:
        return

    model_tag = str(model_file_tag or "model").strip() or "model"
    # Keep exactly one rolling SWA snapshot to avoid file proliferation.
    checkpoint_name = f"{model_tag}_latest_swa.pt"

    ref_val_loss = None
    ref_top1 = None
    if ref_val_losses is not None:
        ref_val_loss = ref_val_losses.get("total")
    if ref_val_metrics is not None:
        ref_top1 = ref_val_metrics.get("policy_top1_acc")

    checkpoint_path = il_dir / checkpoint_name
    metadata = {
        "swa_enabled": True,
        "swa_snapshot": True,
        "swa_bn_updated": False,
        "swa_start_epoch": swa_start,
        "history_positions": history_positions,
        "input_planes": expected_input_planes,
        "pov_enabled": True,
        "version": model_version,
        "startup_mode": start_mode,
        "model_architecture": dict(model_architecture or {}),
    }
    if training_batch_size is not None:
        try:
            metadata["training_batch_size"] = int(training_batch_size)
        except (TypeError, ValueError):
            pass
    if ref_val_loss is not None:
        metadata["val_loss"] = float(ref_val_loss)
        metadata["val_policy_loss"] = ref_val_losses.get("policy")
        metadata["val_value_loss"] = ref_val_losses.get("value")
    if ref_top1 is not None:
        metadata["val_policy_top1"] = float(ref_top1)
    if ref_val_metrics is not None:
        metadata["val_policy_top3"] = ref_val_metrics.get("policy_top3_acc")
        metadata["val_value_mae"] = ref_val_metrics.get("value_mae")

    save_checkpoint(
        swa_model.module,
        None,
        epoch_idx,
        float(fallback_loss),
        str(checkpoint_path),
        metadata,
        save_optimizer=False,
        save_dtype=torch.bfloat16 if use_bfloat16 else None,
    )
    size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
    print(
        f"Saved SWA snapshot: {checkpoint_path.name} ({size_mb:.1f} MB, "
        "BN stats refresh pending)"
    )


def _loader_batch_size(base_loader, preferred_batch_size=None):
    if preferred_batch_size is not None:
        try:
            preferred_batch_size = int(preferred_batch_size)
            if preferred_batch_size > 0:
                return preferred_batch_size
        except (TypeError, ValueError):
            pass

    batch_size = getattr(base_loader, "batch_size", None)
    if batch_size is None:
        batch_sampler = getattr(base_loader, "batch_sampler", None)
        batch_size = getattr(batch_sampler, "batch_size", None)
    if batch_size is None:
        batch_size = 1
    return max(1, int(batch_size))


def _build_single_worker_loader(base_loader, shuffle=False, batch_size=None):
    """Build a safe fallback DataLoader that avoids worker subprocesses."""
    dataset = getattr(base_loader, "dataset", None)
    if dataset is None:
        return None

    batch_size = _loader_batch_size(base_loader, batch_size)

    collate_fn = getattr(base_loader, "collate_fn", None)
    drop_last = bool(getattr(base_loader, "drop_last", False))
    pin_memory = bool(getattr(base_loader, "pin_memory", False))

    try:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
    except Exception:
        return None


def _build_fresh_multi_worker_loader(base_loader, shuffle=False, batch_size=None):
    """Build a fresh multi-worker DataLoader for one-off BN refresh after interrupts."""
    dataset = getattr(base_loader, "dataset", None)
    if dataset is None:
        return None

    workers = int(getattr(base_loader, "num_workers", 0) or 0)
    if workers <= 0:
        return None

    batch_size = _loader_batch_size(base_loader, batch_size)

    collate_fn = getattr(base_loader, "collate_fn", None)
    drop_last = bool(getattr(base_loader, "drop_last", False))
    pin_memory = bool(getattr(base_loader, "pin_memory", False))
    prefetch_factor = getattr(base_loader, "prefetch_factor", None)

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        # Force fresh worker lifecycle for shutdown-time BN refresh.
        "persistent_workers": False,
    }
    if isinstance(prefetch_factor, int) and prefetch_factor > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    try:
        return torch.utils.data.DataLoader(**kwargs)
    except Exception:
        return None


def _extract_model_inputs(batch):
    """Extract model input tensor from common batch formats."""
    if isinstance(batch, dict):
        if "board" in batch:
            return batch["board"]
        raise KeyError("batch dict missing required 'board' key")
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("empty batch tuple/list")
        return batch[0]
    return batch


class _InputOnlyLoader:
    """Adapter exposing only model inputs for torch.optim.swa_utils.update_bn."""

    def __init__(self, base_loader):
        self.base_loader = base_loader
        self.dataset = getattr(base_loader, "dataset", None)
        self.batch_size = getattr(base_loader, "batch_size", None)

    def __iter__(self):
        for batch in self.base_loader:
            yield _extract_model_inputs(batch)

    def __len__(self):
        return len(self.base_loader)


def _update_bn_from_loader(loader, swa_model, device, use_amp, amp_dtype, desc="SWA BN refresh"):
    """Run SWA BatchNorm refresh using only model inputs from loader batches."""
    momenta = {}
    was_training = swa_model.training
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    swa_model.train()
    for module in momenta:
        module.momentum = None

    try:
        iterator = tqdm(
            _InputOnlyLoader(loader),
            total=len(loader),
            desc=desc,
            unit="batch",
            leave=True,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                for inputs in iterator:
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device, non_blocking=True)
                    swa_model(inputs)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        swa_model.train(was_training)


def _refresh_swa_bn_stats(
    train_loader,
    swa_model,
    device,
    use_amp,
    use_bfloat16,
    prefer_fresh_multi_worker=False,
    training_batch_size=None,
):
    """Refresh BN stats with fallback to a single-worker loader if needed.

    When ``prefer_fresh_multi_worker`` is enabled (e.g. after Ctrl+C),
    we first try a freshly spawned multi-worker loader (same worker count),
    then fallback to single-worker only if needed.
    """
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    refreshed_multi_loader = None
    tried_refreshed_multi = False
    resolved_batch_size = _loader_batch_size(train_loader, training_batch_size)
    print(f"  SWA BN refresh batch_size={resolved_batch_size:,}")
    if prefer_fresh_multi_worker:
        refreshed_multi_loader = _build_fresh_multi_worker_loader(
            train_loader,
            shuffle=False,
            batch_size=resolved_batch_size,
        )
        tried_refreshed_multi = True
        if refreshed_multi_loader is None:
            print(
                "WARNING: Could not build fresh multi-worker loader for BN refresh; "
                "falling back."
            )
        else:
            try:
                _update_bn_from_loader(
                    refreshed_multi_loader,
                    swa_model,
                    device,
                    use_amp,
                    amp_dtype,
                    desc="SWA BN refresh",
                )
                print("BatchNorm statistics updated via fresh multi-worker loader.")
                return True, "fresh_multi_worker", None
            except Exception as refresh_exc:
                print(f"WARNING: BN refresh failed on fresh multi-worker loader ({refresh_exc})")

    try:
        _update_bn_from_loader(
            train_loader,
            swa_model,
            device,
            use_amp,
            amp_dtype,
            desc="SWA BN refresh",
        )
        return True, "train_loader", None
    except Exception as primary_exc:
        print(f"WARNING: BN refresh failed on training loader ({primary_exc})")

    if not tried_refreshed_multi:
        refreshed_multi_loader = _build_fresh_multi_worker_loader(
            train_loader,
            shuffle=False,
            batch_size=resolved_batch_size,
        )
        if refreshed_multi_loader is not None:
            try:
                _update_bn_from_loader(
                    refreshed_multi_loader,
                    swa_model,
                    device,
                    use_amp,
                    amp_dtype,
                    desc="SWA BN refresh",
                )
                print("BatchNorm statistics updated via fresh multi-worker loader.")
                return True, "fresh_multi_worker", None
            except Exception as refresh_exc:
                print(f"WARNING: BN refresh failed on fresh multi-worker loader ({refresh_exc})")

    safe_loader = _build_single_worker_loader(
        train_loader,
        shuffle=False,
        batch_size=resolved_batch_size,
    )
    if safe_loader is None:
        return False, "none", "failed to build safe single-worker training loader"

    try:
        _update_bn_from_loader(
            safe_loader,
            swa_model,
            device,
            use_amp,
            amp_dtype,
            desc="SWA BN refresh",
        )
        print("BatchNorm statistics updated via safe single-worker loader.")
        return True, "safe_single_worker", None
    except Exception as fallback_exc:
        return False, "none", str(fallback_exc)


def _evaluate_swa_model_with_fallback(evaluate_il_fn, swa_model, val_loader, config, device):
    """Evaluate SWA model with fallback to single-worker validation loader."""
    try:
        losses, metrics = evaluate_il_fn(swa_model.module, val_loader, config, device)
        return losses, metrics, "val_loader"
    except Exception as primary_exc:
        print(f"WARNING: SWA eval failed on validation loader ({primary_exc})")

    safe_loader = _build_single_worker_loader(val_loader, shuffle=False)
    if safe_loader is None:
        raise RuntimeError("failed to build safe single-worker validation loader")

    losses, metrics = evaluate_il_fn(swa_model.module, safe_loader, config, device)
    print("SWA evaluation completed via safe single-worker validation loader.")
    return losses, metrics, "safe_single_worker"


def _safe_loss_value(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def finalize_swa_model(
    final_epoch_idx,
    interrupted,
    use_swa,
    swa_model,
    use_amp,
    use_bfloat16,
    train_loader,
    val_loader,
    config,
    device,
    best_model_path,
    swa_start,
    history_positions,
    expected_input_planes,
    model_version,
    start_mode,
    best_val_loss,
    evaluate_il_fn,
    best_raw_val_loss=None,
    elo_config=None,
    elo_stop_event=None,
    model_architecture=None,
    training_batch_size=None,
):
    """Finalize SWA model with BN refresh, optional Elo, and save best_model_il_swa.pt."""
    result = {
        "finalized": False,
        "val_loss": None,
        "val_policy_loss": None,
        "val_value_loss": None,
        "val_top1": None,
        "val_top3": None,
        "val_mae": None,
        "val_wdl_acc": None,
        "val_wdl_ce": None,
        "estimated_elo": None,
        "estimated_elo_epoch": None,
        "elo_cancelled": False,
        "model_path": None,
    }
    if not use_swa or swa_model is None:
        return result

    n_averaged_tensor = getattr(swa_model, "n_averaged", None)
    n_averaged = int(n_averaged_tensor.item()) if n_averaged_tensor is not None else 0
    if n_averaged <= 0:
        if interrupted:
            print("SWA finalization skipped: no SWA averages collected yet.")
        return result

    trigger_label = "after interrupt" if interrupted else "at training end"
    print(f"\nFinalizing SWA ({trigger_label})...")
    print("Updating BatchNorm statistics...")
    bn_updated, bn_refresh_mode, bn_refresh_error = _refresh_swa_bn_stats(
        train_loader=train_loader,
        swa_model=swa_model,
        device=device,
        use_amp=use_amp,
        use_bfloat16=use_bfloat16,
        prefer_fresh_multi_worker=bool(interrupted),
        training_batch_size=training_batch_size,
    )
    if not bn_updated:
        print(
            "WARNING: SWA BN refresh failed; continuing with current BN stats "
            f"({bn_refresh_error})"
        )

    print("Evaluating SWA model...")
    eval_error = None
    try:
        swa_val_losses, swa_val_metrics, eval_mode = _evaluate_swa_model_with_fallback(
            evaluate_il_fn=evaluate_il_fn,
            swa_model=swa_model,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        print(
            f"SWA metrics: val_loss={swa_val_losses['total']:.4f}, "
            f"val_mae={swa_val_metrics['value_mae']:.4f}, "
            f"top1={swa_val_metrics['policy_top1_acc']:.2%}, "
            f"top3={swa_val_metrics.get('policy_top3_acc', float('nan')):.2%}, "
            f"wdl_acc={swa_val_metrics.get('value_wdl_acc', float('nan')):.2%}"
        )
        result["val_loss"]        = float(swa_val_losses["total"])
        result["val_policy_loss"] = float(swa_val_losses["policy"]) if swa_val_losses.get("policy") is not None else None
        result["val_value_loss"]  = float(swa_val_losses["value"])  if swa_val_losses.get("value")  is not None else None
        result["val_top1"]    = float(swa_val_metrics["policy_top1_acc"])
        result["val_top3"]    = float(swa_val_metrics.get("policy_top3_acc", float("nan")))
        result["val_mae"]     = float(swa_val_metrics["value_mae"])
        result["val_wdl_acc"] = float(swa_val_metrics.get("value_wdl_acc",  float("nan"))) if swa_val_metrics.get("value_wdl_acc")  is not None else None
        result["val_wdl_ce"]  = float(swa_val_metrics.get("value_wdl_ce",   float("nan"))) if swa_val_metrics.get("value_wdl_ce")   is not None else None
    except Exception as exc:
        eval_error = str(exc)
        eval_mode = "none"
        fallback_loss = _safe_loss_value(best_val_loss, default=0.0)
        swa_val_losses = {
            "total": float(fallback_loss),
            "policy": None,
            "value": None,
        }
        swa_val_metrics = {
            "policy_top1_acc": float("nan"),
            "policy_top3_acc": float("nan"),
            "value_mae": float("nan"),
        }
        result["val_loss"] = float(fallback_loss)
        result["val_top1"] = float("nan")
        result["val_mae"] = float("nan")
        print(
            "WARNING: SWA evaluation failed; saving SWA without eval metrics "
            f"({eval_error})"
        )

    swa_model_path = best_model_path.parent / "best_model_il_swa.pt"
    epoch_to_store = max(int(final_epoch_idx), 0)
    result["model_path"] = str(swa_model_path)

    if isinstance(elo_config, dict) and elo_config.get("enabled", False):
        use_mcts_for_elo = bool(elo_config.get("use_mcts", False))
        simulations_for_elo = int(elo_config.get("mcts_simulations", 0) or 0)
        mode_label = f"MCTS {simulations_for_elo} sims" if use_mcts_for_elo else "raw NN"
        print(f"Estimating SWA Elo ({mode_label}, separate final check)...")
        try:
            elo_result = estimate_model_elo(
                swa_model.module,
                config,
                device,
                elo_config,
                stop_event=elo_stop_event,
            )
        except KeyboardInterrupt:
            if elo_stop_event is not None:
                elo_stop_event.set()
            elo_result = {"cancelled": True}

        if elo_result.get("cancelled"):
            result["elo_cancelled"] = True
            print("SWA Elo estimation cancelled.")
        else:
            swa_elo = elo_result.get("estimated_elo")
            if swa_elo is not None:
                result["estimated_elo"] = float(swa_elo)
                result["estimated_elo_epoch"] = int(epoch_to_store + 1)
                result["estimated_elo_mode"] = "mcts" if use_mcts_for_elo else "nn"
                result["estimated_elo_simulations"] = simulations_for_elo if use_mcts_for_elo else 0
                result["estimated_elo_std_error"] = elo_result.get("elo_std_error")
                result["estimated_elo_ci95"] = elo_result.get("elo_ci95")
                print(f"SWA Estimated Elo ({mode_label}): {int(round(float(swa_elo)))}")
                for lvl, res in sorted(elo_result.get("results", {}).items()):
                    score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
                    print(f"  vs SF {lvl}: {score_str} (score: {res['score']:.0%})")
                print(f"  time {elo_result['total_time']:.1f}s ({elo_result['total_games']} games)")
            elif elo_result.get("error"):
                print(f"SWA Elo estimation failed: {elo_result.get('error')}")
            elif not elo_result.get("skipped"):
                print("SWA Elo estimation: inconclusive")

    metadata = {
        "swa_enabled": True,
        "swa_start_epoch": swa_start,
        "swa_bn_updated": bool(bn_updated),
        "swa_bn_refresh_mode": bn_refresh_mode,
        "swa_finalized_on_interrupt": bool(interrupted),
        "history_positions": history_positions,
        "input_planes": expected_input_planes,
        "pov_enabled": True,
        "version": model_version,
        "startup_mode": start_mode,
        "swa_eval_mode": eval_mode,
        "model_architecture": dict(model_architecture or {}),
    }
    if training_batch_size is not None:
        try:
            metadata["training_batch_size"] = int(training_batch_size)
        except (TypeError, ValueError):
            pass
    if bn_refresh_error:
        metadata["swa_bn_refresh_error"] = str(bn_refresh_error)
    if eval_error:
        metadata["swa_eval_error"] = str(eval_error)

    if swa_val_losses.get("total") is not None:
        metadata["val_loss"] = float(swa_val_losses["total"])
    if swa_val_losses.get("policy") is not None:
        metadata["val_policy_loss"] = float(swa_val_losses["policy"])
    if swa_val_losses.get("value") is not None:
        metadata["val_value_loss"] = float(swa_val_losses["value"])
    if swa_val_metrics.get("policy_top1_acc") is not None:
        metadata["val_policy_top1"] = float(swa_val_metrics["policy_top1_acc"])
    if swa_val_metrics.get("policy_top3_acc") is not None:
        metadata["val_policy_top3"] = float(swa_val_metrics["policy_top3_acc"])
    if swa_val_metrics.get("value_mae") is not None:
        metadata["val_value_mae"] = float(swa_val_metrics["value_mae"])

    if result["estimated_elo"] is not None:
        metadata["estimated_elo"] = float(result["estimated_elo"])
        metadata["estimated_elo_epoch"] = int(result["estimated_elo_epoch"] or (epoch_to_store + 1))
        mode = str(result.get("estimated_elo_mode") or "nn").lower()
        mode_key = "estimated_elo_mcts" if mode == "mcts" else "estimated_elo_nn"
        simulations = int(result.get("estimated_elo_simulations") or 0)
        metadata[mode_key] = float(result["estimated_elo"])
        metadata[f"last_{mode_key}"] = float(result["estimated_elo"])
        metadata["estimated_elo_settings"] = {
            "use_mcts": mode == "mcts",
            "simulations": simulations if mode == "mcts" else 0,
        }
        if mode == "mcts":
            metadata["estimated_elo_mcts_simulations"] = simulations
        if result.get("estimated_elo_std_error") is not None:
            metadata["estimated_elo_se"] = float(result["estimated_elo_std_error"])
        ci95 = result.get("estimated_elo_ci95")
        if isinstance(ci95, (list, tuple)) and len(ci95) == 2:
            metadata["estimated_elo_ci95_low"] = float(ci95[0])
            metadata["estimated_elo_ci95_high"] = float(ci95[1])

    try:
        checkpoint_loss = _safe_loss_value(swa_val_losses.get("total"), default=0.0)
        save_checkpoint(
            swa_model.module,
            None,
            epoch_to_store,
            checkpoint_loss,
            str(swa_model_path),
            metadata,
            save_optimizer=False,
            save_dtype=torch.bfloat16 if use_bfloat16 else None,
        )
    except Exception as exc:
        print(f"WARNING: SWA finalization failed ({exc})")
        return result

    size_mb = swa_model_path.stat().st_size / (1024 ** 2)
    best_loss_source = best_raw_val_loss if best_raw_val_loss is not None else best_val_loss
    best_loss = _safe_loss_value(best_loss_source, default=float("nan"))
    val_loss = _safe_loss_value(result.get("val_loss"), default=float("nan"))
    if math.isfinite(best_loss) and math.isfinite(val_loss):
        delta_text = f"{best_loss - val_loss:.4f}"
    else:
        delta_text = "n/a"
    elo_tail = (
        f", elo={int(round(float(result['estimated_elo'])))}"
        if result["estimated_elo"] is not None
        else ""
    )
    print(
        f"SWA model saved: {swa_model_path.name} ({size_mb:.1f} MB), "
        f"val_loss_delta={delta_text}, bn_refresh={bn_refresh_mode}{elo_tail}"
    )
    result["finalized"] = True
    return result
