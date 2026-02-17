"""IL checkpoint/SWA helper functions."""

import torch

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
    use_mtl,
    history_positions,
    expected_input_planes,
    stride,
    model_version,
    start_mode,
    use_bfloat16,
    model_file_tag,
    model_architecture,
):
    """Save SWA snapshot checkpoint (raw SWA weights, without BN refresh)."""
    if not use_swa or swa_model is None:
        return
    if (epoch_idx + 1) < swa_start:
        return

    epoch_num = epoch_idx + 1
    model_tag = str(model_file_tag or "model").strip() or "model"
    checkpoint_name = f"{model_tag}_epoch_{epoch_num:02d}_swa.pt"

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
        "use_mtl": use_mtl,
        "history_positions": history_positions,
        "input_planes": expected_input_planes,
        "sliding_window_stride": stride,
        "pov_enabled": True,
        "version": model_version,
        "startup_mode": start_mode,
        "model_architecture": dict(model_architecture or {}),
    }
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
    use_mtl,
    swa_start,
    history_positions,
    expected_input_planes,
    stride,
    model_version,
    start_mode,
    best_val_loss,
    evaluate_il_fn,
    elo_config=None,
    model_architecture=None,
):
    """Finalize SWA model with BN refresh, optional Elo, and save best_model_il_swa.pt."""
    result = {
        "finalized": False,
        "val_loss": None,
        "val_top1": None,
        "val_mae": None,
        "estimated_elo": None,
        "estimated_elo_epoch": None,
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
    try:
        print("Updating BatchNorm statistics...")
        amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        print("Evaluating SWA model...")
        swa_val_losses, swa_val_metrics = evaluate_il_fn(swa_model.module, val_loader, config, device)
        print(
            f"SWA metrics: val_loss={swa_val_losses['total']:.4f}, "
            f"val_mae={swa_val_metrics['value_mae']:.4f}, "
            f"top1={swa_val_metrics['policy_top1_acc']:.2%}"
        )
        result["val_loss"] = float(swa_val_losses["total"])
        result["val_top1"] = float(swa_val_metrics["policy_top1_acc"])
        result["val_mae"] = float(swa_val_metrics["value_mae"])

        swa_model_path = best_model_path.parent / "best_model_il_swa.pt"
        epoch_to_store = max(int(final_epoch_idx), 0)
        result["model_path"] = str(swa_model_path)

        swa_elo = None
        if isinstance(elo_config, dict) and elo_config.get("enabled", False):
            print("Estimating SWA Elo (separate final check)...")
            elo_result = estimate_model_elo(
                swa_model.module,
                config,
                device,
                elo_config,
            )
            swa_elo = elo_result.get("estimated_elo")
            if swa_elo is not None:
                result["estimated_elo"] = float(swa_elo)
                result["estimated_elo_epoch"] = int(epoch_to_store + 1)
                print(f"SWA Estimated Elo: {int(round(float(swa_elo)))}")
                for lvl, res in sorted(elo_result.get("results", {}).items()):
                    score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
                    print(f"  vs SF {lvl}: {score_str} (score: {res['score']:.0%})")
                print(f"  time {elo_result['total_time']:.1f}s ({elo_result['total_games']} games)")
            elif elo_result.get("error"):
                print(f"SWA Elo estimation failed: {elo_result.get('error')}")
            elif not elo_result.get("skipped"):
                print("SWA Elo estimation: inconclusive")

        metadata = {
            "val_loss": swa_val_losses["total"],
            "val_policy_loss": swa_val_losses["policy"],
            "val_value_loss": swa_val_losses["value"],
            "val_policy_top1": swa_val_metrics["policy_top1_acc"],
            "val_policy_top3": swa_val_metrics["policy_top3_acc"],
            "val_value_mae": swa_val_metrics["value_mae"],
            "swa_enabled": True,
            "swa_start_epoch": swa_start,
            "swa_bn_updated": True,
            "swa_finalized_on_interrupt": bool(interrupted),
            "use_mtl": use_mtl,
            "history_positions": history_positions,
            "input_planes": expected_input_planes,
            "sliding_window_stride": stride,
            "pov_enabled": True,
            "version": model_version,
            "startup_mode": start_mode,
            "model_architecture": dict(model_architecture or {}),
        }
        if result["estimated_elo"] is not None:
            metadata["estimated_elo"] = float(result["estimated_elo"])
            metadata["estimated_elo_epoch"] = int(result["estimated_elo_epoch"] or (epoch_to_store + 1))

        save_checkpoint(
            swa_model.module,
            None,
            epoch_to_store,
            swa_val_losses["total"],
            str(swa_model_path),
            metadata,
            save_optimizer=False,
            save_dtype=torch.bfloat16 if use_bfloat16 else None,
        )

        size_mb = swa_model_path.stat().st_size / (1024 ** 2)
        elo_tail = (
            f", elo={int(round(float(result['estimated_elo'])))}"
            if result["estimated_elo"] is not None
            else ""
        )
        print(
            f"SWA model saved: {swa_model_path.name} ({size_mb:.1f} MB), "
            f"val_loss_delta={best_val_loss - swa_val_losses['total']:.4f}{elo_tail}"
        )
        result["finalized"] = True
        return result
    except Exception as exc:
        print(f"WARNING: SWA finalization failed ({exc})")
        return result
