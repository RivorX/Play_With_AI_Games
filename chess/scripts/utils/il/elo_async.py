"""Async/sync Elo coordination for IL training."""

import copy
import queue
import threading

import torch

from src.model import ChessNet
from utils.shared.elo_estimator import estimate_model_elo


def _resolve_async_device(main_device, configured_value):
    """Resolve async Elo worker device (`cpu`, `cuda`, or `same`)."""
    value = str(configured_value).strip().lower()
    if value in {"same", ""}:
        resolved = main_device
    elif value == "cuda":
        resolved = torch.device("cuda")
    else:
        resolved = torch.device("cpu")

    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _snapshot_model_state_cpu(model):
    """Create an immutable CPU snapshot of model weights for async Elo."""
    return {
        key: tensor.detach().to(device="cpu", copy=True)
        for key, tensor in model.state_dict().items()
    }


def _async_elo_worker(
    epoch_num,
    model_state_cpu,
    config_snapshot,
    elo_config_snapshot,
    worker_device_str,
    result_queue,
    cancel_event,
):
    """Background Elo estimation worker for IL (thread target)."""
    try:
        if cancel_event is not None and cancel_event.is_set():
            result_queue.put(
                {
                    "epoch": int(epoch_num),
                    "cancelled": True,
                }
            )
            return

        worker_config = copy.deepcopy(config_snapshot)
        worker_config.setdefault("model", {})
        worker_config["model"]["print_summary"] = False

        worker_device = torch.device(worker_device_str)
        if worker_device.type == "cuda" and not torch.cuda.is_available():
            worker_device = torch.device("cpu")

        worker_model = ChessNet(worker_config).to(worker_device)
        worker_model = worker_model.to(memory_format=torch.channels_last)
        worker_model.load_state_dict(model_state_cpu)
        worker_model.eval()

        elo_result = estimate_model_elo(
            worker_model,
            worker_config,
            worker_device,
            elo_config_snapshot,
            stop_event=cancel_event,
        )
        result_queue.put(
            {
                "epoch": int(epoch_num),
                "result": elo_result,
                "device": worker_device.type,
            }
        )
    except KeyboardInterrupt:
        result_queue.put(
            {
                "epoch": int(epoch_num),
                "cancelled": True,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "epoch": int(epoch_num),
                "error": str(exc),
            }
        )


class ILEloCoordinator:
    """Coordinates periodic Elo estimation (sync or async) for IL."""

    def __init__(self, model, config, device, elo_config_il, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.elo_config = dict(elo_config_il or {})

        self.enabled = bool(self.elo_config.get("enabled", False))
        self.eval_every = int(self.elo_config.get("eval_every", 5))
        self.async_enabled = bool(self.elo_config.get("async_in_il", True)) if self.enabled else False
        self.async_device = _resolve_async_device(device, self.elo_config.get("async_device", "cpu"))
        self.shutdown_wait_sec = float(self.elo_config.get("async_wait_on_shutdown_sec", 0.0))

        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.worker_epoch = None
        self.worker_cancel_event = None

    def print_startup_summary(self):
        """Print runtime Elo configuration."""
        if not self.enabled:
            return
        print(
            f"Elo eval: every {self.eval_every} ep, "
            f"levels={self.elo_config.get('levels', [1000, 1300, 1600, 1900, 2200])}, "
            f"games/level={self.elo_config.get('games_per_level', 4)}, "
            f"mcts={self.elo_config.get('use_mcts', False)}"
        )
        if self.async_enabled:
            print(
                f"Elo async (IL): enabled, worker_device={self.async_device.type}, "
                "training continues while Elo is running"
            )

    def is_due(self, epoch_num):
        """Return True if Elo should run for given epoch number (1-based)."""
        return self.enabled and (int(epoch_num) % self.eval_every == 0)

    def poll_results(self):
        """Collect finished async Elo jobs, write CSV backfills, refresh PNG if needed."""
        any_new_elo = False

        while True:
            try:
                payload = self.result_queue.get_nowait()
            except queue.Empty:
                break

            result_epoch = int(payload.get("epoch", -1))
            worker_error = payload.get("error")
            if payload.get("cancelled"):
                pass
            elif worker_error:
                print(f"     [async] Elo failed for epoch {result_epoch}: {worker_error}")
            else:
                elo_result = payload.get("result", {}) or {}
                estimated_elo = elo_result.get("estimated_elo")
                if estimated_elo is not None:
                    self.logger.record_estimated_elo(result_epoch, estimated_elo, update_csv=True)
                    print(f"     [async] Epoch {result_epoch} Estimated Elo: {estimated_elo}")
                    for lvl, res in sorted(elo_result.get("results", {}).items()):
                        score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
                        print(f"       vs SF {lvl}: {score_str} (score: {res['score']:.0%})")
                    print(f"       time {elo_result['total_time']:.1f}s ({elo_result['total_games']} games)")
                    any_new_elo = True
                elif "error" not in elo_result and not elo_result.get("skipped"):
                    print(f"     [async] Epoch {result_epoch} Elo estimation: inconclusive")

            if self.worker_epoch == result_epoch:
                self.worker_epoch = None
                self.worker_thread = None
                self.worker_cancel_event = None

        # Defensive cleanup in case worker ended without queue payload.
        if self.worker_thread is not None and not self.worker_thread.is_alive() and self.worker_epoch is not None:
            print(f"     [async] Elo worker for epoch {self.worker_epoch} finished without payload.")
            self.worker_epoch = None
            self.worker_thread = None
            self.worker_cancel_event = None

        if any_new_elo:
            self.logger.plot()

    def start_async(self, epoch_num):
        """Start async Elo job if worker is available."""
        if not self.async_enabled:
            return False

        if self.worker_thread is not None and self.worker_thread.is_alive():
            print(
                f"     [async] Worker busy (epoch {self.worker_epoch}); "
                f"skipping new Elo launch for epoch {epoch_num}."
            )
            return False

        model_state_cpu = _snapshot_model_state_cpu(self.model)
        worker_config = copy.deepcopy(self.config)
        worker_elo_cfg = dict(self.elo_config)
        worker_elo_cfg.setdefault("prioritize_training", True)
        worker_elo_cfg.setdefault("reserve_dataloader_workers", True)
        worker_elo_cfg.setdefault("free_threads_utilization", 0.8)
        worker_elo_cfg.setdefault("stockfish_priority", "below_normal")
        worker_elo_cfg.setdefault("stockfish_hide_window", True)
        worker_elo_cfg.setdefault("max_error_logs_per_type", 8)
        cancel_event = threading.Event()

        worker = threading.Thread(
            target=_async_elo_worker,
            args=(
                int(epoch_num),
                model_state_cpu,
                worker_config,
                worker_elo_cfg,
                self.async_device.type,
                self.result_queue,
                cancel_event,
            ),
            daemon=True,
        )
        worker.start()
        self.worker_thread = worker
        self.worker_epoch = int(epoch_num)
        self.worker_cancel_event = cancel_event
        print(
            f"     [async] Started Elo for epoch {epoch_num} on {self.async_device.type}. "
            "Training continues."
        )
        return True

    def evaluate_if_due(self, epoch_num):
        """Run/schedule Elo for epoch and return sync `estimated_elo` or None."""
        if not self.is_due(epoch_num):
            return None

        if self.async_enabled:
            self.poll_results()
            self.start_async(epoch_num)
            return None

        print("\n     Estimating Elo (vs Stockfish)...")
        elo_result = estimate_model_elo(self.model, self.config, self.device, self.elo_config)
        estimated_elo = elo_result.get("estimated_elo")
        if estimated_elo is not None:
            print(f"     Estimated Elo: {estimated_elo}")
            for lvl, res in sorted(elo_result.get("results", {}).items()):
                score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
                print(f"       vs SF {lvl}: {score_str} (score: {res['score']:.0%})")
            print(f"       time {elo_result['total_time']:.1f}s ({elo_result['total_games']} games)")
        elif "error" not in elo_result and not elo_result.get("skipped"):
            print("     Elo estimation: inconclusive")
        return estimated_elo

    def shutdown(self, interrupted=False):
        """Handle outstanding async Elo job at shutdown."""
        self.poll_results()
        if not self.async_enabled or self.worker_thread is None or not self.worker_thread.is_alive():
            return

        if interrupted and self.worker_cancel_event is not None:
            self.worker_cancel_event.set()
            print(f"Cancelling async Elo worker for epoch {self.worker_epoch}...")

        # Normal shutdown with zero wait should still request cancellation,
        # otherwise nested worker pools/processes can keep Python alive.
        if (not interrupted) and self.shutdown_wait_sec <= 0 and self.worker_cancel_event is not None:
            self.worker_cancel_event.set()
            print(
                f"Cancelling async Elo worker for epoch {self.worker_epoch} "
                "(shutdown wait=0)."
            )

        if (not interrupted) and self.shutdown_wait_sec > 0:
            print(
                f"Waiting up to {self.shutdown_wait_sec:.1f}s for async Elo "
                f"(epoch {self.worker_epoch})..."
            )
            self.worker_thread.join(timeout=self.shutdown_wait_sec)
            self.poll_results()
        elif (not interrupted) and self.shutdown_wait_sec <= 0:
            # Give worker a brief chance to observe cancellation and exit.
            self.worker_thread.join(timeout=1.0)
            self.poll_results()
        elif interrupted:
            # On Ctrl+C wait briefly so worker can stop and flush result queue.
            self.worker_thread.join(timeout=2.0)
            self.poll_results()

        if self.worker_thread is not None and self.worker_thread.is_alive():
            print(
                f"Async Elo for epoch {self.worker_epoch} still running; "
                "continuing shutdown without waiting."
            )
