"""
Training and evaluation functions for IL with WDL and move-weighted losses
"""

import queue
import threading
import time
import torch
from tqdm import tqdm

from src.utils.data_helpers import ACTION_SIZE, build_hflip_inverse_index_map

from .loss import CombinedLoss
from ..shared.metrics import MetricsCalculator


_IL_HFLIP_FORWARD_INDEX_MAP = None


class _DataLoaderTailSkip(StopIteration):
    def __init__(self, skipped_batches):
        super().__init__(skipped_batches)
        self.skipped_batches = max(0, int(skipped_batches or 0))


def _move_batch_to_device(batch_data, device, non_blocking=True):
    """Move a DataLoader batch to device, keeping board tensors channels-last."""
    if torch.is_tensor(batch_data):
        return batch_data.to(device, non_blocking=non_blocking)
    if isinstance(batch_data, dict):
        moved = {}
        for key, value in batch_data.items():
            if torch.is_tensor(value):
                if key == 'board':
                    moved[key] = value.to(
                        device,
                        memory_format=torch.channels_last,
                        non_blocking=non_blocking,
                    )
                else:
                    moved[key] = value.to(device, non_blocking=non_blocking)
            else:
                moved[key] = value
        return moved
    if isinstance(batch_data, (tuple, list)):
        moved_items = []
        for idx, value in enumerate(batch_data):
            if torch.is_tensor(value):
                if idx == 0:
                    moved_items.append(
                        value.to(
                            device,
                            memory_format=torch.channels_last,
                            non_blocking=non_blocking,
                        )
                    )
                else:
                    moved_items.append(value.to(device, non_blocking=non_blocking))
            else:
                moved_items.append(value)
        return type(batch_data)(moved_items)
    return batch_data


def _record_batch_stream(batch_data, stream):
    if torch.is_tensor(batch_data):
        batch_data.record_stream(stream)
    elif isinstance(batch_data, dict):
        for value in batch_data.values():
            _record_batch_stream(value, stream)
    elif isinstance(batch_data, (tuple, list)):
        for value in batch_data:
            _record_batch_stream(value, stream)


def _prepare_boards_for_model(boards, use_amp, amp_dtype):
    if use_amp:
        if boards.is_cuda and boards.dtype in (torch.float16, torch.bfloat16):
            return boards.to(dtype=amp_dtype) if boards.dtype != amp_dtype else boards
        return boards
    return boards.float() if boards.dtype != torch.float32 else boards


def _il_hflip_forward_index_map(device):
    global _IL_HFLIP_FORWARD_INDEX_MAP
    if _IL_HFLIP_FORWARD_INDEX_MAP is None:
        inverse = torch.from_numpy(build_hflip_inverse_index_map()).long()
        if int(inverse.numel()) != int(ACTION_SIZE):
            raise ValueError(f"IL horizontal-flip map length mismatch: {inverse.numel()} != {ACTION_SIZE}")
        forward = torch.empty_like(inverse)
        forward[inverse] = torch.arange(inverse.numel(), dtype=inverse.dtype)
        _IL_HFLIP_FORWARD_INDEX_MAP = forward
    return _IL_HFLIP_FORWARD_INDEX_MAP.to(device=device, non_blocking=True)


def _maybe_augment_il_batch(boards, moves, policy_indices, config):
    """Apply exact horizontal board/policy symmetry to a random train subset."""
    aug_cfg = (config.get('imitation_learning', {}) or {}).get('augmentation', {}) or {}
    if not aug_cfg.get('enabled', False):
        return boards, moves, policy_indices
    try:
        probability = float(aug_cfg.get('horizontal_flip_prob', 0.0) or 0.0)
    except (TypeError, ValueError):
        probability = 0.0
    probability = max(0.0, min(1.0, probability))
    if probability <= 0.0 or boards.size(0) <= 0:
        return boards, moves, policy_indices

    flip_mask = torch.rand(boards.size(0), device=boards.device) < probability
    if not bool(flip_mask.any()):
        return boards, moves, policy_indices

    boards[flip_mask] = torch.flip(boards[flip_mask], dims=[3])
    forward_map = _il_hflip_forward_index_map(boards.device)
    if moves is not None:
        valid_moves = (moves >= 0) & (moves < int(ACTION_SIZE))
        active_moves = flip_mask.reshape(-1, *([1] * (moves.dim() - 1))) & valid_moves
        if bool(active_moves.any()):
            moves[active_moves] = forward_map[moves[active_moves].long()].to(dtype=moves.dtype)
    if policy_indices is not None:
        valid_policy = (policy_indices >= 0) & (policy_indices < int(ACTION_SIZE))
        active_policy = flip_mask.unsqueeze(1) & valid_policy
        if bool(active_policy.any()):
            policy_indices[active_policy] = forward_map[policy_indices[active_policy].long()].to(
                dtype=policy_indices.dtype
            )
    return boards, moves, policy_indices


def _iter_prefetched_batches(loader, device, non_blocking=True, enabled=True, queue_size=2):
    """
    Copy CPU batches to CUDA on a background thread and side stream.

    The previous single-threaded prefetch path still had to wait for
    DataLoader.__next__ before yielding the current GPU batch. With expensive IL
    batches that creates visible GPU idle gaps. This version lets the DataLoader
    wait and enqueue host->device copies while the main thread runs the model.
    """
    if not enabled or device.type != 'cuda' or not torch.cuda.is_available():
        for batch_data in loader:
            yield batch_data
        return

    queue_size = max(1, int(queue_size or 1))
    batch_queue = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    sentinel = object()

    def _put(item):
        while not stop_event.is_set():
            try:
                batch_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_worker():
        try:
            prefetch_stream = torch.cuda.Stream(device=device)
            with torch.cuda.device(device):
                for cpu_batch in loader:
                    if stop_event.is_set():
                        break
                    with torch.cuda.stream(prefetch_stream):
                        gpu_batch = _move_batch_to_device(
                            cpu_batch,
                            device,
                            non_blocking=non_blocking,
                        )
                        event = torch.cuda.Event()
                        event.record(prefetch_stream)
                    if not _put((gpu_batch, event, None)):
                        break
        except BaseException as exc:
            _put((None, None, exc))
        finally:
            _put((sentinel, None, None))

    current_stream = torch.cuda.current_stream(device)
    worker = threading.Thread(target=_prefetch_worker, name="cuda-batch-prefetch", daemon=True)
    worker.start()

    try:
        while True:
            batch, event, exc = batch_queue.get()
            if exc is not None:
                raise exc
            if batch is sentinel:
                break
            current_stream.wait_event(event)
            _record_batch_stream(batch, current_stream)
            yield batch
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


def _is_dataloader_worker_failure(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "dataloader worker" in text
        or "exited unexpectedly" in text
        or ("worker" in text and "killed" in text)
        or "broken pipe" in text
        or "eoferror" in text
    )


def _make_prefetched_iterator(loader, device, non_blocking, enabled, queue_size):
    return _iter_prefetched_batches(
        loader,
        device,
        non_blocking=non_blocking,
        enabled=enabled,
        queue_size=queue_size,
    )


def _next_resilient_batch(iterator_state, label):
    iterator = iterator_state['iterator']
    while True:
        try:
            return next(iterator)
        except StopIteration:
            raise
        except BaseException as exc:
            is_worker_failure = _is_dataloader_worker_failure(exc)
            expected_batches = int(iterator_state.get('expected_batches', 0) or 0)
            processed_batches = int(iterator_state.get('processed_batches', 0) or 0)
            remaining_batches = max(0, expected_batches - processed_batches)
            tail_tolerance = max(0, int(iterator_state.get('tail_tolerance_batches', 0) or 0))
            if (
                iterator_state['enabled']
                and is_worker_failure
                and expected_batches > 0
                and 0 < remaining_batches <= tail_tolerance
            ):
                print(
                    f"\n  Warning: {label} DataLoader worker failed with "
                    f"{remaining_batches}/{expected_batches} batches left; "
                    "skipping tail batch(es); workers will be recreated next epoch."
                )
                raise _DataLoaderTailSkip(remaining_batches)

            if (
                not iterator_state['enabled']
                or not is_worker_failure
                or iterator_state['restarts'] >= iterator_state['max_restarts']
            ):
                raise

            iterator_state['restarts'] += 1
            try:
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
            except Exception:
                pass

            print(
                f"\n  Warning: {label} DataLoader worker failed; "
                f"restarting iterator {iterator_state['restarts']}/{iterator_state['max_restarts']}..."
            )
            iterator = _make_prefetched_iterator(
                iterator_state['loader'],
                iterator_state['device'],
                iterator_state['non_blocking'],
                iterator_state['prefetch_enabled'],
                iterator_state['queue_size'],
            )
            iterator_state['iterator'] = iterator


def train_epoch_il(
    model,
    train_loader,
    optimizer,
    scheduler,
    config,
    device,
    scaler,
    epoch=0,
    debug_log_file=None,
    profile=False,
    step_scheduler=True,
    non_blocking_transfer=True,
    progress_callback=None,
):
    """
    đź†• v4.3: Train one epoch with WDL value head and move-weighted losses
    
    Key Changes:
    - Value predictions are now (B, 3) WDL logits
    - Win predictions use move-weighted BCE
    - Simplified loss computation via CombinedLoss
    
    Args:
        model: ChessNet model
        train_loader: DataLoader with training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration dict
        device: torch.device
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        debug_log_file: Path to debug log (optional)
        step_scheduler: Whether to step the provided scheduler at epoch end
        non_blocking_transfer: Use async host->device copies when possible
        progress_callback: Optional cheap callback polled during the epoch
    
    Returns:
        Tuple of (losses_dict, metrics_dict, profile_stats or None)
    """
    model.train()

    debug_cfg = config.get('debug', {}) or {}
    il_debug_cfg = debug_cfg.get('il', {}) or {}
    if not isinstance(il_debug_cfg, dict):
        il_debug_cfg = {}
    debug_enabled = bool(debug_cfg.get('enabled', False))
    log_gpu_memory = bool(
        debug_enabled and il_debug_cfg.get('log_gpu_memory', debug_cfg.get('log_gpu_memory', False))
    )
    log_grad_diagnostics = bool(debug_enabled)
    
    # WDL-only path
    criterion = CombinedLoss(config)
    
    # Accumulators
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_moves_left_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")

    profile_stats = None
    profile_enabled = bool(profile)
    grad_diag_logged = False
    grad_diag = None
    if profile_enabled:
        timers = {
            'data': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'optim': 0.0,
            'metrics': 0.0,
        }
        batch_count = 0

        def _sync():
            if device.type == 'cuda':
                torch.cuda.synchronize()

        data_timer_start = time.perf_counter()

    if device.type == 'cuda' and torch.cuda.is_available() and (profile_enabled or log_gpu_memory):
        torch.cuda.reset_peak_memory_stats(device)
    
    # đź”Ť DIAGNOSTIC: Track target distributions (gated by config)
    first_batch_targets = True
    first_batch_predictions = True
    show_batch0_diagnostics = (
        debug_enabled and
        il_debug_cfg.get('print_batch0_diagnostics', debug_cfg.get('print_batch0_diagnostics', False))
    )
    
    # âšˇ Pre-read AMP config outside loop (avoid dict lookups per batch)
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    non_blocking = bool(non_blocking_transfer and device.type == 'cuda')
    hw_cfg = config.get('hardware', {}) or {}
    cuda_prefetch = bool(hw_cfg.get('cuda_prefetch_batches', True))
    cuda_prefetch_queue_size = max(
        1,
        int(hw_cfg.get('cuda_train_prefetch_queue_size', hw_cfg.get('cuda_prefetch_queue_size', 2)) or 1),
    )
    metrics_interval = max(1, int(config['imitation_learning'].get('train_metrics_interval', 16)))
    progress_interval = max(1, int(config.get('logging', {}).get('print_every', 10)))
    dataloader_restart_enabled = bool(hw_cfg.get('dataloader_restart_on_worker_failure', True))
    dataloader_max_restarts = max(0, int(hw_cfg.get('dataloader_worker_restart_limit', 2) or 0))
    dataloader_tail_tolerance = max(
        0,
        int(hw_cfg.get('dataloader_worker_failure_tail_tolerance_batches', 2) or 0),
    )
    iterator_state = {
        'loader': train_loader,
        'device': device,
        'non_blocking': non_blocking,
        'prefetch_enabled': cuda_prefetch,
        'queue_size': cuda_prefetch_queue_size,
        'enabled': dataloader_restart_enabled,
        'max_restarts': dataloader_max_restarts,
        'restarts': 0,
        'tail_tolerance_batches': dataloader_tail_tolerance,
    }
    iterator_state['iterator'] = _make_prefetched_iterator(
        train_loader,
        device,
        non_blocking,
        cuda_prefetch,
        cuda_prefetch_queue_size,
    )
    processed_batches = 0
    expected_batches = len(train_loader)
    while processed_batches < expected_batches:
        iterator_state['processed_batches'] = processed_batches
        iterator_state['expected_batches'] = expected_batches
        try:
            batch_data = _next_resilient_batch(iterator_state, f"train epoch {epoch}")
        except _DataLoaderTailSkip as exc:
            skipped = max(0, int(getattr(exc, 'skipped_batches', 0) or 0))
            if skipped:
                pbar.update(skipped)
            break
        except StopIteration:
            break

        batch_idx = processed_batches
        if profile_enabled:
            _sync()
            timers['data'] += time.perf_counter() - data_timer_start
        # Unpack batch
        if isinstance(batch_data, dict):
            boards = batch_data['board']
            moves = batch_data['move']
            outcomes = batch_data['value']
            # đź”§ v4.4 FIX: Get move_idx (not move_indices) from dict
            move_indices = batch_data.get('move_idx', None)
            total_moves = batch_data.get('total_moves', None)
            policy_indices = batch_data.get('policy_indices', None)
            policy_values = batch_data.get('policy_values', None)
            value_wdl = batch_data.get('value_wdl', None)
            occurrence_count = batch_data.get('occurrence_count', None)
            sample_weight = batch_data.get('sample_weight', None)
            value_sample_weight = batch_data.get('value_sample_weight', None)
            moves_left_log = batch_data.get('moves_left_log', None)
            policy_mass_kept = batch_data.get('policy_mass_kept', None)

        else:
            # đź”§ v4.4 FIX: Unpack move_indices from tuple
            if len(batch_data) == 5:
                boards, moves, outcomes, move_indices, total_moves = batch_data
            elif len(batch_data) == 4:
                boards, moves, outcomes, move_indices = batch_data
                total_moves = None
            else:
                boards, moves, outcomes = batch_data
                move_indices = None
            total_moves = None
            policy_indices = None
            policy_values = None
            value_wdl = None
            occurrence_count = None
            sample_weight = None
            value_sample_weight = None
            moves_left_log = None
            policy_mass_kept = None

        
        # Move to device. With CUDA prefetch enabled these are already on GPU,
        # so the calls below are cheap no-ops that keep the non-prefetch path simple.
        boards = boards.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
        boards = _prepare_boards_for_model(boards, use_amp, amp_dtype)
        moves = moves.to(device, non_blocking=non_blocking)
        outcomes = outcomes.to(device, non_blocking=non_blocking)
        
        if move_indices is not None:
            move_indices = move_indices.to(device, non_blocking=non_blocking)
        if total_moves is not None:
            total_moves = total_moves.to(device, non_blocking=non_blocking)
        if policy_indices is not None:
            policy_indices = policy_indices.to(device, non_blocking=non_blocking)
        if policy_values is not None:
            policy_values = policy_values.to(device, non_blocking=non_blocking)
        if value_wdl is not None:
            value_wdl = value_wdl.to(device, non_blocking=non_blocking)
        if occurrence_count is not None:
            occurrence_count = occurrence_count.to(device, non_blocking=non_blocking)
        if sample_weight is not None:
            sample_weight = sample_weight.to(device, non_blocking=non_blocking)
        if value_sample_weight is not None:
            value_sample_weight = value_sample_weight.to(device, non_blocking=non_blocking)
        if moves_left_log is not None:
            moves_left_log = moves_left_log.to(device, non_blocking=non_blocking)
        if policy_mass_kept is not None:
            policy_mass_kept = policy_mass_kept.to(device, non_blocking=non_blocking)

        boards, moves, policy_indices = _maybe_augment_il_batch(
            boards,
            moves,
            policy_indices,
            config,
        )
        

        # đź”Ť DIAGNOSTIC: Print distributions for first batch
        if show_batch0_diagnostics and first_batch_targets:
            print(f"\nđź”Ť DIAGNOSTIC - Batch 0:")
            print(f"  Outcome targets (Value):")
            print(f"    Min: {outcomes.min().item():.3f}, Max: {outcomes.max().item():.3f}, Mean: {outcomes.mean().item():.3f}")
            print(f"    Unique values: {torch.unique(outcomes).cpu().numpy()[:10]}")  # First 10 unique
            print(f"    Distribution: +1: {(outcomes > 0.9).sum().item()}, 0: {(outcomes.abs() < 0.1).sum().item()}, -1: {(outcomes < -0.9).sum().item()}")
            
            first_batch_targets = False
        
        optimizer.zero_grad(set_to_none=True)

        if profile_enabled:
            _sync()
            t0 = time.perf_counter()
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            policy_pred, value_pred, moves_left_pred = model(
                boards,
                apply_log_softmax=False,
                return_moves_left=True,
            )
            
            # đź†• v4.3: Compute loss using CombinedLoss
            # Pack predictions and targets for CombinedLoss
            predictions = {
                'policy': policy_pred,
                'moves_left': moves_left_pred,
                'value': value_pred,  # đź†• Now (B, 3) WDL logits!
            }
                
            targets = {
                'moves': moves,
                'values': outcomes,  # Still scalar {-1, 0, +1}
            }
            if move_indices is not None:
                targets['move_indices'] = move_indices
            if policy_indices is not None and policy_values is not None:
                targets['policy_indices'] = policy_indices
                targets['policy_values'] = policy_values
            if value_wdl is not None:
                targets['value_wdl'] = value_wdl
            if sample_weight is not None:
                targets['sample_weight'] = sample_weight
            if value_sample_weight is not None:
                targets['value_sample_weight'] = value_sample_weight
            if moves_left_log is not None:
                targets['moves_left_log'] = moves_left_log
            if total_moves is not None:
                targets['total_moves'] = total_moves
                if move_indices is not None:
                    if moves_left_log is None:
                        targets['moves_left'] = torch.clamp(
                            total_moves.reshape(-1).to(dtype=torch.float32)
                            - move_indices.reshape(-1).to(dtype=torch.float32),
                            min=0.0,
                        )
            # đź”Ť DIAGNOSTIC: Print WDL predictions for first batch
            if show_batch0_diagnostics and first_batch_predictions and batch_idx == 0:
                wdl_probs = torch.softmax(value_pred[:10], dim=1)
                value_scalars = (wdl_probs[:, 0] * 1.0 + 
                                wdl_probs[:, 1] * 0.0 + 
                                wdl_probs[:, 2] * (-1.0))
                print(f"\n  đź”Ť WDL Predictions (first 10 samples):")
                print(f"    Value pred shape: {value_pred.shape}")
                print(f"    Sample logits: {value_pred[:3].float().cpu().detach().numpy()}")
                print(f"    WDL Probs (Win/Draw/Loss) -> Scalar vs Target:")
                for i in range(10):
                    print(f"      [{i}] W:{wdl_probs[i,0]:.3f} D:{wdl_probs[i,1]:.3f} L:{wdl_probs[i,2]:.3f} "
                          f"-> Scalar:{value_scalars[i]:.3f} | Target: {outcomes[i].item():+.3f}")
                print(f"    Mean predicted scalar: {value_scalars.mean().item():.3f}")
                print(f"    Mean target: {outcomes[:10].mean().item():.3f}\n")
                first_batch_predictions = False
                

            # Single loss computation
            loss, loss_dict = criterion(predictions, targets)

        if profile_enabled:
            _sync()
            timers['forward'] += time.perf_counter() - t0
            t0 = time.perf_counter()

        # Backward pass
        scaler.scale(loss).backward()

        if profile_enabled:
            _sync()
            timers['backward'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Gradient clipping
        grad_clip = config['imitation_learning'].get('grad_clip', 1.0)
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if log_grad_diagnostics and not grad_diag_logged:
            trainable_tensors = 0
            grad_tensors = 0
            trainable_params = 0
            grad_params = 0
            missing_grad_names = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                trainable_tensors += 1
                trainable_params += int(param.numel())
                if param.grad is not None:
                    grad_tensors += 1
                    grad_params += int(param.numel())
                elif len(missing_grad_names) < 8:
                    missing_grad_names.append(name)

            grad_diag = {
                'batch_idx': int(batch_idx),
                'trainable_tensors': int(trainable_tensors),
                'grad_tensors': int(grad_tensors),
                'trainable_params': int(trainable_params),
                'grad_params': int(grad_params),
                'missing_grad_names': missing_grad_names,
            }

            if device.type == 'cuda' and torch.cuda.is_available() and log_gpu_memory:
                grad_diag.update({
                    'peak_allocated_mb': float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
                    'peak_reserved_mb': float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
                    'current_allocated_mb': float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
                    'current_reserved_mb': float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
                })
            grad_diag_logged = True
        
        scaler.step(optimizer)
        scaler.update()
        
        if profile_enabled:
            _sync()
            timers['optim'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Update train metrics on a configurable sample of batches. Validation
        # still computes full metrics; this keeps the hot training loop lighter.
        if batch_idx % metrics_interval == 0 or batch_idx == len(train_loader) - 1:
            metric_moves = moves
            if policy_indices is not None:
                soft_top_move = policy_indices[:, 0].to(dtype=moves.dtype)
                metric_moves = torch.where(soft_top_move >= 0, soft_top_move, moves)
            metric_outcomes = outcomes
            if value_wdl is not None:
                metric_outcomes = (value_wdl[:, 0] - value_wdl[:, 2]).reshape(-1, 1)
            metrics_calc.update(
                policy_pred,
                value_pred,
                metric_moves,
                metric_outcomes,
                move_indices=move_indices,
                total_moves=total_moves,
                value_max_moves=config['data'].get('max_moves_per_game', 200),
                policy_is_logits=True,
                target_policy_indices=policy_indices,
                target_policy_values=policy_values,
                target_wdl=value_wdl,
            )
            if occurrence_count is not None:
                metrics_calc.update_soft_target_stats(
                    occurrence_count=occurrence_count,
                    sample_weight=sample_weight,
                    policy_mass_kept=policy_mass_kept,
                )
            metrics_calc.update_value_phase_loss(
                value_pred,
                target_value=outcomes,
                target_wdl=value_wdl,
                move_indices=move_indices,
                sample_weight=value_sample_weight if value_sample_weight is not None else sample_weight,
            )
            metrics_calc.update_moves_left_metrics(
                moves_left_pred,
                target_moves_left_log=moves_left_log,
                move_indices=move_indices,
                total_moves=total_moves,
                sample_weight=sample_weight,
            )
        
        # Accumulate losses
        total_loss += loss_dict['total']
        total_policy_loss += loss_dict['policy']
        total_value_loss += loss_dict['value']
        total_moves_left_loss += loss_dict.get('moves_left', 0.0)
        
        # Update progress bar less often; formatting and terminal writes are
        # surprisingly expensive with very large IL batches.
        if batch_idx % progress_interval == 0 or batch_idx == len(train_loader) - 1:
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'policy': f'{total_policy_loss / (batch_idx + 1):.4f}',
                'value': f'{total_value_loss / (batch_idx + 1):.4f}',
            })
            if progress_callback is not None:
                progress_callback()
        pbar.update(1)
        processed_batches += 1

        if profile_enabled:
            _sync()
            timers['metrics'] += time.perf_counter() - t0
            batch_count += 1
            data_timer_start = time.perf_counter()
    pbar.close()
    
    # Step scheduler once per epoch (if enabled by caller).
    if scheduler is not None and step_scheduler:
        scheduler.step()
    
    # Compute final metrics
    n = max(1, processed_batches)
    
    losses = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n,
        'moves_left': total_moves_left_loss / n,
    }
    
    metrics = metrics_calc.compute()

    if profile_enabled:
        total_profile_time = sum(timers.values())
        profile_stats = {
            'data': timers['data'],
            'forward': timers['forward'],
            'backward': timers['backward'],
            'optim': timers['optim'],
            'metrics': timers['metrics'],
            'total': total_profile_time,
            'batches': max(1, batch_count),
        }

    if grad_diag is not None:
        if profile_stats is None:
            profile_stats = {}
        profile_stats['grad_diag'] = grad_diag

    if device.type == 'cuda' and torch.cuda.is_available() and (profile_enabled or log_gpu_memory):
        if profile_stats is None:
            profile_stats = {}
        profile_stats.update({
            'peak_allocated_mb': float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            'peak_reserved_mb': float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
            'current_allocated_mb': float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
            'current_reserved_mb': float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        })
    
    return losses, metrics, profile_stats


def evaluate_il(model, val_loader, config, device, non_blocking_transfer=True):
    """
    Evaluate model with WDL value head
    
    Identical to train_epoch_il but without gradient updates
    """
    model.eval()
    
    # WDL-only path
    criterion = CombinedLoss(config)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_moves_left_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    # Pre-read AMP config outside loop
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    non_blocking = bool(non_blocking_transfer and device.type == 'cuda')
    hw_cfg = config.get('hardware', {}) or {}
    cuda_prefetch = bool(hw_cfg.get('cuda_prefetch_batches', True))
    cuda_prefetch_queue_size = max(
        1,
        int(hw_cfg.get('cuda_eval_prefetch_queue_size', hw_cfg.get('cuda_prefetch_queue_size', 2)) or 1),
    )
    metrics_interval = max(1, int(config['imitation_learning'].get('eval_metrics_interval', 32)))
    
    with torch.inference_mode():
        eval_pbar = tqdm(total=len(val_loader), desc="Evaluating")
        dataloader_restart_enabled = bool(hw_cfg.get('dataloader_restart_on_worker_failure', True))
        dataloader_max_restarts = max(0, int(hw_cfg.get('dataloader_worker_restart_limit', 2) or 0))
        dataloader_tail_tolerance = max(
            0,
            int(hw_cfg.get('dataloader_worker_failure_tail_tolerance_batches', 2) or 0),
        )
        eval_iterator_state = {
            'loader': val_loader,
            'device': device,
            'non_blocking': non_blocking,
            'prefetch_enabled': cuda_prefetch,
            'queue_size': cuda_prefetch_queue_size,
            'enabled': dataloader_restart_enabled,
            'max_restarts': dataloader_max_restarts,
            'restarts': 0,
            'tail_tolerance_batches': dataloader_tail_tolerance,
        }
        eval_iterator_state['iterator'] = _make_prefetched_iterator(
            val_loader,
            device,
            non_blocking,
            cuda_prefetch,
            cuda_prefetch_queue_size,
        )
        processed_batches = 0
        expected_batches = len(val_loader)
        while processed_batches < expected_batches:
            eval_iterator_state['processed_batches'] = processed_batches
            eval_iterator_state['expected_batches'] = expected_batches
            try:
                batch_data = _next_resilient_batch(eval_iterator_state, "validation")
            except _DataLoaderTailSkip as exc:
                skipped = max(0, int(getattr(exc, 'skipped_batches', 0) or 0))
                if skipped:
                    eval_pbar.update(skipped)
                break
            except StopIteration:
                break

            batch_idx = processed_batches
            if isinstance(batch_data, dict):
                boards = batch_data['board']
                moves = batch_data['move']
                outcomes = batch_data['value']
                move_indices = batch_data.get('move_idx', None)
                total_moves = batch_data.get('total_moves', None)
                policy_indices = batch_data.get('policy_indices', None)
                policy_values = batch_data.get('policy_values', None)
                value_wdl = batch_data.get('value_wdl', None)
                occurrence_count = batch_data.get('occurrence_count', None)
                sample_weight = batch_data.get('sample_weight', None)
                value_sample_weight = batch_data.get('value_sample_weight', None)
                moves_left_log = batch_data.get('moves_left_log', None)
                policy_mass_kept = batch_data.get('policy_mass_kept', None)
            else:
                if len(batch_data) == 5:
                    boards, moves, outcomes, move_indices, total_moves = batch_data
                elif len(batch_data) == 4:
                    boards, moves, outcomes, move_indices = batch_data
                    total_moves = None
                else:
                    boards, moves, outcomes = batch_data
                    move_indices = None
                total_moves = None
                policy_indices = None
                policy_values = None
                value_wdl = None
                occurrence_count = None
                sample_weight = None
                value_sample_weight = None
                moves_left_log = None
                policy_mass_kept = None
            
            boards = boards.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
            boards = _prepare_boards_for_model(boards, use_amp, amp_dtype)
            moves = moves.to(device, non_blocking=non_blocking)
            outcomes = outcomes.to(device, non_blocking=non_blocking)
            
            if move_indices is not None:
                move_indices = move_indices.to(device, non_blocking=non_blocking)
            if total_moves is not None:
                total_moves = total_moves.to(device, non_blocking=non_blocking)
            if policy_indices is not None:
                policy_indices = policy_indices.to(device, non_blocking=non_blocking)
            if policy_values is not None:
                policy_values = policy_values.to(device, non_blocking=non_blocking)
            if value_wdl is not None:
                value_wdl = value_wdl.to(device, non_blocking=non_blocking)
            if occurrence_count is not None:
                occurrence_count = occurrence_count.to(device, non_blocking=non_blocking)
            if sample_weight is not None:
                sample_weight = sample_weight.to(device, non_blocking=non_blocking)
            if value_sample_weight is not None:
                value_sample_weight = value_sample_weight.to(device, non_blocking=non_blocking)
            if moves_left_log is not None:
                moves_left_log = moves_left_log.to(device, non_blocking=non_blocking)
            if policy_mass_kept is not None:
                policy_mass_kept = policy_mass_kept.to(device, non_blocking=non_blocking)
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                policy_pred, value_pred, moves_left_pred = model(
                    boards,
                    apply_log_softmax=False,
                    return_moves_left=True,
                )
                
                predictions = {
                    'policy': policy_pred,
                    'moves_left': moves_left_pred,
                    'value': value_pred,
                }
                    
                targets = {
                    'moves': moves,
                    'values': outcomes,
                }
                if move_indices is not None:
                    targets['move_indices'] = move_indices
                if policy_indices is not None and policy_values is not None:
                    targets['policy_indices'] = policy_indices
                    targets['policy_values'] = policy_values
                if value_wdl is not None:
                    targets['value_wdl'] = value_wdl
                if sample_weight is not None:
                    targets['sample_weight'] = sample_weight
                if value_sample_weight is not None:
                    targets['value_sample_weight'] = value_sample_weight
                if moves_left_log is not None:
                    targets['moves_left_log'] = moves_left_log
                if total_moves is not None:
                    targets['total_moves'] = total_moves
                    if move_indices is not None:
                        if moves_left_log is None:
                            targets['moves_left'] = torch.clamp(
                                total_moves.reshape(-1).to(dtype=torch.float32)
                                - move_indices.reshape(-1).to(dtype=torch.float32),
                                min=0.0,
                            )
                    
                loss, loss_dict = criterion(predictions, targets)
            
            total_loss += loss_dict['total']
            total_policy_loss += loss_dict['policy']
            total_value_loss += loss_dict['value']
            total_moves_left_loss += loss_dict.get('moves_left', 0.0)
            
            if batch_idx % metrics_interval == 0 or batch_idx == len(val_loader) - 1:
                metrics_calc.update(
                    policy_pred,
                    value_pred,
                    torch.where(
                        policy_indices[:, 0].to(dtype=moves.dtype) >= 0,
                        policy_indices[:, 0].to(dtype=moves.dtype),
                        moves,
                    ) if policy_indices is not None else moves,
                    (value_wdl[:, 0] - value_wdl[:, 2]).reshape(-1, 1) if value_wdl is not None else outcomes,
                    move_indices=move_indices,
                    total_moves=total_moves,
                    value_max_moves=config['data'].get('max_moves_per_game', 200),
                    policy_is_logits=True,
                    target_policy_indices=policy_indices,
                    target_policy_values=policy_values,
                    target_wdl=value_wdl,
                )
                if occurrence_count is not None:
                    metrics_calc.update_soft_target_stats(
                        occurrence_count=occurrence_count,
                        sample_weight=sample_weight,
                        policy_mass_kept=policy_mass_kept,
                    )
                metrics_calc.update_value_phase_loss(
                    value_pred,
                    target_value=outcomes,
                    target_wdl=value_wdl,
                    move_indices=move_indices,
                    sample_weight=value_sample_weight if value_sample_weight is not None else sample_weight,
                )
                metrics_calc.update_moves_left_metrics(
                    moves_left_pred,
                    target_moves_left_log=moves_left_log,
                    move_indices=move_indices,
                    total_moves=total_moves,
                    sample_weight=sample_weight,
                )
            eval_pbar.update(1)
            processed_batches += 1
        eval_pbar.close()
    
    n = max(1, processed_batches)
    
    losses = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n,
        'moves_left': total_moves_left_loss / n,
    }
    
    metrics = metrics_calc.compute()
    
    return losses, metrics
