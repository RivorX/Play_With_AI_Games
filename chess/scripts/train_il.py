"""
Imitation Learning Training Script - v4.5
🆕 v4.5: CRITICAL FIXES - Per-Game Split + Promotions + Better Sampling
🆕 v4.5: CHESS METADATA - Castling, En Passant, Halfmove, Fullmove (16 planes)
🆕 v4.3: WDL VALUE HEAD + TEMPORAL DISCOUNTING
🆕 v4.2: POV + Dynamic Sliding Window
- 🎯 POV: All boards from current player's perspective (flip for black)
- 🔄 Sliding Window: Dynamic history assembly at load time using mmap
- 🎮 GameID tracking: Efficient history reconstruction across positions
- 📊 Chess Metadata: 4 extra planes (castling, en passant, halfmove, fullmove)
- ⚡ WDL: Win/Draw/Loss classification (stronger signal than MSE)
- ⚡ TEMPORAL DISCOUNTING: DISABLED (WDL-only mode, clean ±1.0 targets)
- 🔒 Per-Game Split: Train/Val separated by games (no history leakage)
- 🎲 Per-Game Stride Offset: Per-game offset for unbiased sampling
- 👑 Promotions: Promotion-aware action space (see ACTION_SIZE)
"""

import torch
import torch.optim as optim
import yaml
import sys
import time
import math
from pathlib import Path
import numpy as np
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import (
    ChessNet,
    save_checkpoint,
)
from src.data import process_pgn_files, create_dataloaders
from src.utils.data_helpers import ACTION_SIZE

# Import from utils
from utils.shared.logger import TrainingLogger
from utils.il.training_il import train_epoch_il, evaluate_il
from utils.il.startup import (
    plan_il_startup,
    apply_il_startup_plan,
    ask_resume_additional_epochs,
    ask_il_hyperparam_source,
    ask_il_start_mode,
    has_il_checkpoints,
)
from utils.il.auto_tune import resolve_il_hyperparameters
from utils.il.elo_async import ILEloCoordinator
from utils.il.checkpointing import (
    build_runtime_state,
    save_swa_snapshot_checkpoint,
    finalize_swa_model,
)
from utils.shared.runtime_helpers import (
    build_model_file_tag,
    build_model_architecture_metadata,
    cleanup_interrupted_log_csv,
    run_with_optional_stdout_suppression,
)
from utils.shared.model_view import (
    print_active_model_summary,
    print_status_table,
    print_multi_column_table,
)


_LAST_RUN_LOG_CSV = None
_LAST_RUN_LOG_PNG = None


def main():
    # Load config
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_version = config.get('model', {}).get('version', 'v?.?')
    model_file_tag = build_model_file_tag(config)
    model_architecture = build_model_architecture_metadata(config)
    debug_enabled = config.get('debug', {}).get('enabled', False)

    # Default to compact model logging in IL unless debug mode is enabled.
    config.setdefault('model', {})
    # Keep ChessNet constructor quiet; startup summary is printed via shared table view.
    config['model']['print_summary'] = False
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("✓ TF32 + cuDNN benchmark enabled")
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Check AMP
    use_amp = config['hardware'].get('use_amp', True)
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    
    if use_amp and not torch.cuda.is_available():
        print("⚠️ AMP requested but CUDA not available, disabling AMP")
        use_amp = False
        use_bfloat16 = False
    
    if use_amp:
        amp_dtype = "bfloat16" if use_bfloat16 else "float16"
        print(f"✓ Mixed Precision Training (AMP) enabled with {amp_dtype}")
        
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            print("⚠️ bfloat16 requested but not supported, falling back to float16")
            use_bfloat16 = False
            config['hardware']['use_bfloat16'] = False
    
    # Create directories
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    il_dir = base_dir / config['paths']['il_checkpoints_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    il_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = base_dir / config['paths']['best_model_il']
    
    # Get configuration
    use_mtl = config['model'].get('use_multitask_learning', False)
    history_positions = config['model'].get('history_positions', 0)
    stride = config['data'].get('sliding_window_stride', 1)
    
    # 🆕 v4.5 FIXED: Calculate expected input planes with chess metadata
    expected_input_planes = 16 * (1 + history_positions)  # 🔧 FIXED: 16 planes (12 pieces + 4 metadata)
    
    print(
        f"IL setup: version={model_version}, blocks={config['model']['num_residual_blocks']}, "
        f"filters={config['model']['filters']}, history={history_positions}, stride={stride}, "
        f"input_planes={expected_input_planes}"
    )
    print(
        f"Features: metadata=on, promotions={ACTION_SIZE}, split_by_game=on, "
        f"wdl_value=on, mtl={use_mtl}"
    )

    # Setup debug logging
    debug_log_file = None
    
    # Create model early so startup menu appears before any data processing.
    print("\nPreparing model for startup menu...")

    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    print("Model ready")

    hparam_mode = ask_il_hyperparam_source(default_mode="config")
    selected_start_mode = ask_il_start_mode(
        has_checkpoints=has_il_checkpoints(best_model_path, il_dir)
    )

    hparam_resolution = resolve_il_hyperparameters(
        config=config,
        model=model,
        device=device,
        base_dir=base_dir,
        mode=hparam_mode,
    )

    model_hash_short = str(hparam_resolution.get("model_hash") or "n/a")[:16]
    print_status_table(
        "IL Hyperparameters",
        [
            ("Source", hparam_resolution.get("source", "config")),
            ("Batch size", config['imitation_learning']['batch_size']),
            ("Learning rate", f"{float(config['imitation_learning']['learning_rate']):.6g}"),
            ("Model hash", model_hash_short),
        ],
    )

    if debug_enabled:
        debug_dir = logs_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_file = debug_dir / f"training_profile_{timestamp}.txt"

        print(
            f"Debug mode: profile={config['debug'].get('profile_training', False)}, "
            f"gpu_mem_log={config['debug'].get('log_gpu_memory', False)}, "
            f"every={config['debug'].get('profile_every_n_epochs', 1)} ep, "
            f"log={debug_log_file}"
        )

        with open(debug_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"🐛 TRAINING DEBUG LOG - {model_version} Chess Metadata + WDL\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(
                f"Model: {config['model']['filters']} filters, "
                f"{config['model']['num_residual_blocks']} blocks\n"
            )
            f.write(f"Model hash: {hparam_resolution.get('model_hash', 'n/a')}\n")
            f.write(f"Hyperparameter source: {hparam_resolution.get('source', 'config')}\n")
            f.write(f"Batch size: {config['imitation_learning']['batch_size']}\n")
            f.write(f"Learning rate: {config['imitation_learning']['learning_rate']}\n")
            f.write(f"History positions: {history_positions} (dynamic)\n")
            f.write(f"Sliding window stride: {stride}x\n")
            f.write(f"Input planes: {expected_input_planes} (16 per position)\n")
            f.write("Chess metadata: Castling, En Passant, Halfmove, Fullmove\n")
            f.write(f"MTL enabled: {use_mtl}\n")
            f.write("WDL Value: True (forced)\n")
            f.write("POV enabled: True\n")
            if hparam_resolution.get("mode") == "auto":
                f.write(
                    f"Auto-tune cache: {hparam_resolution.get('cache_path', 'n/a')} "
                    f"(hit={hparam_resolution.get('cache_hit', False)})\n"
                )
                dedicated_vram = hparam_resolution.get("dedicated_vram_bytes")
                if dedicated_vram:
                    f.write(f"Dedicated VRAM: {dedicated_vram}\n")
                budget_bytes = hparam_resolution.get("budget_bytes")
                if budget_bytes:
                    f.write(f"Target VRAM budget: {budget_bytes}\n")
                estimated_peak = hparam_resolution.get("estimated_peak_bytes")
                if estimated_peak:
                    f.write(f"Estimated train peak: {estimated_peak}\n")
            f.write("=" * 70 + "\n\n")

    startup_plan = plan_il_startup(
        model=model,
        device=device,
        base_dir=base_dir,
        best_model_path=best_model_path,
        il_dir=il_dir,
        start_mode=selected_start_mode,
    )

    # For resume mode, allow extending training by additional epochs.
    il_epochs_cfg = int(config['imitation_learning']['epochs'])
    if startup_plan.get("start_mode") == "resume":
        selected_entry = startup_plan.get("selected_entry") or {}
        checkpoint_epoch = selected_entry.get("epoch")
        if checkpoint_epoch is not None:
            completed_epochs = int(checkpoint_epoch) + 1
            default_additional = max(0, il_epochs_cfg - completed_epochs)
            additional_epochs = ask_resume_additional_epochs(completed_epochs, default_additional)
            config['imitation_learning']['epochs'] = completed_epochs + additional_epochs
            print(
                f"Resume target: completed={completed_epochs}, "
                f"additional={additional_epochs}, total_target={config['imitation_learning']['epochs']}"
            )

    # Initialize logger after startup menu selection.
    logger = TrainingLogger(
        logs_dir,
        experiment_name=f"il_training_{model_version}",
        mode="il",
        use_mtl=use_mtl
    )
    global _LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG
    _LAST_RUN_LOG_CSV = logger.csv_path
    _LAST_RUN_LOG_PNG = logger.plot_path
    
    # 🆕 Per-layer learning rates - value head with lower LR to prevent overfitting
    value_head_lr_factor = config['imitation_learning'].get('value_head_lr_factor', 1.0)
    base_lr = config['imitation_learning']['learning_rate']
    
    if value_head_lr_factor != 1.0:
        # Separate value head parameters
        value_head_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'value_' in name:  # value_conv1, value_conv2, value_bn, value_fc1, value_fc2
                value_head_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': base_lr},
            {'params': value_head_params, 'lr': base_lr * value_head_lr_factor}
        ]
        
        print(
            f"Per-layer LR: trunk/policy={base_lr:.4f}, "
            f"value={base_lr * value_head_lr_factor:.4f} ({value_head_lr_factor}x)"
        )
    else:
        param_groups = model.parameters()
    
    # Optimizer
    optimizer = optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=config['imitation_learning']['weight_decay'],
        fused=True if torch.cuda.is_available() else False
    )
    
    if torch.cuda.is_available():
        print("✓ Using fused AdamW optimizer")
    
    # Learning rate scheduler: linear warmup → cosine decay
    total_epochs = config['imitation_learning']['epochs']
    warmup_pct = config['imitation_learning'].get('warmup_pct', 0.05)
    min_lr_ratio = config['imitation_learning'].get('min_lr_ratio', 0.05)

    warmup_epochs = max(1, int(round(total_epochs * warmup_pct)))

    def _lr_lambda(epoch):
        # Linear warmup: 0 → 1 over warmup_epochs
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Cosine decay: 1 → min_lr_ratio
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    print(
        f"✓ LR schedule: warmup={warmup_epochs} ep → "
        f"cosine decay (min_lr_ratio={min_lr_ratio})"
    )
    
    # AMP Gradient Scaler
    # 🔧 v4.8: GradScaler only for float16, NOT for bfloat16 (same dynamic range as float32)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and not use_bfloat16)

    il_cfg = config.get('imitation_learning', {})
    save_optimizer_state = il_cfg.get('save_optimizer_state', True)

    startup_state = apply_il_startup_plan(
        startup_plan=startup_plan,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    start_mode = startup_state['start_mode']
    selected_checkpoint = startup_state['selected_checkpoint']
    selected_checkpoint_label = startup_state['selected_checkpoint_label']
    start_epoch = startup_state['start_epoch']
    best_val_loss = startup_state['best_val_loss']
    patience_counter = startup_state['patience_counter']
    resumed_estimated_elo = startup_state.get('estimated_elo')
    resumed_estimated_elo_epoch = startup_state.get('estimated_elo_epoch')
    selected_compatibility_ratio = startup_state.get('selected_compatibility_ratio')
    transfer_match_ratio = startup_state.get('transfer_match_ratio')
    transfer_freeze_epochs = int(startup_state.get('transfer_freeze_epochs', 0) or 0)
    transfer_trainable_param_names = startup_state.get('transfer_trainable_param_names') or []
    selected_entry = startup_plan.get("selected_entry") or {}

    source_label = "new (scratch)"
    if start_mode in {"resume", "transfer"}:
        source_label = "selected checkpoint"
        if selected_checkpoint_label:
            source_label = Path(selected_checkpoint_label).name

    print_active_model_summary(
        model,
        config,
        title="Active Model (IL)",
        source_label=source_label,
        startup_mode=start_mode,
        checkpoint_label=selected_checkpoint_label,
        device=device,
        selected_entry=selected_entry,
    )

    if resumed_estimated_elo is not None:
        seed_epoch = resumed_estimated_elo_epoch
        if seed_epoch is None:
            seed_epoch = max(1, start_epoch)
        try:
            logger.record_estimated_elo(int(seed_epoch), float(resumed_estimated_elo), update_csv=False)
            print(f"Resume Elo seed: {int(round(float(resumed_estimated_elo)))} (epoch {int(seed_epoch)})")
        except (TypeError, ValueError):
            pass

    if start_mode == "new":
        plot_run_context = "startup: new training from scratch"
    elif start_mode == "resume":
        plot_run_context = f"startup: resumed full state, next epoch {start_epoch + 1}"
    else:
        transfer_ratio = transfer_match_ratio
        if transfer_ratio is None:
            transfer_ratio = selected_compatibility_ratio
        if transfer_ratio is None:
            plot_run_context = "startup: transferred matching weights, compatibility n/a"
        else:
            plot_run_context = (
                "startup: transferred matching weights, "
                f"compatibility {transfer_ratio * 100:.2f}%"
            )
        if transfer_freeze_epochs > 0 and transfer_trainable_param_names:
            plot_run_context += (
                f" | freeze changed-only for {transfer_freeze_epochs} ep"
            )

    if selected_checkpoint_label and start_mode in {"resume", "transfer"}:
        plot_run_context = f"{plot_run_context} | source: {Path(selected_checkpoint_label).name}"
    logger.set_run_context(plot_run_context)

    if start_mode == "resume" and start_epoch >= config['imitation_learning']['epochs']:
        print(f"ℹ️ Checkpoint already at epoch {start_epoch}, no epochs left to run.")
        return

    # Load data with multi-phase processing
    data_dir = script_dir.parent / config['paths']['data_dir']
    pgn_files = sorted(data_dir.glob('*.pgn'))

    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_dir}")

    print_status_table(
        "Loading Data (IL)",
        [
            ("Data dir", str(data_dir)),
            ("PGN files", f"{len(pgn_files):,}"),
            ("History positions", history_positions),
            ("Sliding stride", stride),
            ("MTL", "on" if use_mtl else "off"),
            ("Debug mode", "on" if debug_enabled else "off"),
        ],
    )
    if debug_enabled:
        for pgn in pgn_files:
            print(f"  - {pgn.name}")

    # Convert relative paths in config to absolute paths to keep preprocessing under chess/data.
    config_with_absolute_paths = config.copy()
    config_with_absolute_paths['paths'] = config['paths'].copy()
    config_with_absolute_paths['paths']['data_dir'] = str(data_dir.absolute())

    metadata = run_with_optional_stdout_suppression(
        debug_enabled,
        process_pgn_files,
        pgn_files,
        config_with_absolute_paths,
    )

    # Verify binary format compatibility.
    actual_position_size = metadata.get('position_size')
    expected_position_size = 50 if not use_mtl else 62

    if actual_position_size != expected_position_size:
        print(f"\n⚠️ WARNING: Position size mismatch!")
        print(f"  • Expected: {expected_position_size} bytes")
        print(f"  • Actual: {actual_position_size} bytes")
        print(f"  • This may indicate the data was preprocessed with an old format or different MTL setting")
        
        if actual_position_size == 50 and use_mtl:
            print(f"  ❌ Data was processed WITHOUT MTL, but config has use_multitask_learning=True")
            print(f"     Please either:")
            print(f"     1. Set use_multitask_learning=False in config.yaml, OR")
            print(f"     2. Delete cache and reprocess data with MTL enabled")
            raise ValueError("MTL mismatch between data and config")
        elif actual_position_size == 62 and not use_mtl:
            print(f"  ❌ Data was processed WITH MTL, but config has use_multitask_learning=False")
            print(f"     Please either:")
            print(f"     1. Set use_multitask_learning=True in config.yaml, OR")
            print(f"     2. Delete cache and reprocess data without MTL")
            raise ValueError("MTL mismatch between data and config")
        elif actual_position_size in [48, 60]:
            print(f"  Data was processed with OLD v4.4 format (36B board, no fullmove metadata)")
            print(f"  {model_version} adds fullmove number (38B board)")
            print(f"     Please delete cache (data/preprocessing/) and reprocess with {model_version}")
            raise ValueError("Old data format - please reprocess")
        elif actual_position_size in [44, 56]:
            print(f"  ❌ Data was processed with OLD v4.3 format (32B board, no chess metadata)")
            print(f"     🆕 {model_version} uses 38B board with castling, en passant, halfmove, fullmove")
            print(f"     Please delete cache (data/preprocessing/) and reprocess with {model_version}")
            raise ValueError("Old data format - please reprocess")
        else:
            print(f"  ❌ Unknown format mismatch - please delete cache and reprocess")
            raise ValueError("Position size mismatch")

    train_loader, val_loader = run_with_optional_stdout_suppression(
        debug_enabled,
        create_dataloaders,
        metadata,
        config,
    )

    train_stats = getattr(train_loader.dataset, 'filter_stats', {}) or {}
    val_stats = getattr(val_loader.dataset, 'filter_stats', {}) or {}
    sampling_cfg = config.get('data', {}).get('position_sampling', {})
    sampling_enabled = bool(sampling_cfg.get('enabled', False))

    def _as_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _stat(stats, key, default=0):
        return _as_int(stats.get(key, default), default=default)

    def _fmt_count(value):
        return f"{_as_int(value):,}"

    def _fmt_cut(removed, base):
        removed = max(0, _as_int(removed))
        base = max(0, _as_int(base))
        pct = (removed * 100.0 / base) if base > 0 else 0.0
        return f"{removed:,} ({pct:.2f}%)"

    train_before = _stat(train_stats, 'input_count', len(train_loader.dataset))
    val_before = _stat(val_stats, 'input_count', len(val_loader.dataset))

    train_after_stride = _stat(train_stats, 'after_stride_count', train_before)
    val_after_stride = _stat(val_stats, 'after_stride_count', val_before)

    train_final = _stat(train_stats, 'final_count', len(train_loader.dataset))
    val_final = _stat(val_stats, 'final_count', len(val_loader.dataset))

    train_stride_cut = max(0, train_before - train_after_stride)
    val_stride_cut = max(0, val_before - val_after_stride)
    total_before = train_before + val_before
    total_after_stride = train_after_stride + val_after_stride
    total_stride_cut = train_stride_cut + val_stride_cut

    train_second_cut = max(0, train_after_stride - train_final)
    val_second_cut = max(0, val_after_stride - val_final)
    total_final = train_final + val_final
    total_second_cut = train_second_cut + val_second_cut

    print_status_table(
        "Data Ready (IL) - Overview",
        [
            ("Positions total", f"{metadata.get('total_positions', 0):,}"),
            ("Position size", f"{actual_position_size} bytes"),
            ("Board encoding", "38B board (32B pieces + 6B metadata)"),
            ("POV", "on"),
            ("Sliding stride", stride),
            ("2nd filter", "position_sampling (on)" if sampling_enabled else "position_sampling (off)"),
            ("Batch size", config['imitation_learning']['batch_size']),
        ],
    )

    print_multi_column_table(
        "Data Ready (IL) - Filtering Pipeline",
        headers=["Stage", "Train", "Val", "Total"],
        rows=[
            ("Split (before filters)", _fmt_count(train_before), _fmt_count(val_before), _fmt_count(total_before)),
            (
                "Sliding window removed",
                _fmt_cut(train_stride_cut, train_before),
                _fmt_cut(val_stride_cut, val_before),
                _fmt_cut(total_stride_cut, total_before),
            ),
            (
                "After sliding window",
                _fmt_count(train_after_stride),
                _fmt_count(val_after_stride),
                _fmt_count(total_after_stride),
            ),
            (
                "2nd filter removed",
                _fmt_cut(train_second_cut, train_after_stride),
                _fmt_cut(val_second_cut, val_after_stride),
                _fmt_cut(total_second_cut, total_after_stride),
            ),
            ("Final samples", _fmt_count(train_final), _fmt_count(val_final), _fmt_count(total_final)),
        ],
    )

    # Stochastic Weight Averaging (SWA) for better generalization with large batches
    use_swa = config['imitation_learning'].get('use_swa', False)
    swa_model = None
    swa_scheduler = None
    swa_start = config['imitation_learning'].get('swa_start_epoch', 15)

    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_lr = config['imitation_learning'].get('swa_lr', 0.0005)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
        print(f"SWA enabled: start_epoch={swa_start} (inclusive), swa_lr={swa_lr:.6f}")

    # Training loop
    print("\nStarting training...")
    start_msg = f"mode={start_mode}"
    if selected_checkpoint is not None and start_mode in {"resume", "transfer"}:
        start_msg += f", checkpoint={selected_checkpoint_label}"
    print(start_msg)
    print(
        f"optimizer_state={'on' if save_optimizer_state else 'off'}, "
        f"history={history_positions}, stride={stride}, input_planes={expected_input_planes}"
    )
    if debug_enabled:
        print("Debug profiling is active")
    profile_enabled = debug_enabled and config.get('debug', {}).get('profile_training', False)
    profile_every = config.get('debug', {}).get('profile_every_n_epochs', 1)

    max_patience = config['imitation_learning'].get('max_patience', 15)
    min_delta = config['imitation_learning']['min_delta']
    checkpoint_every = config['imitation_learning'].get('checkpoint_every', 5)
    total_epochs = config['imitation_learning']['epochs']

    print(
        f"early_stopping(patience={max_patience}, min_delta={min_delta}), "
        f"checkpoint_every={checkpoint_every}"
    )
    print(f"paths: best={best_model_path}, checkpoints={il_dir}")

    transfer_freeze_active = False
    transfer_freeze_until_epoch = start_epoch + transfer_freeze_epochs
    if start_mode == "transfer" and transfer_freeze_epochs > 0:
        if transfer_trainable_param_names:
            transfer_trainable_set = set(transfer_trainable_param_names)
            trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
            for param_name, param in model.named_parameters():
                param.requires_grad = param_name in transfer_trainable_set
            trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
            transfer_freeze_active = True
            print(
                "Transfer warmup: frozen trunk active for "
                f"first {transfer_freeze_epochs} epoch(s)."
            )
            print(
                f"  trainable params during warmup: {trainable_after:,}/{trainable_before:,} "
                f"across {len(transfer_trainable_set)} tensors"
            )
        else:
            print(
                "Transfer warmup requested but no changed trainable tensors were detected; "
                "running without freeze."
            )
    elo_config = config.get("elo_estimation", {})
    elo_config_il = dict(elo_config)
    if elo_config_il.get("use_mcts", False):
        print("Note: IL forces elo_estimation.use_mcts=False for speed.")
    elo_config_il["use_mcts"] = False

    elo_coordinator = ILEloCoordinator(
        model=model,
        config=config,
        device=device,
        elo_config_il=elo_config_il,
        logger=logger,
    )
    elo_coordinator.print_startup_summary()

    def _get_elo_state_for_checkpoint():
        elo_epoch, elo_value = logger.get_latest_estimated_elo_with_epoch()
        elo_metadata = {}
        if elo_value is not None:
            elo_metadata['estimated_elo'] = float(elo_value)
        if elo_epoch is not None:
            elo_metadata['estimated_elo_epoch'] = int(elo_epoch)
        return elo_metadata, elo_epoch, elo_value

    current_estimated_elo = logger.get_latest_estimated_elo()

    training_interrupted = False
    last_epoch_idx = start_epoch - 1

    try:
        for epoch in range(start_epoch, total_epochs):
            last_epoch_idx = epoch
            if transfer_freeze_active and epoch >= transfer_freeze_until_epoch:
                for param in model.parameters():
                    param.requires_grad = True
                transfer_freeze_active = False
                total_trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(
                    f"Transfer warmup finished at epoch {epoch + 1}. "
                    f"Full model unfrozen ({total_trainable_now:,} trainable params)."
                )
            elo_coordinator.poll_results()
            current_estimated_elo = logger.get_latest_estimated_elo()
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            epoch_lr = optimizer.param_groups[0]['lr']

            profile_this_epoch = profile_enabled and ((epoch + 1) % profile_every == 0)
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            epoch_start_time = time.perf_counter()

            # Train epoch
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            train_start_time = time.perf_counter()
            train_losses, train_metrics, train_profile = train_epoch_il(
                model, train_loader, optimizer, scheduler, config, device, scaler,
                epoch=epoch, debug_log_file=debug_log_file, profile=profile_this_epoch
            )
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start_time
            if profile_this_epoch and train_profile is not None:
                train_time = train_profile['total']

            # 🆕 SWA: Update averaged model starting from swa_start epoch (inclusive)
            if use_swa and (epoch + 1) >= swa_start:
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
                if debug_enabled:
                    print(
                        f"       SWA update ({epoch + 2 - swa_start}/"
                        f"{config['imitation_learning']['epochs'] - swa_start + 1})"
                    )

            # Print train losses and metrics
            print(f"Train - Loss: {train_losses['total']:.4f}, "
                  f"Policy: {train_losses['policy']:.4f}, "
                  f"Value: {train_losses['value']:.4f}, "
                  f"LR: {epoch_lr:.2e}")

            print(f"       📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
                  f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
                  f"MAE: {train_metrics['value_mae']:.4f}")

            if use_mtl:
                print(f"       🆕 MTL - Win: {train_losses['win']:.4f}, "
                      f"Material: {train_losses['material']:.4f}, "
                      f"Check: {train_losses['check']:.4f}")

            # Evaluate
            if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
                if profile_this_epoch and device.type == 'cuda':
                    torch.cuda.synchronize()
                eval_start_time = time.perf_counter()
                val_losses, val_metrics = evaluate_il(model, val_loader, config, device)
                if profile_this_epoch and device.type == 'cuda':
                    torch.cuda.synchronize()
                eval_time = time.perf_counter() - eval_start_time

                print(f"Val - Loss: {val_losses['total']:.4f}, "
                      f"Policy: {val_losses['policy']:.4f}, "
                      f"Value: {val_losses['value']:.4f}")

                print(f"     📊 Top-1: {val_metrics['policy_top1_acc']:.2%}, "
                      f"Top-3: {val_metrics['policy_top3_acc']:.2%}, "
                      f"MAE: {val_metrics['value_mae']:.4f}")

                if use_mtl:
                    print(f"     🆕 MTL - Win: {val_losses['win']:.4f}, "
                          f"Material: {val_losses['material']:.4f}, "
                          f"Check: {val_losses['check']:.4f}")

                # Log metrics + periodic Elo estimation.
                estimated_elo = elo_coordinator.evaluate_if_due(epoch + 1)
                if estimated_elo is not None:
                    current_estimated_elo = float(estimated_elo)
                else:
                    current_estimated_elo = logger.get_latest_estimated_elo()

                logger.log(epoch + 1, train_losses, val_losses, train_metrics, val_metrics, epoch_lr,
                           estimated_elo=current_estimated_elo)
                logger.plot()
                elo_coordinator.poll_results()
                current_estimated_elo = logger.get_latest_estimated_elo()

                # Save best model
                improvement = best_val_loss - val_losses['total']
                if improvement > min_delta:
                    best_val_loss = val_losses['total']
                    patience_counter = 0

                    print(f"✓ New best model! Val loss: {val_losses['total']:.4f} "
                          f"(improved by {improvement:.4f})")
                    print(f"  📊 Val Top-1: {val_metrics['policy_top1_acc']:.2%}")

                    model_to_save = model
                    elo_metadata, elo_epoch_for_state, elo_for_state = _get_elo_state_for_checkpoint()
                    best_metadata = {
                        'val_policy_loss': val_losses['policy'],
                        'val_value_loss': val_losses['value'],
                        'val_policy_top1': val_metrics['policy_top1_acc'],
                        'val_policy_top3': val_metrics['policy_top3_acc'],
                        'val_value_mae': val_metrics['value_mae'],
                        'use_amp': use_amp,
                        'use_bfloat16': use_bfloat16,
                        'use_mtl': use_mtl,
                        'history_positions': history_positions,
                        'input_planes': expected_input_planes,
                        'sliding_window_stride': stride,
                        'pov_enabled': True,  # 🆕 v4.2
                        'version': model_version,    # 🆕 Track version
                        'startup_mode': start_mode,
                        'model_architecture': model_architecture,
                    }
                    best_metadata.update(elo_metadata)
                    save_checkpoint(
                        model_to_save,
                        optimizer,
                        epoch,
                        val_losses['total'],
                        str(best_model_path),
                        best_metadata,
                        save_optimizer=save_optimizer_state,
                        save_dtype=torch.bfloat16 if use_bfloat16 else None,
                        extra_state=build_runtime_state(
                            scheduler=scheduler,
                            scaler=scaler,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            estimated_elo=elo_for_state,
                            estimated_elo_epoch=elo_epoch_for_state,
                        ),
                    )

                    print(f"  💾 Saved to: {best_model_path}")
                    size_mb = best_model_path.stat().st_size / (1024**2)
                    optimizer_label = "with optimizer" if save_optimizer_state else "without optimizer"
                    print(f"  📦 Model size: {size_mb:.1f} MB ({optimizer_label})")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{max_patience}")

                # Early stopping
                if patience_counter >= max_patience:
                    print(f"\n🛑 Early stopping triggered!")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                # Log only training metrics
                logger.log(
                    epoch + 1,
                    train_losses,
                    None,
                    train_metrics,
                    None,
                    epoch_lr,
                    estimated_elo=current_estimated_elo,
                )
                elo_coordinator.poll_results()
                current_estimated_elo = logger.get_latest_estimated_elo()
                eval_time = 0.0

            if profile_this_epoch:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                epoch_total_time = time.perf_counter() - epoch_start_time
                other_time = max(0.0, epoch_total_time - train_time - eval_time)
                if train_profile is not None:
                    batches = max(1, train_profile['batches'])
                    profile_msg = (
                        f"[PROFILE] Epoch {epoch + 1}: "
                        f"data={train_profile['data']:.2f}s ({train_profile['data']*1000/batches:.1f}ms/b), "
                        f"fwd={train_profile['forward']:.2f}s ({train_profile['forward']*1000/batches:.1f}ms/b), "
                        f"bwd={train_profile['backward']:.2f}s ({train_profile['backward']*1000/batches:.1f}ms/b), "
                        f"optim={train_profile['optim']:.2f}s ({train_profile['optim']*1000/batches:.1f}ms/b), "
                        f"metrics={train_profile['metrics']:.2f}s ({train_profile['metrics']*1000/batches:.1f}ms/b), "
                        f"train_total={train_profile['total']:.2f}s, "
                        f"eval={eval_time:.2f}s, "
                        f"other={other_time:.2f}s, "
                        f"epoch_total={epoch_total_time:.2f}s"
                    )
                else:
                    profile_msg = (
                        f"[PROFILE] Epoch {epoch + 1}: "
                        f"train={train_time:.2f}s, "
                        f"eval={eval_time:.2f}s, "
                        f"other={other_time:.2f}s, "
                        f"total={epoch_total_time:.2f}s"
                    )
                print(profile_msg)
                if debug_log_file is not None:
                    with open(debug_log_file, 'a', encoding='utf-8') as f:
                        f.write(profile_msg + "\n")

            # Save checkpoint every N epochs
            if (epoch + 1) % checkpoint_every == 0:
                # Get current val loss if not already computed
                if (epoch + 1) % config['imitation_learning']['eval_every'] != 0:
                    val_losses, val_metrics = evaluate_il(model, val_loader, config, device)

                checkpoint_name = f"{model_file_tag}_epoch_{epoch + 1:02d}.pt"
                checkpoint_path = il_dir / checkpoint_name

                model_to_save = model
                elo_metadata, elo_epoch_for_state, elo_for_state = _get_elo_state_for_checkpoint()
                periodic_metadata = {
                    'val_loss': val_losses['total'],
                    'val_policy_loss': val_losses['policy'],
                    'val_value_loss': val_losses['value'],
                    'val_policy_top1': val_metrics['policy_top1_acc'],
                    'val_policy_top3': val_metrics['policy_top3_acc'],
                    'val_value_mae': val_metrics['value_mae'],
                    'use_mtl': use_mtl,
                    'history_positions': history_positions,
                    'input_planes': expected_input_planes,
                    'sliding_window_stride': stride,
                    'pov_enabled': True,  # 🆕 v4.2
                    'version': model_version,    # 🆕 Track version
                    'startup_mode': start_mode,
                    'model_architecture': model_architecture,
                }
                periodic_metadata.update(elo_metadata)
                save_checkpoint(
                    model_to_save,
                    optimizer,
                    epoch,
                    train_losses['total'],
                    str(checkpoint_path),
                    periodic_metadata,
                    save_optimizer=save_optimizer_state,
                    save_dtype=torch.bfloat16 if use_bfloat16 else None,
                    extra_state=build_runtime_state(
                        scheduler=scheduler,
                        scaler=scaler,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        estimated_elo=elo_for_state,
                        estimated_elo_epoch=elo_epoch_for_state,
                    ),
                )

                size_mb = checkpoint_path.stat().st_size / (1024**2)
                print(f"💾 Checkpoint saved: {checkpoint_path.name} ({size_mb:.1f} MB)")

                # Save SWA snapshot on the same cadence (if SWA already active).
                save_swa_snapshot_checkpoint(
                    epoch_idx=epoch,
                    fallback_loss=train_losses['total'],
                    ref_val_losses=val_losses,
                    ref_val_metrics=val_metrics,
                    use_swa=use_swa,
                    swa_model=swa_model,
                    swa_start=swa_start,
                    il_dir=il_dir,
                    use_mtl=use_mtl,
                    history_positions=history_positions,
                    expected_input_planes=expected_input_planes,
                    stride=stride,
                    model_version=model_version,
                    start_mode=start_mode,
                    use_bfloat16=use_bfloat16,
                    model_file_tag=model_file_tag,
                    model_architecture=model_architecture,
                )

            # Cleanup
            if (epoch + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        training_interrupted = True
        print("\nCtrl+C detected. Attempting graceful shutdown...")

    elo_coordinator.shutdown(interrupted=training_interrupted)

    # Final plot
    logger.plot()

    # Finalize SWA at training end and also on interrupt (if SWA has updates).
    swa_final_result = finalize_swa_model(
        final_epoch_idx=last_epoch_idx,
        interrupted=training_interrupted,
        use_swa=use_swa,
        swa_model=swa_model,
        use_amp=use_amp,
        use_bfloat16=use_bfloat16,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        best_model_path=best_model_path,
        use_mtl=use_mtl,
        swa_start=swa_start,
        history_positions=history_positions,
        expected_input_planes=expected_input_planes,
        stride=stride,
        model_version=model_version,
        start_mode=start_mode,
        best_val_loss=best_val_loss,
        evaluate_il_fn=evaluate_il,
        elo_config=elo_config_il,
        model_architecture=model_architecture,
    )

    if swa_final_result.get("finalized"):
        swa_note = (
            "SWA final: "
            f"val_loss={swa_final_result['val_loss']:.4f}, "
            f"top1={swa_final_result['val_top1']:.2%}, "
            f"mae={swa_final_result['val_mae']:.4f}"
        )
        if swa_final_result.get("estimated_elo") is not None:
            swa_note += f", elo={int(round(float(swa_final_result['estimated_elo'])))}"
        logger.append_final_note(swa_note)
        logger.plot()

    if training_interrupted:
        cleanup_interrupted_log_csv(_LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG, "IL")
        print("IL training stopped gracefully.")
        return

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Checkpoints: {il_dir}")
    print(f"Logs: {logs_dir}")
    if debug_log_file:
        print(f"Debug log: {debug_log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_interrupted_log_csv(_LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG, "IL")
        print("\nCtrl+C detected. IL training stopped gracefully.")
