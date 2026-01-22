import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint
from src.data import parse_pgn_files, create_dataloaders


def train_epoch(model, train_loader, optimizer, config, device):
    """Train for one epoch"""
    model.train()
    
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (boards, moves, outcomes) in enumerate(pbar):
        boards = boards.to(device)
        moves = moves.to(device)
        outcomes = outcomes.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        policy_pred, value_pred = model(boards)
        
        # Compute losses
        policy_loss = criterion_policy(policy_pred, moves)
        value_loss = criterion_value(value_pred, outcomes)
        
        loss = policy_weight * policy_loss + value_weight * value_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['imitation_learning']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['imitation_learning']['grad_clip']
            )
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        if batch_idx % config['logging']['print_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}"
            })
        
        # Periodic garbage collection
        if batch_idx % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    n = len(train_loader)
    return total_loss / n, total_policy_loss / n, total_value_loss / n


def evaluate(model, val_loader, config, device):
    """Evaluate model on validation set"""
    model.eval()
    
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()
    
    with torch.no_grad():
        for boards, moves, outcomes in tqdm(val_loader, desc="Evaluating"):
            boards = boards.to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)
            
            policy_pred, value_pred = model(boards)
            
            policy_loss = criterion_policy(policy_pred, moves)
            value_loss = criterion_value(value_pred, outcomes)
            
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    n = len(val_loader)
    return total_loss / n, total_policy_loss / n, total_value_loss / n


def main():
    # Load config
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Create directories
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data (now returns metadata instead of full data)
    print("\n=== Loading data ===")
    
    # Check if we should use cached data
    preprocessing_dir = base_dir / "data" / "preprocessing"
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    
    binary_file = preprocessing_dir / f"positions_elo{min_elo}_games{max_games}_moves{max_moves}.bin"
    metadata_file = preprocessing_dir / f"positions_elo{min_elo}_games{max_games}_moves{max_moves}_meta.pkl"
    
    use_cache = True
    if binary_file.exists() and metadata_file.exists():
        file_size_mb = binary_file.stat().st_size / (1024**2)
        response = input(f"\nFound cached binary data:\n{binary_file}\nSize: {file_size_mb:.2f} MB\n\nUse cached data? (y/n): ").strip().lower()
        use_cache = response == 'y'
        
        if not use_cache:
            print("Will reprocess PGN files...")
            # Delete old cache files
            binary_file.unlink()
            metadata_file.unlink()
            print(f"Deleted cache files")
    
    metadata = parse_pgn_files(
        config['paths']['data_dir'], 
        config,
        num_workers=config['hardware']['num_workers']
    )
    
    print(f"Total positions: {metadata['total_positions']}")
    
    # Create dataloaders (now uses chunked lazy loading)
    train_loader, val_loader = create_dataloaders(metadata, config)
    
    # Create model
    print("\n=== Creating model ===")
    model = ChessNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['imitation_learning']['learning_rate'],
        weight_decay=config['imitation_learning']['weight_decay']
    )
    
    # Training loop
    print("\n=== Starting training ===")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['imitation_learning']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['imitation_learning']['epochs']}")
        
        # Train
        train_loss, train_policy, train_value = train_epoch(
            model, train_loader, optimizer, config, device
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Policy: {train_policy:.4f}, Value: {train_value:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
            val_loss, val_policy, val_value = evaluate(model, val_loader, config, device)
            print(f"Val - Loss: {val_loss:.4f}, Policy: {val_policy:.4f}, Value: {val_value:.4f}")
            
            # Save best model
            if val_loss < best_val_loss - config['imitation_learning']['min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_path = base_dir / config['paths']['best_model']
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    str(best_model_path),
                    {'val_policy_loss': val_policy, 'val_value_loss': val_value}
                )
                print(f"✓ New best model saved! Val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{config['imitation_learning']['patience']}")
            
            # Early stopping
            if patience_counter >= config['imitation_learning']['patience']:
                print("\nEarly stopping triggered!")
                break
        
        # Save checkpoint
        if (epoch + 1) % config['imitation_learning']['save_every'] == 0:
            checkpoint_path = base_dir / config['paths']['il_checkpoint']
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                str(checkpoint_path)
            )
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n=== Training complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {base_dir / config['paths']['best_model']}")


if __name__ == "__main__":
    main()