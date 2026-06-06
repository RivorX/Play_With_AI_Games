"""
Chess Data Validation Script
Validates preprocessed binary data for quality, integrity, and correctness.

Usage (from project root or anywhere):
    python chess/scripts/debug/IL_validate_data.py                    # Validate all files
    python chess/scripts/debug/IL_validate_data.py --file data.bin    # Validate specific file
    python chess/scripts/debug/IL_validate_data.py --quick            # Quick validation (fewer samples)
"""

import sys
from pathlib import Path

# Add project root to path
# If script is in chess/scripts/debug/IL_validate_data.py, go up 3 levels to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import struct
import mmap
import numpy as np
import chess
from collections import defaultdict, Counter
from tqdm import tqdm
import yaml
import argparse


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compact_to_board(compact_bytes):
    """
    Convert compact 32-byte representation back to chess.Board
    Format: 64 squares × 4 bits = 32 bytes (2 squares per byte)
    """
    board = chess.Board(fen=None)  # Empty board
    board.clear()
    
    # Piece code mapping
    code_to_piece = {
        1: (chess.PAWN, chess.WHITE),
        2: (chess.KNIGHT, chess.WHITE),
        3: (chess.BISHOP, chess.WHITE),
        4: (chess.ROOK, chess.WHITE),
        5: (chess.QUEEN, chess.WHITE),
        6: (chess.KING, chess.WHITE),
        7: (chess.PAWN, chess.BLACK),
        8: (chess.KNIGHT, chess.BLACK),
        9: (chess.BISHOP, chess.BLACK),
        10: (chess.ROOK, chess.BLACK),
        11: (chess.QUEEN, chess.BLACK),
        12: (chess.KING, chess.BLACK),
    }
    
    # Unpack piece positions (32 bytes = 64 squares, 2 per byte)
    for i in range(32):
        byte = compact_bytes[i]
        code1 = (byte >> 4) & 0x0F  # Upper 4 bits
        code2 = byte & 0x0F          # Lower 4 bits
        
        square1 = i * 2
        square2 = i * 2 + 1
        
        # Place pieces
        for square, code in [(square1, code1), (square2, code2)]:
            if code > 0 and code in code_to_piece:
                piece_type, color = code_to_piece[code]
                piece = chess.Piece(piece_type, color)
                board.set_piece_at(square, piece)
    
    return board


def index_to_move(move_index, board=None):
    """
    Convert move index back to UCI string
    Format: from_square * 64 + to_square
    """
    from_square = move_index // 64
    to_square = move_index % 64
    
    uci = chess.square_name(from_square) + chess.square_name(to_square)
    
    return uci


def move_to_index(move):
    """
    Convert chess.Move to index
    Format: from_square * 64 + to_square
    """
    return move.from_square * 64 + move.to_square


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

class DataValidator:
    """Comprehensive data validation"""
    
    def __init__(self, binary_file, metadata_file, config_path='chess/config/config.yaml', quick_mode=False):
        """
        Initialize validator
        
        Args:
            binary_file: Path to binary data file (str or Path)
            metadata_file: Path to metadata pickle (str or Path)
            config_path: Path to config.yaml (relative to project root)
            quick_mode: If True, use fewer samples for faster validation
        """
        self.binary_file = Path(binary_file)
        self.metadata_file = Path(metadata_file)
        self.quick_mode = quick_mode
        
        # Load config (config_path is relative to project root)
        # Find project root from script location
        # chess/scripts/debug/IL_validate_data.py -> go up 3 levels
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        config_abs_path = project_root / config_path
        
        with open(config_abs_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Load metadata
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Debug: Show metadata structure
        print(f"\n🔍 Metadata type: {type(self.metadata)}")
        if isinstance(self.metadata, dict):
            print(f"🔍 Metadata keys: {list(self.metadata.keys())}")
        
        # Parse parameters from binary filename as fallback
        # Format: positions_elo1800_hist4.bin
        bin_name = self.binary_file.stem
        
        # Extract history count
        import re
        hist_match = re.search(r'hist(\d+)', bin_name)
        history_from_name = int(hist_match.group(1)) if hist_match else 0
        
        # Handle different metadata formats
        if isinstance(self.metadata, dict) and 'total_positions' in self.metadata:
            # Standard metadata format
            self.total_positions = self.metadata['total_positions']
            self.position_size = self.metadata.get('position_size', 38)
            self.history_positions = self.metadata.get('history_positions', history_from_name)
            self.input_planes = self.metadata.get('input_planes', 12)
        else:
            # Fallback: infer from file and filename
            print(f"⚠️  Non-standard metadata format, using filename parameters")
            
            # Use parameters from filename
            self.history_positions = history_from_name
            self.input_planes = 12 * (1 + self.history_positions)
            
            # Calculate position size
            # Base: 32 (board) + 32*history + 2 (move) + 4 (outcome)
            base_size = 32 + (32 * self.history_positions) + 2 + 4
            self.position_size = base_size
            
            # Calculate total positions from file size
            file_size = self.binary_file.stat().st_size
            self.total_positions = file_size // self.position_size
            
            print(f"   ✓ Parsed from filename: History={self.history_positions}")
            print(f"   ✓ Calculated: position_size={self.position_size}, total={self.total_positions:,}")
        
        # Statistics
        self.stats = defaultdict(int)
        self.errors = []
        self.warnings = []
        
        mode_str = "⚡ QUICK MODE" if quick_mode else "🔍 FULL MODE"
        print(f"\n{'='*70}")
        print(f"🔍 DATA VALIDATOR INITIALIZED ({mode_str})")
        print(f"{'='*70}")
        print(f"📁 File: {self.binary_file.name}")
        print(f"📊 Total positions: {self.total_positions:,}")
        print(f"📦 Position size: {self.position_size} bytes")
        print(f"📜 History positions: {self.history_positions}")
        print(f"🎲 Input planes: {self.input_planes}")
        print(f"{'='*70}\n")
    
    def validate_file_integrity(self):
        """Check if file size matches expected size"""
        print("🔍 Validating file integrity...")
        
        expected_size = self.total_positions * self.position_size
        actual_size = self.binary_file.stat().st_size
        
        if expected_size != actual_size:
            self.errors.append(
                f"File size mismatch! Expected: {expected_size:,} bytes, "
                f"Actual: {actual_size:,} bytes"
            )
            return False
        
        print(f"  ✅ File size correct: {actual_size:,} bytes")
        return True
    
    def validate_sample_positions(self, num_samples=1000, check_legality=True):
        """Validate random sample of positions"""
        print(f"\n🔍 Validating {num_samples} random positions...")
        
        # Random sample
        indices = np.random.choice(self.total_positions, size=min(num_samples, self.total_positions), replace=False)
        
        legal_moves = 0
        illegal_moves = 0
        invalid_positions = 0
        
        with open(self.binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            for idx in tqdm(indices, desc="  Checking positions"):
                offset = idx * self.position_size
                data = mm[offset:offset + self.position_size]
                
                try:
                    # Unpack current board
                    compact_board = data[:32]
                    current_offset = 32
                    
                    # Skip history
                    history_size = 32 * self.history_positions
                    current_offset += history_size
                    
                    # Unpack move
                    move_index = struct.unpack('H', data[current_offset:current_offset+2])[0]
                    current_offset += 2
                    
                    # Unpack outcome
                    outcome = struct.unpack('f', data[current_offset:current_offset+4])[0]
                    current_offset += 4
                    
                    # Validate outcome range
                    if not (-1.0 <= outcome <= 1.0):
                        self.warnings.append(f"Position {idx}: Outcome out of range: {outcome}")
                    
                    # Convert board (note: compact format doesn't store turn/castling/ep)
                    board = compact_to_board(compact_board)
                    
                    # Basic validation - check if board has pieces
                    piece_count = len(board.piece_map())
                    if piece_count < 2:  # At least 2 kings
                        invalid_positions += 1
                        self.errors.append(f"Position {idx}: Too few pieces ({piece_count})")
                        continue
                    
                    # Check for kings
                    white_king = len(board.pieces(chess.KING, chess.WHITE))
                    black_king = len(board.pieces(chess.KING, chess.BLACK))
                    
                    if white_king != 1 or black_king != 1:
                        invalid_positions += 1
                        self.errors.append(
                            f"Position {idx}: Invalid king count "
                            f"(White: {white_king}, Black: {black_king})"
                        )
                        continue
                    
                    # Check move index range (0-4095 for 64x64)
                    if move_index >= 64 * 64:
                        illegal_moves += 1
                        self.errors.append(f"Position {idx}: Move index out of range: {move_index}")
                        continue
                    
                    # Note: We can't fully validate move legality because compact format
                    # doesn't store turn, castling rights, or en passant
                    # We can only check if move squares are valid
                    if check_legality:
                        from_square = move_index // 64
                        to_square = move_index % 64
                        
                        if from_square >= 64 or to_square >= 64:
                            illegal_moves += 1
                            self.errors.append(
                                f"Position {idx}: Invalid move squares "
                                f"(from: {from_square}, to: {to_square})"
                            )
                        else:
                            legal_moves += 1
                    
                    
                except Exception as e:
                    self.errors.append(f"Position {idx}: Error reading data: {str(e)}")
                    invalid_positions += 1
            
            mm.close()
        
        # Summary
        print(f"\n  📊 Sample Validation Results:")
        print(f"    • Positions checked: {len(indices):,}")
        print(f"    • Invalid positions: {invalid_positions:,}")
        
        if check_legality:
            print(f"    • Valid move indices: {legal_moves:,}")
            print(f"    • Invalid move indices: {illegal_moves:,}")
            
            if legal_moves + illegal_moves > 0:
                validity_rate = 100 * legal_moves / (legal_moves + illegal_moves)
                print(f"    • Move validity rate: {validity_rate:.2f}%")
                
                if validity_rate < 99.0:
                    self.warnings.append(f"Low move validity rate: {validity_rate:.2f}%")
        
        return invalid_positions == 0 and illegal_moves == 0
    
    def analyze_move_distribution(self, num_samples=10000):
        """Analyze move distribution for sanity check"""
        print(f"\n🔍 Analyzing move distribution ({num_samples:,} samples)...")
        
        move_counter = Counter()
        outcome_distribution = {'win': 0, 'loss': 0, 'draw': 0}
        
        indices = np.random.choice(self.total_positions, size=min(num_samples, self.total_positions), replace=False)
        
        with open(self.binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            for idx in tqdm(indices, desc="  Analyzing"):
                offset = idx * self.position_size
                data = mm[offset:offset + self.position_size]
                
                # Skip to move index
                current_offset = 32 + (32 * self.history_positions)
                move_index = struct.unpack('H', data[current_offset:current_offset+2])[0]
                current_offset += 2
                
                outcome = struct.unpack('f', data[current_offset:current_offset+4])[0]
                
                move_counter[move_index] += 1
                
                if outcome > 0.5:
                    outcome_distribution['win'] += 1
                elif outcome < -0.5:
                    outcome_distribution['loss'] += 1
                else:
                    outcome_distribution['draw'] += 1
            
            mm.close()
        
        # Results
        print(f"\n  📊 Move Distribution:")
        print(f"    • Unique moves: {len(move_counter):,}")
        print(f"    • Most common move: index {move_counter.most_common(1)[0][0]} "
              f"({move_counter.most_common(1)[0][1]:,} times)")
        
        print(f"\n  📊 Outcome Distribution:")
        total = sum(outcome_distribution.values())
        for outcome, count in outcome_distribution.items():
            percentage = 100 * count / total
            print(f"    • {outcome.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        # Sanity checks
        if len(move_counter) < 100:
            self.warnings.append(f"Very few unique moves: {len(move_counter)}")
        
        # Check for balanced outcomes (should be somewhat balanced in master games)
        win_rate = 100 * outcome_distribution['win'] / total
        if win_rate < 30 or win_rate > 70:
            self.warnings.append(f"Unusual win rate: {win_rate:.1f}%")
        
        return True
    
    def validate_history_consistency(self, num_samples=500):
        """Validate that history positions are consistent"""
        if self.history_positions == 0:
            print("\n⏭️  No history positions, skipping history validation")
            return True
        
        print(f"\n🔍 Validating history consistency ({num_samples:,} samples)...")
        
        inconsistent = 0
        empty_histories = 0
        
        indices = np.random.choice(self.total_positions, size=min(num_samples, self.total_positions), replace=False)
        
        with open(self.binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            for idx in tqdm(indices, desc="  Checking history"):
                offset = idx * self.position_size
                data = mm[offset:offset + self.position_size]
                
                # Unpack current board
                compact_board = data[:32]
                current_board = compact_to_board(compact_board)
                
                # Unpack history
                current_offset = 32
                for h in range(self.history_positions):
                    hist_board_bytes = data[current_offset:current_offset+32]
                    current_offset += 32
                    
                    try:
                        # Check if history board is empty (padding for early game)
                        is_empty = all(b == 0 for b in hist_board_bytes)
                        
                        if is_empty:
                            # Empty history board is OK (padding for beginning of game)
                            empty_histories += 1
                            continue
                        
                        hist_board = compact_to_board(hist_board_bytes)
                        
                        # Basic validity check for non-empty boards
                        piece_count = len(hist_board.piece_map())
                        if piece_count < 2:
                            inconsistent += 1
                            self.errors.append(f"Position {idx}: Invalid history board {h} (too few pieces)")
                            continue
                        
                        # Check for kings in non-empty history
                        white_king = len(hist_board.pieces(chess.KING, chess.WHITE))
                        black_king = len(hist_board.pieces(chess.KING, chess.BLACK))
                        
                        if white_king != 1 or black_king != 1:
                            inconsistent += 1
                            self.errors.append(
                                f"Position {idx}: Invalid history board {h} "
                                f"(kings: W={white_king}, B={black_king})"
                            )
                    except Exception as e:
                        inconsistent += 1
                        self.errors.append(f"Position {idx}: Error in history {h}: {str(e)}")
            
            mm.close()
        
        print(f"\n  📊 History Validation:")
        print(f"    • Positions checked: {len(indices):,}")
        print(f"    • Empty history boards: {empty_histories:,} (padding for early game)")
        print(f"    • Invalid history boards: {inconsistent:,}")
        
        if inconsistent > 0:
            self.warnings.append(f"Found {inconsistent} invalid history boards (excluding empty padding)")
        
        # Only fail if we have truly invalid boards (not just empty padding)
        return inconsistent == 0
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print(f"\n{'='*70}")
        print("🚀 STARTING FULL VALIDATION")
        print(f"{'='*70}\n")
        
        # Adjust sample sizes based on mode
        if self.quick_mode:
            samples_positions = 500
            samples_moves = 2000
            samples_history = 200
        else:
            samples_positions = 2000
            samples_moves = 10000
            samples_history = 1000
        
        # Run all checks
        checks = [
            ("File Integrity", self.validate_file_integrity),
            ("Sample Positions", lambda: self.validate_sample_positions(num_samples=samples_positions)),
            ("Move Distribution", lambda: self.analyze_move_distribution(num_samples=samples_moves)),
            ("History Consistency", lambda: self.validate_history_consistency(num_samples=samples_history)),
        ]
        
        results = {}
        for check_name, check_func in checks:
            try:
                results[check_name] = check_func()
            except Exception as e:
                results[check_name] = False
                self.errors.append(f"{check_name}: {str(e)}")
        
        # Final Report
        print(f"\n{'='*70}")
        print("📋 VALIDATION REPORT")
        print(f"{'='*70}\n")
        
        print("✅ Checks Passed:")
        for check_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        # Overall status
        all_passed = all(results.values()) and len(self.errors) == 0
        
        print(f"\n{'='*70}")
        if all_passed:
            print("✅ VALIDATION PASSED!")
            print("Data appears to be correct and ready for training.")
        else:
            print("❌ VALIDATION FAILED!")
            print("Please fix the errors before training.")
        print(f"{'='*70}\n")
        
        return all_passed


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main validation function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Validate preprocessed chess data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  python chess/scripts/debug/IL_validate_data.py                    # Validate all files
  python chess/scripts/debug/IL_validate_data.py --file data.bin    # Validate specific file
  python chess/scripts/debug/IL_validate_data.py --quick            # Quick validation (fewer samples)
  python chess/scripts/debug/IL_validate_data.py --file data.bin --quick
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Specific binary file to validate (e.g., "data.bin")'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick validation mode (fewer samples, faster)'
    )
    
    args = parser.parse_args()
    
    # Get project root
    # If script is in chess/scripts/debug/IL_validate_data.py, go up 3 levels
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Debug: Print paths
    print(f"🔍 DEBUG: Script location: {Path(__file__).absolute()}")
    print(f"🔍 DEBUG: script_dir = {script_dir.absolute()}")
    print(f"🔍 DEBUG: project_root = {project_root.absolute()}")
    
    # Data directory (chess/data/preprocessing/)
    data_dir = project_root / "chess" / "data" / "preprocessing"
    print(f"🔍 DEBUG: data_dir = {data_dir.absolute()}")
    print(f"🔍 DEBUG: data_dir exists? {data_dir.exists()}")
    print()
    
    # Find binary files
    if args.file:
        # Specific file requested
        binary_file = data_dir / args.file
        if binary_file.exists():
            binary_files = [binary_file]
        else:
            print(f"❌ File not found: {binary_file}")
            print(f"   Looking in: {data_dir}")
            return
    else:
        # All files in directory
        binary_files = list(data_dir.glob("*.bin"))
    
    if not binary_files:
        print("❌ No .bin files found in data/preprocessing directory!")
        print(f"   Looking in: {data_dir}")
        print(f"\n💡 Tip: Binary files should be in data/preprocessing/ directory")
        return
    
    print(f"\n{'='*70}")
    print("🔍 FOUND BINARY FILES:")
    print(f"{'='*70}")
    for i, bf in enumerate(binary_files, 1):
        size_mb = bf.stat().st_size / (1024**2)
        print(f"  {i}. {bf.name} ({size_mb:.1f} MB)")
    print(f"{'='*70}\n")
    
    if args.quick:
        print("⚡ Running in QUICK mode (fewer samples for faster validation)\n")
    
    # Validate each file
    for binary_file in binary_files:
        # Try to find corresponding metadata file
        # Your naming: filename.bin -> filename_meta.pkl
        metadata_file = binary_file.with_name(binary_file.stem + '_meta.pkl')
        
        if not metadata_file.exists():
            print(f"⚠️  Skipping {binary_file.name}: Metadata not found")
            print(f"     Expected: {metadata_file.name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📂 VALIDATING: {binary_file.name}")
        print(f"{'='*70}")
        
        validator = DataValidator(
            binary_file=binary_file,
            metadata_file=metadata_file,
            config_path='chess/config/config.yaml',  # Relative to project root
            quick_mode=args.quick
        )
        
        validator.run_full_validation()
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
