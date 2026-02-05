"""
Facade module for chess data pipeline and dataset utilities.
Implementation lives in src/utils to keep files smaller.
"""

from src.utils.data_dataset import (
    BinaryChessDataset,
    _build_game_ranges,
    _split_indices_by_game,
    create_dataloaders,
)
from src.utils.data_helpers import board_to_tensor, move_to_index
from src.utils.data_pipeline import (
    DatasetTracker,
    cleanup_intermediate_files,
    extract_auxiliary_labels,
    extract_game_data,
    extract_games_from_pgn_multiprocess,
    extract_games_from_pgn_parallel,
    extract_games_sequential,
    extract_positions_from_game_worker,
    extract_positions_parallel,
    extract_positions_sequential,
    get_dataset_metadata,
    merge_binary_datasets,
    parse_games_batch_worker,
    process_pgn_files,
    sort_games_by_elo,
    write_positions_to_disk,
)

__all__ = [
    'BinaryChessDataset',
    '_build_game_ranges',
    '_split_indices_by_game',
    'create_dataloaders',
    'board_to_tensor',
    'move_to_index',
    'DatasetTracker',
    'cleanup_intermediate_files',
    'extract_auxiliary_labels',
    'extract_game_data',
    'extract_games_from_pgn_multiprocess',
    'extract_games_from_pgn_parallel',
    'extract_games_sequential',
    'extract_positions_from_game_worker',
    'extract_positions_parallel',
    'extract_positions_sequential',
    'get_dataset_metadata',
    'merge_binary_datasets',
    'parse_games_batch_worker',
    'process_pgn_files',
    'sort_games_by_elo',
    'write_positions_to_disk',
]
