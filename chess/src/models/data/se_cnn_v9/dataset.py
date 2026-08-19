"""
Chess dataset and dataloader utilities.
Dataset and DataLoader implementation owned by the SE-CNN v9 data contract.
"""

from src.game import backend as chess
from collections import defaultdict
import gc
import hashlib
import json
import math
import mmap
import os
import random
import struct
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from src.models.data.se_cnn_v9.helpers import (
    ACTION_SIZE,
    AZ_ACTION_SIZE,
    az_index_to_policy_index,
    compact_to_board,
    compact_to_tensor,
    get_policy_index_maps,
    get_turn_from_move_idx,
    index_to_move,
)
from src.config import normalize_data_config


_SOFT_TARGET_AGGREGATE_CONTEXT = {}
_SOFT_TARGET_MATERIALIZE_CONTEXT = {}
_COUNT_AWARE_SELECTION_CONTEXT = {}
_SAMPLE_DEDUP_CONTEXT = {}


# Compact boards encode two piece codes in each byte.  Position selection reads
# these fields hundreds of millions of times during a full IL cache build, so
# decode each byte once at import time instead of repeatedly unpacking nibbles
# and looking up a dictionary in Python.
_COMPACT_PIECE_VALUES = (0, 1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0, 0, 0, 0)
_COMPACT_PIECE_COUNT_BY_BYTE = tuple(
    int(((byte >> 4) & 0x0F) != 0) + int((byte & 0x0F) != 0)
    for byte in range(256)
)
_COMPACT_MATERIAL_BY_BYTE = tuple(
    _COMPACT_PIECE_VALUES[(byte >> 4) & 0x0F] + _COMPACT_PIECE_VALUES[byte & 0x0F]
    for byte in range(256)
)
_COMPACT_PAWNS_BY_BYTE = tuple(
    int(((byte >> 4) & 0x0F) in (1, 7)) + int((byte & 0x0F) in (1, 7))
    for byte in range(256)
)
_COMPACT_PIECE_COUNT_LOOKUP = np.asarray(_COMPACT_PIECE_COUNT_BY_BYTE, dtype=np.uint8)
_COMPACT_MATERIAL_LOOKUP = np.asarray(_COMPACT_MATERIAL_BY_BYTE, dtype=np.uint8)
_COMPACT_PAWNS_LOOKUP = np.asarray(_COMPACT_PAWNS_BY_BYTE, dtype=np.uint8)


def _cache_candidates(primary_path, legacy_paths=None):
    paths = []
    if primary_path is not None:
        paths.append(Path(primary_path))
    for path in legacy_paths or ():
        if path is not None:
            path = Path(path)
            if path not in paths:
                paths.append(path)
    return paths


def _load_index_cache(cache_path, label, stage, legacy_cache_paths=None):
    candidates = _cache_candidates(cache_path, legacy_cache_paths)
    if not candidates:
        return None
    primary = candidates[0]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            cached = np.load(candidate, mmap_mode='r')
            if cached.dtype != np.uint32:
                continue
            if candidate != primary:
                primary.parent.mkdir(parents=True, exist_ok=True)
                np.save(primary, cached, allow_pickle=False)
                cached = np.load(primary, mmap_mode='r')
                print(
                    f"  - {label} {stage}: migrated cached indices "
                    f"({len(cached):,}) -> {primary.name}"
                )
            else:
                print(f"  - {label} {stage}: loaded cached indices ({len(cached):,})")
            return cached
        except (OSError, ValueError):
            continue
    return None


def _load_uint32_index_array(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        cached = np.load(path, mmap_mode='r')
        if cached.dtype != np.uint32:
            return None
        return cached
    except (OSError, ValueError):
        return None


def _read_game_id_at(binary_file, position_size, index):
    if index < 0:
        return None
    with open(binary_file, 'rb') as f:
        f.seek(int(index) * int(position_size) + 38)
        raw = f.read(4)
    if len(raw) != 4:
        return None
    return int(struct.unpack('I', raw)[0])


def _validate_cached_ranges_prefix(binary_file, position_size, total_positions, ranges, sample_points=16):
    if not ranges:
        return False
    try:
        total_positions = int(total_positions)
        old_total = int(ranges[-1][2])
    except (TypeError, ValueError, IndexError):
        return False
    if old_total <= 0 or old_total > total_positions:
        return False
    check_indices = {0, len(ranges) - 1}
    if len(ranges) > 2:
        for pos in np.linspace(0, len(ranges) - 1, num=min(sample_points, len(ranges)), dtype=np.int64):
            check_indices.add(int(pos))
    for pos in sorted(check_indices):
        try:
            game_id, start, end = ranges[pos]
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return False
        if start < 0 or end <= start or end > total_positions:
            return False
        if _read_game_id_at(binary_file, position_size, start) != int(game_id):
            return False
        if _read_game_id_at(binary_file, position_size, end - 1) != int(game_id):
            return False
    if old_total < total_positions:
        last_game = int(ranges[-1][0])
        next_game = _read_game_id_at(binary_file, position_size, old_total)
        if next_game is None or next_game == last_game:
            return False
    return True


def _close_numpy_mmap(array):
    mmap_obj = getattr(array, '_mmap', None)
    if mmap_obj is not None:
        try:
            mmap_obj.close()
        except (BufferError, OSError, ValueError):
            pass


def _save_npy_atomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, 'wb') as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _lookup_game_length(game_length_by_id, game_id):
    if game_length_by_id is None:
        return 0
    if isinstance(game_length_by_id, np.ndarray):
        game_id = int(game_id)
        if 0 <= game_id < len(game_length_by_id):
            return int(game_length_by_id[game_id])
        return 0
    return int(game_length_by_id.get(int(game_id), 0))


def _build_game_length_lookup(game_ranges):
    if not game_ranges:
        return None
    max_game_id = max(int(game_id) for game_id, _, _ in game_ranges)
    # Game IDs generated by this pipeline are dense, so an array is far smaller
    # and much cheaper to pickle into DataLoader workers than a Python dict.
    if max_game_id <= max(1_000_000, len(game_ranges) * 4):
        lengths = np.zeros(max_game_id + 1, dtype=np.uint16)
        for game_id, start, end in game_ranges:
            lengths[int(game_id)] = min(65535, int(end - start))
        return lengths
    return {int(game_id): int(end - start) for game_id, start, end in game_ranges}


def _evenly_spaced_values(values, count):
    values = list(values)
    count = int(count)
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    selected = []
    seen = set()
    for pos in np.linspace(0, len(values) - 1, num=count):
        idx = int(round(float(pos)))
        idx = max(0, min(idx, len(values) - 1))
        value = values[idx]
        if value not in seen:
            selected.append(value)
            seen.add(value)
    if len(selected) < count:
        for value in values:
            if value in seen:
                continue
            selected.append(value)
            seen.add(value)
            if len(selected) >= count:
                break
    return selected


def _largest_remainder_budgets(max_total, weights):
    max_total = max(1, int(max_total))
    cleaned = {key: max(0.0, float(value or 0.0)) for key, value in weights.items()}
    total_weight = sum(cleaned.values())
    if total_weight <= 0.0:
        cleaned = {'opening': 20.0, 'middlegame': 45.0, 'endgame': 35.0, 'rare_or_eventful': 0.0}
        total_weight = sum(cleaned.values())

    raw = {key: (value / total_weight) * max_total for key, value in cleaned.items()}
    budgets = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = max_total - sum(budgets.values())
    order = sorted(raw, key=lambda key: (raw[key] - budgets[key], raw[key]), reverse=True)
    for key in order[:remaining]:
        budgets[key] += 1
    return budgets


def _positions_per_game_budgets(cfg):
    max_total = max(1, int(cfg.get('max_total', 32)))
    percent_keys = ('opening_pct', 'middlegame_pct', 'endgame_pct', 'rare_or_eventful_pct')
    if any(key in cfg for key in percent_keys):
        return _largest_remainder_budgets(
            max_total,
            {
                'opening': cfg.get('opening_pct', 20),
                'middlegame': cfg.get('middlegame_pct', 45),
                'endgame': cfg.get('endgame_pct', 35),
                'rare_or_eventful': cfg.get('rare_or_eventful_pct', 0),
            },
        )

    # Backward compatibility for old configs that used absolute counts.
    return {
        'opening': max(0, int(cfg.get('opening', 6))),
        'middlegame': max(0, int(cfg.get('middlegame', 14))),
        'endgame': max(0, int(cfg.get('endgame', 10))),
        'rare_or_eventful': max(0, int(cfg.get('rare_or_eventful', 0))),
    }


def _is_rare_or_eventful_record(record):
    compact_board = record[:38]
    move_idx = struct.unpack('H', record[42:44])[0]
    move_target = struct.unpack('H', record[44:46])[0]
    if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
        move_target = az_index_to_policy_index(move_target)

    turn = get_turn_from_move_idx(move_idx)
    board = compact_to_board(compact_board, turn=turn)
    move = index_to_move(move_target, is_black_turn=(turn == chess.BLACK), board=board)
    legal_moves = chess.legal_moves(board)
    if move is None or move not in legal_moves:
        return False

    if move.promotion is not None:
        return True
    if chess.is_en_passant(board, move) or chess.is_castling(board, move):
        return True
    if chess.is_check(board) or chess.gives_check(board, move):
        return True
    if len(legal_moves) <= 2:
        return True
    if chess.piece_count(board) <= 7:
        return True
    if int(board.halfmove_clock) >= 80:
        return True

    chess.apply_move(board, move)
    try:
        return chess.is_game_over(board, claim_draw=True)
    finally:
        chess.undo_move(board)


def _compact_piece_stats(compact_board):
    piece_count = 0
    material = 0
    pawns = 0
    for byte in compact_board[:32]:
        piece_count += _COMPACT_PIECE_COUNT_BY_BYTE[byte]
        material += _COMPACT_MATERIAL_BY_BYTE[byte]
        pawns += _COMPACT_PAWNS_BY_BYTE[byte]
    return piece_count, material, pawns


def _quick_record_features(record):
    compact_board = record[:38]
    piece_count, material, pawns = _compact_piece_stats(compact_board)
    return (
        piece_count,
        material,
        pawns,
        int(compact_board[32]),
        int(compact_board[33]),
        (int(compact_board[34]) << 8) | int(compact_board[35]),
        (int(compact_board[36]) << 8) | int(compact_board[37]),
    )


def _record_at(mm, position_size, index):
    if mm is None:
        return None
    offset = int(index) * int(position_size)
    record = mm[offset:offset + int(position_size)]
    if len(record) != int(position_size):
        return None
    return record


def _smart_position_score_from_features(features, abs_idx, start, end):
    rel = int(abs_idx) - int(start)
    current = features[rel] if 0 <= rel < len(features) else None
    if current is None:
        return 0.0

    (
        piece_count,
        material,
        pawns,
        castling,
        ep_square,
        halfmove_clock,
        fullmove,
    ) = current
    length = max(1, int(end) - int(start))
    progress = float(rel) / float(max(1, length - 1))

    score = 0.0

    if progress < 0.10:
        score -= 1.10
    elif progress < 0.25:
        score += 0.10
    elif progress < 0.75:
        score += 0.55
    elif progress < 0.95:
        score += 0.45
    else:
        score -= 0.15

    if fullmove <= 8:
        score -= 0.75
    elif fullmove <= 14:
        score += 0.10
    elif fullmove <= 40:
        score += 0.35
    else:
        score += 0.25

    if piece_count <= 8:
        score += 0.75
    elif piece_count <= 14:
        score += 0.55
    elif piece_count <= 22:
        score += 0.25
    elif piece_count >= 30:
        score -= 0.25

    if pawns <= 8:
        score += 0.15
    if ep_square != 255:
        score += 0.30
    if halfmove_clock == 0 and fullmove > 8:
        score += 0.20

    prev_features = features[rel - 1] if rel > 0 else None
    next_features = features[rel + 1] if rel + 1 < len(features) else None
    for neighbor in (prev_features, next_features):
        if neighbor is None:
            continue
        n_piece_count, n_material = neighbor[0], neighbor[1]
        if n_piece_count != piece_count:
            score += 0.50
        if n_material != material:
            score += 0.35
        if neighbor[3] != castling:
            score += 0.20

    # Stable tiny jitter prevents deterministic ties from always picking the
    # earliest position inside a phase while keeping cache reproducibility.
    score += ((int(abs_idx) * 1103515245 + 12345) & 0xFFFF) / 0xFFFF * 0.01
    return score


def _is_far_enough(rel, selected, min_distance):
    return all(abs(int(rel) - int(existing)) >= int(min_distance) for existing in selected)


def _add_spaced_by_score(candidates, selected, count, min_distance):
    added = []
    if count <= 0:
        return added
    ordered = sorted(candidates, key=lambda item: (-float(item[0]), int(item[1])))
    for _, rel in ordered:
        rel = int(rel)
        if rel in selected:
            continue
        if not _is_far_enough(rel, selected, min_distance):
            continue
        selected.add(rel)
        added.append(rel)
        if len(added) >= count:
            break
    return added


def _add_spaced_score_range(scores, start, end, selected, count, min_distance):
    """Allocation-light equivalent of ``_add_spaced_by_score`` for score arrays."""
    if count <= 0:
        return 0
    added = 0
    for rel in sorted(range(int(start), int(end)), key=lambda rel: (-float(scores[rel]), int(rel))):
        if rel in selected or not _is_far_enough(rel, selected, min_distance):
            continue
        selected.add(rel)
        added += 1
        if added >= count:
            break
    return added


def _select_positions_for_game_smart(mm, position_size, start, end, cfg):
    length = int(end) - int(start)
    if length <= 0:
        return []

    max_total = max(1, int(cfg.get('max_total', 32)))
    min_distance = max(1, int(cfg.get('min_distance', 4)))
    budgets = _positions_per_game_budgets(cfg)
    rel_positions = list(range(length))
    opening_cut = max(1, int(round(length * 0.25)))
    endgame_cut = max(opening_cut + 1, int(round(length * 0.75)))
    phase_pools = {
        'opening': rel_positions[:opening_cut],
        'middlegame': rel_positions[opening_cut:endgame_cut],
        'endgame': rel_positions[endgame_cut:],
    }

    # Read and decode each record once.  The old implementation decoded the
    # current, previous and next record for every score, tripling the memory
    # traffic in this full-dataset preprocessing stage.
    features = []
    for abs_idx in range(int(start), int(end)):
        record = _record_at(mm, position_size, abs_idx)
        features.append(None if record is None else _quick_record_features(record))
    scored = {
        rel: _smart_position_score_from_features(features, int(start) + rel, start, end)
        for rel in rel_positions
    }
    selected = set()

    rare_budget = max(0, int(budgets.get('rare_or_eventful', 0)))
    if rare_budget > 0:
        rare_candidates = []
        for rel in rel_positions:
            record = _record_at(mm, position_size, int(start) + rel)
            if record is not None and _is_rare_or_eventful_record(record):
                rare_candidates.append((scored[rel] + 2.0, rel))
        _add_spaced_by_score(rare_candidates, selected, min(rare_budget, max_total), min_distance)

    for phase in ('opening', 'middlegame', 'endgame'):
        remaining = max_total - len(selected)
        if remaining <= 0:
            break
        budget = min(max(0, int(budgets.get(phase, 0))), remaining)
        candidates = [(scored[rel], rel) for rel in phase_pools[phase]]
        _add_spaced_by_score(candidates, selected, budget, min_distance)

    remaining = max_total - len(selected)
    if remaining > 0:
        candidates = [(score, rel) for rel, score in scored.items()]
        _add_spaced_by_score(candidates, selected, remaining, min_distance)

    return [int(start) + rel for rel in sorted(selected)]


def _select_positions_for_game_smart_from_arrays(start, end, cfg, base_index, piece_count, material,
                                                  pawns, castling, ep_square, halfmove, fullmove):
    """Same smart selector, with compact-board features decoded in NumPy batches."""
    length = int(end) - int(start)
    if length <= 0:
        return []

    max_total = max(1, int(cfg.get('max_total', 32)))
    min_distance = max(1, int(cfg.get('min_distance', 4)))
    budgets = _positions_per_game_budgets(cfg)
    opening_cut = max(1, int(round(length * 0.25)))
    endgame_cut = max(opening_cut + 1, int(round(length * 0.75)))
    local_start = int(start) - int(base_index)

    scores = [0.0] * length
    for rel in range(length):
        local = local_start + rel
        current_piece_count = int(piece_count[local])
        current_material = int(material[local])
        current_pawns = int(pawns[local])
        current_castling = int(castling[local])
        current_ep_square = int(ep_square[local])
        current_halfmove = int(halfmove[local])
        current_fullmove = int(fullmove[local])
        progress = float(rel) / float(max(1, length - 1))
        score = 0.0

        if progress < 0.10:
            score -= 1.10
        elif progress < 0.25:
            score += 0.10
        elif progress < 0.75:
            score += 0.55
        elif progress < 0.95:
            score += 0.45
        else:
            score -= 0.15

        if current_fullmove <= 8:
            score -= 0.75
        elif current_fullmove <= 14:
            score += 0.10
        elif current_fullmove <= 40:
            score += 0.35
        else:
            score += 0.25

        if current_piece_count <= 8:
            score += 0.75
        elif current_piece_count <= 14:
            score += 0.55
        elif current_piece_count <= 22:
            score += 0.25
        elif current_piece_count >= 30:
            score -= 0.25

        if current_pawns <= 8:
            score += 0.15
        if current_ep_square != 255:
            score += 0.30
        if current_halfmove == 0 and current_fullmove > 8:
            score += 0.20

        for neighbor in (local - 1 if rel > 0 else None, local + 1 if rel + 1 < length else None):
            if neighbor is None:
                continue
            if int(piece_count[neighbor]) != current_piece_count:
                score += 0.50
            if int(material[neighbor]) != current_material:
                score += 0.35
            if int(castling[neighbor]) != current_castling:
                score += 0.20

        abs_idx = int(start) + rel
        scores[rel] = score + ((abs_idx * 1103515245 + 12345) & 0xFFFF) / 0xFFFF * 0.01

    selected = set()
    phase_ranges = (
        ('opening', 0, opening_cut),
        ('middlegame', opening_cut, endgame_cut),
        ('endgame', endgame_cut, length),
    )
    for phase, phase_start, phase_end in phase_ranges:
        remaining = max_total - len(selected)
        if remaining <= 0:
            break
        budget = min(max(0, int(budgets.get(phase, 0))), remaining)
        _add_spaced_score_range(scores, phase_start, phase_end, selected, budget, min_distance)

    remaining = max_total - len(selected)
    if remaining > 0:
        _add_spaced_score_range(scores, 0, length, selected, remaining, min_distance)
    return [int(start) + rel for rel in sorted(selected)]


def _select_positions_for_game(mm, position_size, start, end, cfg):
    length = int(end) - int(start)
    if length <= 0:
        return []

    max_total = max(1, int(cfg.get('max_total', 32)))
    selection_mode = str(cfg.get('selection_mode', 'even')).strip().lower()
    if selection_mode == 'smart' and mm is not None:
        return _select_positions_for_game_smart(mm, position_size, start, end, cfg)

    if length <= max_total:
        return list(range(int(start), int(end)))

    budgets = _positions_per_game_budgets(cfg)
    opening_budget = max(0, int(budgets.get('opening', 0)))
    middlegame_budget = max(0, int(budgets.get('middlegame', 0)))
    endgame_budget = max(0, int(budgets.get('endgame', 0)))
    rare_budget = max(0, int(budgets.get('rare_or_eventful', 0)))

    rel_positions = list(range(length))
    opening_cut = max(1, int(round(length * 0.25)))
    endgame_cut = max(opening_cut + 1, int(round(length * 0.75)))

    phase_pools = {
        'opening': rel_positions[:opening_cut],
        'middlegame': rel_positions[opening_cut:endgame_cut],
        'endgame': rel_positions[endgame_cut:],
    }

    selected = set()
    if rare_budget > 0 and mm is not None:
        rare = []
        for rel in rel_positions:
            offset = (int(start) + rel) * position_size
            record = mm[offset:offset + position_size]
            if len(record) == position_size and _is_rare_or_eventful_record(record):
                rare.append(rel)
        for rel in _evenly_spaced_values(rare, min(rare_budget, max_total)):
            selected.add(rel)

    for phase, budget in (
        ('opening', opening_budget),
        ('middlegame', middlegame_budget),
        ('endgame', endgame_budget),
    ):
        remaining_budget = max_total - len(selected)
        if remaining_budget <= 0:
            break
        budget = min(int(budget), remaining_budget)
        pool = [rel for rel in phase_pools[phase] if rel not in selected]
        for rel in _evenly_spaced_values(pool, budget):
            selected.add(rel)

    remaining_budget = max_total - len(selected)
    if remaining_budget > 0:
        pool = [rel for rel in rel_positions if rel not in selected]
        for rel in _evenly_spaced_values(pool, remaining_budget):
            selected.add(rel)

    return [int(start) + rel for rel in sorted(selected)]


def _resolve_positions_per_game_workers(cfg, chunk_count):
    raw = cfg.get('workers', 'auto')
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {'', 'auto', 'all'}:
            # These phases run before training, so use every logical CPU.  There
            # is no GPU/DataLoader work to reserve a core for at this point.
            workers = max(1, int(os.cpu_count() or 1))
        elif lowered in {'off', 'false', 'no'}:
            workers = 1
        else:
            try:
                workers = int(lowered)
            except ValueError:
                workers = max(1, int(os.cpu_count() or 1))
    else:
        try:
            workers = int(raw)
        except (TypeError, ValueError):
            workers = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(workers), max(1, int(chunk_count))))


def _positions_per_game_chunk_size(cfg, game_count, workers):
    raw = cfg.get('chunk_size', 2048)
    if isinstance(raw, str) and raw.strip().lower() in {'auto', ''}:
        target_chunks = max(1, int(workers) * 12)
        return max(256, min(8192, int(np.ceil(float(game_count) / float(target_chunks)))))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2048


def _chunk_game_ranges(game_ranges, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for chunk_id, start in enumerate(range(0, len(game_ranges), chunk_size)):
        yield chunk_id, list(game_ranges[start:start + chunk_size])


def _uses_vectorized_smart_selector(cfg):
    if str(cfg.get('selection_mode', 'even')).strip().lower() != 'smart':
        return False
    return max(0, int(_positions_per_game_budgets(cfg).get('rare_or_eventful', 0))) == 0


def _select_positions_per_game_smart_vectorized(mm, position_size, ranges, cfg):
    """Decode contiguous compact-board ranges in NumPy, then retain exact smart ranking."""
    selected = []
    max_batch_positions = 1_000_000
    batch_start = 0
    while batch_start < len(ranges):
        first_position = int(ranges[batch_start][1])
        batch_end = batch_start + 1
        while batch_end < len(ranges) and int(ranges[batch_end][2]) - first_position <= max_batch_positions:
            batch_end += 1
        last_position = int(ranges[batch_end - 1][2])
        position_count = last_position - first_position
        raw = np.ndarray(
            (position_count, int(position_size)),
            dtype=np.uint8,
            buffer=mm,
            offset=first_position * int(position_size),
        )
        board_bytes = raw[:, :32]
        piece_count = _COMPACT_PIECE_COUNT_LOOKUP[board_bytes].sum(axis=1, dtype=np.uint8)
        material = _COMPACT_MATERIAL_LOOKUP[board_bytes].sum(axis=1, dtype=np.uint8)
        pawns = _COMPACT_PAWNS_LOOKUP[board_bytes].sum(axis=1, dtype=np.uint8)
        castling = raw[:, 32]
        ep_square = raw[:, 33]
        halfmove = (raw[:, 34].astype(np.uint16) << 8) | raw[:, 35]
        fullmove = (raw[:, 36].astype(np.uint16) << 8) | raw[:, 37]
        for _, start, end in ranges[batch_start:batch_end]:
            selected.extend(_select_positions_for_game_smart_from_arrays(
                start,
                end,
                cfg,
                first_position,
                piece_count,
                material,
                pawns,
                castling,
                ep_square,
                halfmove,
                fullmove,
            ))
        batch_start = batch_end
    return selected


def _select_positions_per_game_worker(args):
    chunk_id, binary_file, position_size, ranges, cfg, needs_records = args
    selected = []
    if needs_records:
        with open(binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                if _uses_vectorized_smart_selector(cfg):
                    selected.extend(_select_positions_per_game_smart_vectorized(mm, position_size, ranges, cfg))
                else:
                    for _, start, end in ranges:
                        selected.extend(_select_positions_for_game(mm, position_size, start, end, cfg))
            finally:
                mm.close()
    else:
        for _, start, end in ranges:
            selected.extend(_select_positions_for_game(None, position_size, start, end, cfg))
    return chunk_id, np.asarray(selected, dtype=np.uint32)


def _select_positions_per_game_parallel(binary_file, position_size, game_ranges, cfg, label, needs_records):
    game_ranges = list(game_ranges)
    if not game_ranges:
        return np.empty(0, dtype=np.uint32)

    preliminary_workers = _resolve_positions_per_game_workers(cfg, len(game_ranges))
    chunk_size = _positions_per_game_chunk_size(cfg, len(game_ranges), preliminary_workers)
    chunks = list(_chunk_game_ranges(game_ranges, chunk_size))
    workers = _resolve_positions_per_game_workers(cfg, len(chunks))
    if workers <= 1 or len(chunks) <= 1:
        _, arr = _select_positions_per_game_worker(
            (0, binary_file, position_size, game_ranges, cfg, needs_records)
        )
        return arr

    results = [None] * len(chunks)
    task_args = [
        (chunk_id, binary_file, position_size, ranges, dict(cfg), needs_records)
        for chunk_id, ranges in chunks
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_select_positions_per_game_worker, args) for args in task_args]
        with tqdm(total=len(game_ranges), desc=f"  {label} positions/game", unit="game") as pbar:
            for future in as_completed(futures):
                chunk_id, arr = future.result()
                results[int(chunk_id)] = arr
                pbar.update(len(chunks[int(chunk_id)][1]))

    arrays = [arr for arr in results if arr is not None and len(arr) > 0]
    if not arrays:
        return np.empty(0, dtype=np.uint32)
    return np.concatenate(arrays).astype(np.uint32, copy=False)


def _select_positions_per_game(binary_file, position_size, game_ranges, cfg, label, cache_path=None,
                               legacy_cache_paths=None):
    if not cfg.get('enabled', False):
        return _ranges_to_index_array((start, end) for _, start, end in game_ranges)

    cached = _load_index_cache(cache_path, label, "positions_per_game", legacy_cache_paths)
    if cached is not None:
        return cached

    selected = []
    before = sum(max(0, int(end) - int(start)) for _, start, end in game_ranges)
    rare_budget = max(0, int(_positions_per_game_budgets(cfg).get('rare_or_eventful', 0)))
    selection_mode = str(cfg.get('selection_mode', 'even')).strip().lower()
    needs_records = rare_budget > 0 or selection_mode == 'smart'
    if needs_records:
        if rare_budget > 0:
            print(f"  - {label} positions_per_game: rare_or_eventful scan enabled; this scans full game records.")
        indices = _select_positions_per_game_parallel(
            binary_file,
            position_size,
            game_ranges,
            cfg,
            label,
            needs_records=True,
        )
        if cache_path is not None:
            _save_npy_atomic(cache_path, indices)
            indices = np.load(cache_path, mmap_mode='r')
        print(
            f"  {label} positions/game: {before:,} -> {len(indices):,} "
            f"(max={int(cfg.get('max_total', 32))}, min_dist={int(cfg.get('min_distance', 1))})"
        )
        return indices
    iterator = tqdm(game_ranges, desc=f"  {label} positions/game", unit="game")
    if selection_mode == 'smart' and rare_budget <= 0:
        with open(binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                for _, start, end in iterator:
                    selected.extend(_select_positions_for_game(mm, position_size, start, end, cfg))
            finally:
                mm.close()

        indices = np.asarray(selected, dtype=np.uint32)
        if cache_path is not None:
            _save_npy_atomic(cache_path, indices)
            indices = np.load(cache_path, mmap_mode='r')
        print(
            f"  {label} positions/game: {before:,} -> {len(indices):,} "
            f"(max={int(cfg.get('max_total', 32))}, min_dist={int(cfg.get('min_distance', 1))})"
        )
        return indices
    if needs_records:
        print(f"  • {label} positions_per_game: rare_or_eventful scan enabled; this scans full game records.")
        with open(binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                for _, start, end in iterator:
                    selected.extend(_select_positions_for_game(mm, position_size, start, end, cfg))
            finally:
                mm.close()
    else:
        for _, start, end in iterator:
            selected.extend(_select_positions_for_game(None, position_size, start, end, cfg))

    indices = np.asarray(selected, dtype=np.uint32)
    if cache_path is not None:
        _save_npy_atomic(cache_path, indices)
        indices = np.load(cache_path, mmap_mode='r')
    print(
        f"  {label} positions/game: {before:,} -> {len(indices):,} "
        f"(max={int(cfg.get('max_total', 32))}, min_dist={int(cfg.get('min_distance', 1))})"
    )
    return indices


def _mask_sorted_indices_by_ranges(indices, ranges):
    indices = np.asarray(indices, dtype=np.uint32)
    mask = np.zeros(len(indices), dtype=bool)
    if len(indices) == 0 or not ranges:
        return mask
    starts = np.fromiter((int(start) for _, start, _ in ranges), dtype=np.uint32)
    ends = np.fromiter((int(end) for _, _, end in ranges), dtype=np.uint32)
    if len(starts) == 0:
        return mask
    chunk_size = 5_000_000
    for start_idx in range(0, len(indices), chunk_size):
        end_idx = min(len(indices), start_idx + chunk_size)
        chunk = indices[start_idx:end_idx]
        positions = np.searchsorted(ends, chunk, side='right')
        valid = positions < len(starts)
        if np.any(valid):
            chunk_mask = np.zeros(len(chunk), dtype=bool)
            valid_positions = positions[valid]
            chunk_mask[valid] = chunk[valid] >= starts[valid_positions]
            mask[start_idx:end_idx] = chunk_mask
    return mask


def _load_all_ppg_from_split_caches(all_cache_path, train_cache_path, val_cache_path, label):
    train_cached = _load_uint32_index_array(train_cache_path)
    val_cached = _load_uint32_index_array(val_cache_path)
    if train_cached is None or val_cached is None:
        return None
    combined = np.concatenate([
        np.asarray(train_cached, dtype=np.uint32),
        np.asarray(val_cached, dtype=np.uint32),
    ])
    combined.sort()
    if all_cache_path is not None:
        _save_npy_atomic(all_cache_path, combined)
        combined = np.load(all_cache_path, mmap_mode='r')
    print(
        f"  - {label} positions_per_game: built all-game cache from train/val caches "
        f"({len(combined):,})"
    )
    return combined


def _selector_digest_for_prefix(binary_file, position_size, total_positions, config):
    data_cfg = config.get('data', {}) or {}
    binary_path = Path(binary_file)
    binary_identity = {
        'binary_file': str(binary_path.resolve()),
        'binary_size': int(total_positions) * int(position_size),
        'total_positions': int(total_positions),
        'position_size': int(position_size),
    }
    selector_payload = {
        **binary_identity,
        'selector_cache_schema': 4,
        'seed': int(config.get('seed', 0) or 0),
        'train_split': float(data_cfg.get('train_split', 0.85) or 0.85),
        'positions_per_game': _positions_per_game_cache_config(data_cfg.get('positions_per_game', {})),
    }
    return hashlib.sha1(
        json.dumps(selector_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]


def _ppg_all_digest_for_prefix(binary_file, position_size, total_positions, config):
    data_cfg = config.get('data', {}) or {}
    binary_path = Path(binary_file)
    payload = {
        'binary_file': str(binary_path.resolve()),
        'binary_size': int(total_positions) * int(position_size),
        'total_positions': int(total_positions),
        'position_size': int(position_size),
        'ppg_all_cache_schema': 1,
        'positions_per_game': _positions_per_game_cache_config(data_cfg.get('positions_per_game', {})),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]


def _load_append_base_ppg_selection(binary_file, position_size, total_positions, config,
                                    cache_dir, game_ranges):
    cache_dir = Path(cache_dir)
    manifests = sorted(
        cache_dir.glob("il_prepare_*.json"),
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    current_total = int(total_positions)
    for manifest_path in manifests:
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        try:
            old_total = int(manifest.get('total_positions', 0) or 0)
            old_position_size = int(manifest.get('position_size', 0) or 0)
        except (TypeError, ValueError):
            continue
        if old_total <= 0 or old_total >= current_total or old_position_size != int(position_size):
            continue
        old_ranges = [item for item in game_ranges if int(item[2]) <= old_total]
        if not old_ranges or int(old_ranges[-1][2]) != old_total:
            continue
        if not _validate_cached_ranges_prefix(binary_file, position_size, total_positions, old_ranges):
            continue

        expected_selector = _selector_digest_for_prefix(binary_file, position_size, old_total, config)
        expected_all = _ppg_all_digest_for_prefix(binary_file, position_size, old_total, config)
        digests = manifest.get('digests', {}) or {}
        selector_digest = digests.get('selector')
        all_digest = digests.get('ppg_all')

        old_selected = None
        if all_digest == expected_all:
            old_selected = _load_uint32_index_array(cache_dir / f"il_indices_{all_digest}_all_ppg.npy")
        if old_selected is None and selector_digest == expected_selector:
            old_selected = _load_all_ppg_from_split_caches(
                None,
                cache_dir / f"il_indices_{selector_digest}_train_ppg.npy",
                cache_dir / f"il_indices_{selector_digest}_val_ppg.npy",
                "Append base",
            )
        if old_selected is None:
            continue
        if len(old_selected) and int(np.max(old_selected)) >= old_total:
            continue
        return old_total, np.asarray(old_selected, dtype=np.uint32)
    return None, None


def _select_positions_per_game_all(binary_file, position_size, game_ranges, cfg, config,
                                   cache_path=None, legacy_cache_paths=None,
                                   train_cache_path=None, val_cache_path=None):
    cached = _load_index_cache(cache_path, "All", "positions_per_game", legacy_cache_paths)
    if cached is not None:
        return np.sort(np.asarray(cached, dtype=np.uint32))

    migrated = _load_all_ppg_from_split_caches(cache_path, train_cache_path, val_cache_path, "All")
    if migrated is not None:
        return np.sort(np.asarray(migrated, dtype=np.uint32))

    sorted_ranges = sorted(list(game_ranges), key=lambda item: int(item[1]))
    total_positions = int(sorted_ranges[-1][2]) if sorted_ranges else 0
    old_total, old_selected = _load_append_base_ppg_selection(
        binary_file,
        position_size,
        total_positions,
        config,
        Path(cache_path).parent if cache_path is not None else Path(binary_file).parent / "index_cache",
        sorted_ranges,
    )
    if old_selected is not None and old_total:
        tail_ranges = [item for item in sorted_ranges if int(item[1]) >= int(old_total)]
        if tail_ranges:
            print(
                f"  - All positions_per_game: append cache hit for first {int(old_total):,} "
                f"records; selecting {len(tail_ranges):,} new games only"
            )
            tail_selected = _select_positions_per_game(
                binary_file,
                position_size,
                tail_ranges,
                cfg,
                "Append-tail",
                cache_path=None,
            )
            selected = np.concatenate([
                np.asarray(old_selected, dtype=np.uint32),
                np.asarray(tail_selected, dtype=np.uint32),
            ])
            selected.sort()
            if cache_path is not None:
                _save_npy_atomic(cache_path, selected)
                selected = np.load(cache_path, mmap_mode='r')
            return selected

    selected = _select_positions_per_game(
        binary_file,
        position_size,
        sorted_ranges,
        cfg,
        "All",
        cache_path=cache_path,
        legacy_cache_paths=legacy_cache_paths,
    )
    return np.sort(np.asarray(selected, dtype=np.uint32))


def _sample_signature_for_index(mm, position_size, index, cfg, history_positions):
    valid_modes = {"fen", "fen_no_counters", "pieces", "position_plus_move"}
    mode = cfg.get('mode', 'position_plus_move')
    if mode not in valid_modes:
        raise ValueError(f"sample_dedup.mode must be one of {sorted(valid_modes)}, got: {mode}")

    offset = int(index) * int(position_size)
    record = mm[offset:offset + position_size]
    if len(record) != position_size:
        return None

    if mode == "fen":
        signature_board_bytes = 38
        parts = [record[:signature_board_bytes]]
    elif mode == "fen_no_counters":
        signature_board_bytes = 34
        parts = [record[:signature_board_bytes]]
    elif mode == "pieces":
        signature_board_bytes = 32
        parts = [record[:signature_board_bytes]]
    else:
        signature_board_bytes = 34
        move_target = struct.unpack('H', record[44:46])[0]
        if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
            move_target = az_index_to_policy_index(move_target)
        parts = [record[:34], struct.pack('H', int(move_target))]

    if cfg.get('include_turn', True):
        # MoveIdx is little-endian at bytes 42:44; only the parity is needed.
        parts.append(bytes((record[42] & 1,)))

    history_positions = max(0, int(history_positions or 0))
    if cfg.get('include_history', True) and history_positions > 0:
        history_boards = []
        current_index = int(index) - 1
        game_id_bytes = record[38:42]
        while len(history_boards) < history_positions and current_index >= 0:
            hist_offset = current_index * int(position_size)
            hist_record = mm[hist_offset:hist_offset + position_size]
            if len(hist_record) != position_size:
                break
            if hist_record[38:42] != game_id_bytes:
                break
            history_boards.append(hist_record[:signature_board_bytes])
            current_index -= 1

        missing = history_positions - len(history_boards)
        if missing > 0:
            parts.append(bytes(missing * signature_board_bytes))
        parts.extend(reversed(history_boards))

    return hashlib.blake2b(b''.join(parts), digest_size=16).digest()


def _sample_signature_from_record_and_history(record, cfg, history_boards, history_positions):
    """Build the exact sample signature from a record and cached prior boards."""
    mode = cfg.get('mode', 'position_plus_move')
    if mode == "fen":
        signature_board_bytes = 38
        parts = [record[:signature_board_bytes]]
    elif mode == "fen_no_counters":
        signature_board_bytes = 34
        parts = [record[:signature_board_bytes]]
    elif mode == "pieces":
        signature_board_bytes = 32
        parts = [record[:signature_board_bytes]]
    else:
        signature_board_bytes = 34
        move_target = struct.unpack('H', record[44:46])[0]
        if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
            move_target = az_index_to_policy_index(move_target)
        parts = [record[:34], struct.pack('H', int(move_target))]

    if cfg.get('include_turn', True):
        parts.append(bytes((record[42] & 1,)))
    history_positions = max(0, int(history_positions or 0))
    if cfg.get('include_history', True) and history_positions > 0:
        missing = history_positions - len(history_boards)
        if missing > 0:
            parts.append(bytes(missing * signature_board_bytes))
        parts.extend(history_boards)
    return hashlib.blake2b(b''.join(parts), digest_size=16).digest()


def _iter_sorted_sample_signatures(mm, position_size, indices, cfg, history_positions):
    """Yield exact signatures while reusing history for nearby sorted indices."""
    history_positions = max(0, int(history_positions or 0))
    if not (cfg.get('include_history', True) and history_positions > 0):
        for idx in indices:
            index = int(idx)
            yield index, _sample_signature_for_index(mm, position_size, index, cfg, history_positions)
        return

    mode = cfg.get('mode', 'position_plus_move')
    signature_board_bytes = 38 if mode == 'fen' else 34 if mode in {'fen_no_counters', 'position_plus_move'} else 32
    history_boards = []
    active_game_id = None
    next_read_index = -1
    for idx in indices:
        target_index = int(idx)
        # Long gaps cannot share useful history.  Jump to the small lookback
        # window instead of walking every record between two selected samples.
        if next_read_index < 0 or target_index - next_read_index + 1 > history_positions + 1:
            next_read_index = max(0, target_index - history_positions)
            history_boards.clear()
            active_game_id = None

        while next_read_index <= target_index:
            offset = next_read_index * int(position_size)
            record = mm[offset:offset + position_size]
            if len(record) != position_size:
                break
            game_id = record[38:42]
            if game_id != active_game_id:
                history_boards.clear()
                active_game_id = game_id
            if next_read_index == target_index:
                yield target_index, _sample_signature_from_record_and_history(
                    record, cfg, history_boards, history_positions
                )
            history_boards.append(record[:signature_board_bytes])
            if len(history_boards) > history_positions:
                del history_boards[0]
            next_read_index += 1


def _init_sample_dedup_worker(binary_file, position_size, cfg, history_positions):
    global _SAMPLE_DEDUP_CONTEXT
    handle = open(binary_file, 'rb')
    _SAMPLE_DEDUP_CONTEXT = {
        'handle': handle,
        'mm': mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ),
        'position_size': int(position_size),
        'cfg': dict(cfg or {}),
        'history_positions': int(history_positions or 0),
    }


def _sample_dedup_signature_worker(args):
    chunk_id, indices = args
    ctx = _SAMPLE_DEDUP_CONTEXT
    indices = np.asarray(indices, dtype=np.uint32)
    keys = np.zeros((len(indices), 16), dtype=np.uint8)
    sorted_indices = len(indices) < 2 or bool(np.all(indices[1:] >= indices[:-1]))
    iterator = (
        _iter_sorted_sample_signatures(
            ctx['mm'], ctx['position_size'], indices, ctx['cfg'], ctx['history_positions']
        )
        if sorted_indices else
        (
            (int(idx), _sample_signature_for_index(
                ctx['mm'], ctx['position_size'], int(idx), ctx['cfg'], ctx['history_positions']
            ))
            for idx in indices
        )
    )
    for row, (_, key) in enumerate(iterator):
        if key is not None:
            keys[row] = np.frombuffer(key, dtype=np.uint8)
    return int(chunk_id), keys


def _dedupe_indices_by_signature(binary_file, position_size, indices, cfg, label, history_positions,
                                 cache_path=None, legacy_cache_paths=None):
    if not cfg.get('enabled', False):
        return indices

    max_count = cfg.get('max_count', 4)
    max_count = None if max_count is None else max(1, int(max_count))

    cached = _load_index_cache(cache_path, label, "sample_dedup", legacy_cache_paths)
    if cached is not None:
        return cached

    indices_array = np.asarray(indices, dtype=np.uint32)
    before = len(indices_array)
    sig_to_indices = defaultdict(list)

    def consume_keys(row_start, key_bytes):
        chunk_indices = indices_array[int(row_start):int(row_start) + len(key_bytes)]
        for idx, key in zip(chunk_indices, key_bytes):
            if np.any(key):
                sig_to_indices[key.tobytes()].append(int(idx))

    # This module imports the training runtime, so Windows spawn workers have a
    # meaningful fixed memory cost.  Four workers keep dedup CPU-parallel while
    # avoiding the 12-process RAM multiplier used by lightweight PGN parsing.
    worker_count = min(
        4,
        _resolve_positions_per_game_workers({'workers': 'auto'}, max(1, before // 100_000)),
    )
    parallel = worker_count > 1 and before >= 100_000
    if not parallel:
        with open(binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                sorted_indices = len(indices_array) < 2 or bool(np.all(indices_array[1:] >= indices_array[:-1]))
                iterator = (
                    _iter_sorted_sample_signatures(mm, position_size, indices_array, cfg, history_positions)
                    if sorted_indices else
                    ((int(idx), _sample_signature_for_index(mm, position_size, int(idx), cfg, history_positions)) for idx in indices_array)
                )
                with tqdm(total=len(indices_array), desc=f"  {label} sample_dedup", unit="pos") as pbar:
                    for idx, sig in iterator:
                        if sig is not None:
                            sig_to_indices[sig].append(int(idx))
                        pbar.update(1)
            finally:
                mm.close()
    else:
        # Each pending result contains only 150k x 16 B of exact keys.  The
        # parent merges chunks in input order, preserving the old median/max
        # count semantics without holding a second full key array in memory.
        chunk_size = 150_000
        chunks = [
            (chunk_id, start, np.asarray(indices_array[start:start + chunk_size], dtype=np.uint32))
            for chunk_id, start in enumerate(range(0, before, chunk_size))
        ]
        next_chunk = 0
        ready = {}
        pending = {}
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_sample_dedup_worker,
            initargs=(binary_file, int(position_size), dict(cfg), int(history_positions or 0)),
        ) as executor:
            def submit_next():
                nonlocal next_chunk
                if next_chunk >= len(chunks):
                    return False
                chunk_id, _, chunk_indices = chunks[next_chunk]
                pending[executor.submit(_sample_dedup_signature_worker, (chunk_id, chunk_indices))] = chunk_id
                next_chunk += 1
                return True

            for _ in range(min(worker_count, len(chunks))):
                submit_next()
            with tqdm(total=before, desc=f"  {label} sample_dedup", unit="pos") as pbar:
                expected_chunk = 0
                while pending or ready:
                    while expected_chunk in ready:
                        row_start, keys = ready.pop(expected_chunk)
                        consume_keys(row_start, keys)
                        pbar.update(len(keys))
                        expected_chunk += 1
                    while len(pending) + len(ready) < worker_count and submit_next():
                        pass
                    if not pending:
                        continue
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        chunk_id = pending.pop(future)
                        result_chunk_id, keys = future.result()
                        if int(result_chunk_id) != int(chunk_id):
                            raise RuntimeError("sample_dedup worker returned an unexpected chunk")
                        row_start = chunks[chunk_id][1]
                        ready[chunk_id] = (row_start, keys)

    if max_count is None:
        kept = [items[0] for items in sig_to_indices.values()]
    else:
        kept = []
        for items in sig_to_indices.values():
            kept.extend(_evenly_spaced_values(items, max_count))

    selected = set(kept)
    final_indices = np.asarray([idx for idx in indices_array if int(idx) in selected], dtype=np.uint32)

    if cache_path is not None:
        _save_npy_atomic(cache_path, final_indices)
        final_indices = np.load(cache_path, mmap_mode='r')

    removed = before - len(final_indices)
    mode = cfg.get('mode', 'position_plus_move')
    print(
        f"  {label} dedup: {before:,} -> {len(final_indices):,} "
        f"(-{removed:,}, {mode}, max={max_count})"
    )
    return final_indices


def _position_signature_from_record(mm, position_size, index, record, cfg, history_positions):
    mode = cfg.get('mode', 'fen')
    valid_modes = {"fen", "fen_no_counters", "pieces"}
    if mode not in valid_modes:
        raise ValueError(f"soft_targets.mode must be one of {sorted(valid_modes)}, got: {mode}")

    if len(record) != position_size:
        return None

    board = record[:38]
    game_id = struct.unpack('I', record[38:42])[0]
    move_idx = struct.unpack('H', record[42:44])[0]

    if mode == "fen":
        signature_board_bytes = 38
    elif mode == "fen_no_counters":
        signature_board_bytes = 34
    else:
        signature_board_bytes = 32
    sig_bytes = bytes(board[:signature_board_bytes])

    if cfg.get('include_turn', True):
        sig_bytes += bytes([move_idx & 1])

    history_positions = max(0, int(history_positions or 0))
    if cfg.get('include_history', True) and history_positions > 0:
        history_boards = []
        current_index = int(index) - 1
        while len(history_boards) < history_positions and current_index >= 0:
            hist_offset = current_index * int(position_size)
            hist_record = mm[hist_offset:hist_offset + position_size]
            if len(hist_record) != position_size:
                break
            hist_game_id = struct.unpack('I', hist_record[38:42])[0]
            if hist_game_id != game_id:
                break
            history_boards.append(bytes(hist_record[:signature_board_bytes]))
            current_index -= 1

        history_boards.reverse()
        missing = history_positions - len(history_boards)
        if missing > 0:
            sig_bytes += bytes(missing * signature_board_bytes)
        for hist_board in history_boards:
            sig_bytes += hist_board

    return hashlib.blake2b(sig_bytes, digest_size=16).digest()


def _position_signature_for_index(mm, position_size, index, cfg, history_positions):
    offset = int(index) * int(position_size)
    record = mm[offset:offset + position_size]
    return _position_signature_from_record(mm, position_size, index, record, cfg, history_positions)


def _uses_compact_soft_key(cfg, history_positions):
    """Whether the configured soft key can be read without generic FEN logic."""
    return (
        str((cfg or {}).get('mode', 'fen')) == 'fen_no_counters'
        and bool((cfg or {}).get('include_turn', True))
        and not bool((cfg or {}).get('include_history', False))
        and int(history_positions or 0) <= 0
    )


def _compact_soft_key_from_record(record):
    """Exact fast path for ``fen_no_counters + turn`` soft keys."""
    if len(record) < 44:
        return None
    # MoveIdx is little-endian at bytes 42:44; only its parity encodes turn.
    return hashlib.blake2b(record[:34] + bytes([record[42] & 1]), digest_size=16).digest()


def _load_soft_target_arrays(paths, expected_len):
    if not paths:
        return None
    required = (
        'policy_indices',
        'policy_values',
        'value_wdl',
        'occurrence_count',
        'sample_weight',
        'moves_left_log',
        'policy_mass_kept',
    )
    if not all(paths.get(key) and Path(paths[key]).exists() for key in required):
        return None
    arrays = {}
    try:
        for key in required:
            arrays[key] = np.load(paths[key], mmap_mode='r')
        if any(len(arrays[key]) != expected_len for key in required):
            return None
        optional = ('value_occurrence_count', 'value_sample_weight')
        for key in optional:
            path = paths.get(key)
            if path and Path(path).exists():
                arrays[key] = np.load(path, mmap_mode='r')
                if len(arrays[key]) != expected_len:
                    arrays.pop(key, None)
    except (OSError, ValueError):
        return None
    return arrays


def _copy_soft_target_arrays(src_arrays, dst_paths):
    if not src_arrays or not dst_paths:
        return
    for path in dst_paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    for key, array in src_arrays.items():
        if key in dst_paths:
            np.save(dst_paths[key], array, allow_pickle=False)


def _allow_soft_target_legacy_migration(cfg):
    cfg = dict(cfg or {})
    if cfg.get('value_key'):
        return False
    if float(cfg.get('value_wdl_shrinkage_alpha', 1.0) or 0.0) > 0.0:
        return False
    if cfg.get('value_sample_weight'):
        return False
    return True


def _load_soft_target_arrays_with_legacy(paths, expected_len, cfg, label, legacy_paths=None):
    arrays = _validate_soft_target_arrays(_load_soft_target_arrays(paths, expected_len), cfg, label)
    if arrays is not None:
        return arrays
    if not _allow_soft_target_legacy_migration(cfg):
        return None
    for legacy in legacy_paths or ():
        arrays = _validate_soft_target_arrays(_load_soft_target_arrays(legacy, expected_len), cfg, label)
        if arrays is None:
            continue
        print(f"  - {label} soft_targets: migrated cached arrays ({expected_len:,})")
        _copy_soft_target_arrays(arrays, paths)
        return _load_soft_target_arrays(paths, expected_len) or arrays
    return None


def _soft_target_value_key_config(cfg):
    cfg = dict(cfg or {})
    value_key_cfg = dict(cfg.get('value_key', {}) or {})
    key_cfg = dict(cfg)
    key_cfg.pop('value_key', None)
    key_cfg.update(value_key_cfg)
    return key_cfg


def _soft_target_key_signature_tuple(cfg):
    cfg = dict(cfg or {})
    return (
        str(cfg.get('mode', 'fen')),
        bool(cfg.get('include_turn', True)),
        bool(cfg.get('include_history', True)),
    )


def _soft_target_keys_equivalent(policy_cfg, value_cfg):
    return _soft_target_key_signature_tuple(policy_cfg) == _soft_target_key_signature_tuple(value_cfg)


_POLICY_TARGET_MIN_KEY_COUNT = 4
_POLICY_TARGET_MIN_MOVE_COUNT = 2
_POLICY_TARGET_POWER = 1.0
_WDL_PRIOR_STRENGTH = 1.0
_COUNT_AWARE_MIN_OCCURRENCE = 3.0
_COUNT_AWARE_MAX_OCCURRENCE = 256.0
_COUNT_AWARE_OCCURRENCE_POWER = 0.90
_COUNT_AWARE_DIVERSITY_BONUS = 5.0
_COUNT_AWARE_SINGLE_MOVE_WEIGHT = 0.005
_COUNT_AWARE_MAX_SINGLE_FRACTION = 0.12


def _shrink_wdl_counts(wdl_counts, count, prior):
    wdl = np.asarray(wdl_counts, dtype=np.float64)
    total = float(count if count is not None else np.sum(wdl))
    if total <= 0.0 or float(np.sum(wdl)) <= 0.0:
        return np.asarray(prior, dtype=np.float32)
    alpha = _WDL_PRIOR_STRENGTH
    prior = np.asarray(prior, dtype=np.float64)
    prior_sum = float(np.sum(prior))
    if prior_sum <= 0.0:
        prior = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    else:
        prior = prior / prior_sum
    smoothed = (wdl + alpha * prior) / (float(np.sum(wdl)) + alpha)
    return smoothed.astype(np.float32)


def _normalize_move_distribution(move_to_mass):
    clean = {int(move): max(0.0, float(mass)) for move, mass in move_to_mass.items()}
    total = sum(clean.values())
    if total <= 0.0:
        return {}
    return {move: mass / total for move, mass in clean.items() if mass > 0.0}


def _select_and_renormalize_policy_moves(move_probs, max_policy_moves, mass_threshold, force_move=None):
    if not move_probs:
        return [], 1.0
    ordered = sorted(move_probs.items(), key=lambda item: (-float(item[1]), int(item[0])))
    selected = []
    kept_mass = 0.0
    for move, prob in ordered:
        if len(selected) >= max_policy_moves:
            break
        selected.append((int(move), float(prob)))
        kept_mass += float(prob)
        if kept_mass >= mass_threshold:
            break
    if force_move is not None and int(force_move) in move_probs and all(move != int(force_move) for move, _ in selected):
        if len(selected) >= max_policy_moves and selected:
            kept_mass -= float(selected[-1][1])
            selected[-1] = (int(force_move), float(move_probs[int(force_move)]))
            kept_mass += float(move_probs[int(force_move)])
        else:
            selected.append((int(force_move), float(move_probs[int(force_move)])))
            kept_mass += float(move_probs[int(force_move)])
    renorm = sum(prob for _, prob in selected)
    if renorm <= 0.0:
        return [], max(0.0, min(1.0, kept_mass))
    return [(move, float(prob) / renorm) for move, prob in selected], max(0.0, min(1.0, kept_mass))


def _build_policy_target_distribution(moves, hard_move_target, policy_count,
                                      max_policy_moves=32, policy_mass_threshold=1.0):
    # A soft target must be a function of the position key only.  Blending it with
    # hard_move_target makes two copies of the same position receive different
    # labels; the expectation stays empirical, but SGD sees needless label noise.
    count = max(1.0, float(policy_count))
    hard_move_target = int(hard_move_target)

    if count < _POLICY_TARGET_MIN_KEY_COUNT or not moves:
        return [(hard_move_target, 1.0)], 1.0

    reliable_moves = {
        int(move): float(count_value) ** _POLICY_TARGET_POWER
        for move, count_value in moves.items()
        if float(count_value) >= _POLICY_TARGET_MIN_MOVE_COUNT
    }
    empirical = _normalize_move_distribution(reliable_moves)
    if not empirical:
        # For a well-observed key with no repeated alternative, use the modal
        # move for every copy rather than record-specific hard labels.
        fallback = _normalize_move_distribution({
            int(move): float(count_value)
            for move, count_value in moves.items()
            if float(count_value) > 0.0
        })
        if fallback:
            modal_move = min(fallback, key=lambda move: (-float(fallback[move]), int(move)))
            return [(int(modal_move), 1.0)], 1.0
        return [(hard_move_target, 1.0)], 1.0

    selected, kept_mass = _select_and_renormalize_policy_moves(
        empirical,
        max(1, int(max_policy_moves)),
        max(0.0, min(1.0, float(policy_mass_threshold))),
    )
    if not selected:
        return [(hard_move_target, 1.0)], 1.0
    return selected, kept_mass


def _validate_soft_target_arrays(arrays, cfg, label):
    return arrays


def _resolve_soft_target_workers(cfg, chunk_count):
    return _resolve_positions_per_game_workers(cfg, chunk_count)


def _soft_target_chunk_size(cfg, source_count, workers):
    raw = cfg.get('chunk_size', 'auto')
    if isinstance(raw, str) and raw.strip().lower() in {'auto', ''}:
        target_chunks = max(1, int(workers) * 24)
        return max(50000, min(250000, int(np.ceil(float(source_count) / float(target_chunks)))))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 250000


def _soft_target_max_pending(cfg, workers, *, default_multiplier=1.0):
    try:
        raw = cfg.get('max_pending_chunks', None)
    except AttributeError:
        raw = None
    if raw is None:
        raw = max(1, int(round(float(workers) * float(default_multiplier))))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = max(1, int(round(float(workers) * float(default_multiplier))))
    return max(1, min(max(1, int(workers) * 2), value))


def _soft_target_index_chunks(indices, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for chunk_id, start in enumerate(range(0, len(indices), chunk_size)):
        yield chunk_id, np.asarray(indices[start:start + chunk_size], dtype=np.uint32)


def _key_hashes_from_bytes(key_bytes):
    key_bytes = np.asarray(key_bytes, dtype=np.uint8)
    if key_bytes.size == 0:
        return np.zeros(0, dtype=np.uint64)
    key_bytes = np.ascontiguousarray(key_bytes[:, :8])
    return key_bytes.view(np.uint64).reshape(-1)


def _sorted_unique_key_hashes(key_bytes):
    hashes = _key_hashes_from_bytes(key_bytes)
    if len(hashes) == 0:
        return hashes
    hashes = hashes[hashes != np.uint64(0)]
    if len(hashes) == 0:
        return hashes.astype(np.uint64, copy=False)
    return np.unique(hashes)


def _key_ids_from_key_bytes(key_bytes, unique_hashes):
    key_hashes = _key_hashes_from_bytes(key_bytes)
    unique_hashes = np.asarray(unique_hashes, dtype=np.uint64)
    ids = np.full((len(key_hashes),), -1, dtype=np.int32)
    if len(key_hashes) == 0 or len(unique_hashes) == 0:
        return ids
    positions = np.searchsorted(unique_hashes, key_hashes)
    in_bounds = positions < len(unique_hashes)
    valid = np.zeros((len(key_hashes),), dtype=bool)
    if np.any(in_bounds):
        valid[in_bounds] = (
            (key_hashes[in_bounds] != np.uint64(0))
            & (unique_hashes[positions[in_bounds]] == key_hashes[in_bounds])
        )
    ids[valid] = positions[valid].astype(np.int32, copy=False)
    return ids


def _soft_targets_cache_config(cfg):
    cfg = dict(cfg or {})
    cfg.pop('workers', None)
    cfg.pop('chunk_size', None)
    cfg.pop('materialize_workers', None)
    cfg.pop('materialize_chunk_size', None)
    return cfg


def _init_soft_target_aggregate_worker(binary_file, position_size, cfg, history_positions,
                                       game_length_by_id, policy_enabled, value_enabled,
                                       policy_hash_filter_path=None, value_hash_filter_path=None):
    global _SOFT_TARGET_AGGREGATE_CONTEXT
    _SOFT_TARGET_AGGREGATE_CONTEXT = {
        'binary_file': binary_file,
        'position_size': int(position_size),
        'cfg': dict(cfg or {}),
        'value_key_cfg': _soft_target_value_key_config(cfg),
        'history_positions': int(history_positions or 0),
        'use_compact_key': _uses_compact_soft_key(cfg, history_positions),
        'policy_rating_params': _policy_rating_params(cfg),
        'game_length_by_id': game_length_by_id,
        'policy_enabled': bool(policy_enabled),
        'value_enabled': bool(value_enabled),
        'policy_hash_filter': (
            np.load(policy_hash_filter_path, mmap_mode='r')
            if policy_hash_filter_path else None
        ),
        'value_hash_filter': (
            np.load(value_hash_filter_path, mmap_mode='r')
            if value_hash_filter_path else None
        ),
    }


def _policy_rating_params(cfg):
    weight_cfg = dict((cfg or {}).get('policy_rating_weight', {}) or {})
    if not weight_cfg.get('enabled', False):
        return None
    try:
        min_elo = float(weight_cfg.get('min_elo', 0) or 0)
        reference_elo = max(
            min_elo + 1.0,
            float(weight_cfg.get('reference_elo', min_elo + 1.0) or (min_elo + 1.0)),
        )
        max_multiplier = max(1.0, float(weight_cfg.get('max_multiplier', 1.0) or 1.0))
    except (TypeError, ValueError):
        return None
    return min_elo, reference_elo, max_multiplier, bool(weight_cfg.get('apply_to_value', False))


def _rating_weight_from_params(actor_elo, params):
    if params is None:
        return 1.0
    min_elo, reference_elo, max_multiplier, _ = params
    rating = float(actor_elo)
    if rating <= min_elo:
        return 1.0
    progress = max(0.0, min(1.0, (rating - min_elo) / (reference_elo - min_elo)))
    return 1.0 + (max_multiplier - 1.0) * progress


def _signature_key_array_for_indices(binary_file, position_size, indices, cfg, history_positions):
    indices = np.asarray(indices, dtype=np.uint32)
    keys = np.zeros((len(indices), 16), dtype=np.uint8)
    if len(indices) == 0:
        return keys
    use_compact_key = _uses_compact_soft_key(cfg, history_positions)
    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for row, idx in enumerate(indices):
                offset = int(idx) * int(position_size)
                record = mm[offset:offset + position_size]
                key = (
                    _compact_soft_key_from_record(record)
                    if use_compact_key else
                    _position_signature_from_record(mm, position_size, int(idx), record, cfg, history_positions)
                )
                if key is not None:
                    keys[row] = np.frombuffer(key, dtype=np.uint8)
        finally:
            mm.close()
    return keys


def _soft_target_key_worker(args):
    row_start, indices, binary_file, position_size, cfg, value_key_cfg, history_positions = args
    final_keys = _signature_key_array_for_indices(
        binary_file,
        position_size,
        indices,
        cfg,
        history_positions,
    )
    if _soft_target_keys_equivalent(cfg, value_key_cfg):
        final_value_keys = None
    else:
        final_value_keys = _signature_key_array_for_indices(
            binary_file,
            position_size,
            indices,
            value_key_cfg,
            history_positions,
        )
    return int(row_start), final_keys, final_value_keys


def _soft_target_array_aggregate_worker(args):
    chunk_id, indices = args
    ctx = _SOFT_TARGET_AGGREGATE_CONTEXT
    binary_file = ctx['binary_file']
    position_size = int(ctx['position_size'])
    cfg = ctx['cfg']
    history_positions = ctx['history_positions']
    use_compact_key = ctx['use_compact_key']
    rating_params = ctx['policy_rating_params']
    unique_hashes = ctx.get('policy_hash_filter')
    if unique_hashes is None or len(unique_hashes) == 0:
        return (
            int(chunk_id),
            len(indices),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.float32),
        )

    key_ids = []
    move_encodings = []
    move_weights = []
    wdl_encodings = []
    wdl_weights = []
    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for raw_idx in np.asarray(indices, dtype=np.uint32):
                idx = int(raw_idx)
                offset = idx * position_size
                record = mm[offset:offset + position_size]
                if len(record) != position_size:
                    continue
                key = (
                    _compact_soft_key_from_record(record)
                    if use_compact_key else
                    _position_signature_from_record(
                        mm, position_size, idx, record, cfg, history_positions,
                    )
                )
                if key is None:
                    continue
                key_hash = np.frombuffer(key[:8], dtype=np.uint64, count=1)[0]
                pos = int(np.searchsorted(unique_hashes, key_hash))
                if pos >= len(unique_hashes) or unique_hashes[pos] != key_hash:
                    continue

                move_target = struct.unpack('H', record[44:46])[0]
                if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
                    move_target = az_index_to_policy_index(move_target)
                outcome = float(struct.unpack('f', record[46:50])[0])
                wdl_bucket = 0 if outcome > 0.9 else 2 if outcome < -0.9 else 1

                key_ids.append(pos)
                move_encodings.append((int(pos) * ACTION_SIZE) + int(move_target))
                actor_elo = struct.unpack('H', record[50:52])[0] if len(record) >= 52 else 0
                rating_weight = _rating_weight_from_params(actor_elo, rating_params)
                move_weights.append(rating_weight)
                wdl_encodings.append((int(pos) * 3) + int(wdl_bucket))
                wdl_weights.append(
                    rating_weight if rating_params is not None and rating_params[3] else 1.0
                )
        finally:
            mm.close()

    if not key_ids:
        return (
            int(chunk_id),
            len(indices),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.float32),
        )

    key_ids = np.asarray(key_ids, dtype=np.int32)
    unique_key_ids, key_counts = np.unique(key_ids, return_counts=True)

    move_encodings = np.asarray(move_encodings, dtype=np.uint64)
    unique_moves, inverse = np.unique(move_encodings, return_inverse=True)
    move_counts = np.bincount(
        inverse,
        weights=np.asarray(move_weights, dtype=np.float64),
    ).astype(np.float32, copy=False)

    wdl_encodings = np.asarray(wdl_encodings, dtype=np.uint64)
    unique_wdl, wdl_inverse = np.unique(wdl_encodings, return_inverse=True)
    wdl_counts = np.bincount(
        wdl_inverse,
        weights=np.asarray(wdl_weights, dtype=np.float64),
    ).astype(np.float32, copy=False)
    return (
        int(chunk_id),
        len(indices),
        unique_key_ids.astype(np.int32, copy=False),
        key_counts.astype(np.uint32, copy=False),
        unique_wdl.astype(np.uint64, copy=False),
        wdl_counts,
        unique_moves.astype(np.uint64, copy=False),
        move_counts,
    )


def _build_policy_move_csr_from_map(move_count_map, key_count):
    key_count = int(key_count)
    if not move_count_map:
        return {
            'policy_offsets': np.zeros((key_count + 1,), dtype=np.uint64),
            'policy_moves': np.zeros(0, dtype=np.int16),
            'policy_move_counts': np.zeros(0, dtype=np.float32),
        }
    encoded = np.fromiter(move_count_map.keys(), dtype=np.uint64, count=len(move_count_map))
    counts = np.fromiter(move_count_map.values(), dtype=np.float64, count=len(move_count_map))
    order = np.argsort(encoded, kind='stable')
    encoded = encoded[order]
    counts = counts[order]
    key_ids = (encoded // np.uint64(ACTION_SIZE)).astype(np.int64, copy=False)
    moves = (encoded % np.uint64(ACTION_SIZE)).astype(np.int16, copy=False)
    per_key = np.bincount(key_ids, minlength=key_count).astype(np.uint64, copy=False)
    offsets = np.zeros((key_count + 1,), dtype=np.uint64)
    np.cumsum(per_key, out=offsets[1:])
    return {
        'policy_offsets': offsets,
        'policy_moves': moves,
        'policy_move_counts': counts.astype(np.float32, copy=False),
    }


def _aggregate_soft_targets_array_backend(binary_file, position_size, source_indices_array, chunks, workers,
                                          cfg, label, history_positions, unique_hashes,
                                          final_key_ids):
    key_count = int(len(unique_hashes))
    policy_total = np.zeros((key_count,), dtype=np.uint32)
    value_total = np.zeros((key_count,), dtype=np.uint32)
    wdl_counts = np.zeros((key_count, 3), dtype=np.float32)
    move_count_map = defaultdict(float)
    if key_count <= 0:
        return {
            'backend': 'array',
            'unique_hashes': np.asarray(unique_hashes, dtype=np.uint64),
            'final_key_ids': np.asarray(final_key_ids, dtype=np.int32),
            'policy_total': policy_total,
            'value_total': value_total,
            'wdl_counts': wdl_counts,
            'policy_offsets': np.zeros(1, dtype=np.uint64),
            'policy_moves': np.zeros(0, dtype=np.int16),
            'policy_move_counts': np.zeros(0, dtype=np.float32),
        }

    def _merge_array_result(key_ids, key_counts, wdl_encoded, wdl_chunk_counts,
                            move_encoded, move_chunk_counts):
        if len(key_ids):
            np.add.at(policy_total, key_ids, key_counts)
            np.add.at(value_total, key_ids, key_counts)
        if len(wdl_encoded):
            wdl_key_ids = (wdl_encoded // np.uint64(3)).astype(np.int64, copy=False)
            wdl_buckets = (wdl_encoded % np.uint64(3)).astype(np.int64, copy=False)
            np.add.at(wdl_counts, (wdl_key_ids, wdl_buckets), wdl_chunk_counts)
        for encoded, count in zip(move_encoded, move_chunk_counts):
            move_count_map[int(encoded)] += int(count)

    hash_filter_path = _save_soft_target_hash_filter(unique_hashes, label, "array_filter")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_soft_target_aggregate_worker,
            initargs=(
                binary_file,
                int(position_size),
                dict(cfg),
                int(history_positions or 0),
                None,
                True,
                True,
                hash_filter_path,
                hash_filter_path,
            ),
        ) as executor:
            task_iter = iter((chunk_id, indices) for chunk_id, indices in chunks)
            pending = set()
            max_pending = _soft_target_max_pending(cfg, workers, default_multiplier=1.25)
            merged_chunks = 0
            for _ in range(max_pending):
                try:
                    pending.add(executor.submit(_soft_target_array_aggregate_worker, next(task_iter)))
                except StopIteration:
                    break
            with tqdm(total=len(source_indices_array), desc=f"  {label} soft_targets aggregate", unit="pos") as pbar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        (
                            _chunk_id,
                            processed,
                            key_ids,
                            key_counts,
                            wdl_encoded,
                            wdl_chunk_counts,
                            move_encoded,
                            move_chunk_counts,
                        ) = future.result()
                        _merge_array_result(
                            key_ids,
                            key_counts,
                            wdl_encoded,
                            wdl_chunk_counts,
                            move_encoded,
                            move_chunk_counts,
                        )
                        merged_chunks += 1
                        if merged_chunks % max(8, int(workers) * 2) == 0:
                            gc.collect()
                        pbar.update(int(processed))
                        try:
                            pending.add(executor.submit(_soft_target_array_aggregate_worker, next(task_iter)))
                        except StopIteration:
                            pass
    finally:
        if hash_filter_path:
            try:
                os.remove(hash_filter_path)
            except OSError:
                pass

    csr = _build_policy_move_csr_from_map(move_count_map, key_count)
    del move_count_map
    gc.collect()
    return {
        'backend': 'array',
        'unique_hashes': np.asarray(unique_hashes, dtype=np.uint64),
        'final_key_ids': np.asarray(final_key_ids, dtype=np.int32),
        'policy_total': policy_total,
        'value_total': value_total,
        'wdl_counts': wdl_counts,
        **csr,
    }


def _build_final_soft_target_key_arrays(binary_file, position_size, final_indices_array, cfg,
                                        value_key_cfg, label, history_positions, workers,
                                        chunk_size):
    final_key_bytes = np.zeros((len(final_indices_array), 16), dtype=np.uint8)
    shared_key_config = _soft_target_keys_equivalent(cfg, value_key_cfg)
    final_value_key_bytes = final_key_bytes if shared_key_config else np.zeros((len(final_indices_array), 16), dtype=np.uint8)
    if len(final_indices_array) == 0:
        return final_key_bytes, final_value_key_bytes

    row_chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(final_indices_array), chunk_size):
        end = min(len(final_indices_array), start + chunk_size)
        row_chunks.append((start, np.asarray(final_indices_array[start:end], dtype=np.uint32)))

    if workers <= 1 or len(row_chunks) <= 1:
        iterator = tqdm(row_chunks, desc=f"  {label} soft_targets final keys", unit="chunk")
        for start, indices in iterator:
            end = start + len(indices)
            keys = _signature_key_array_for_indices(
                binary_file,
                position_size,
                indices,
                cfg,
                history_positions,
            )
            final_key_bytes[start:end] = keys
            if not shared_key_config:
                final_value_key_bytes[start:end] = _signature_key_array_for_indices(
                    binary_file,
                    position_size,
                    indices,
                    value_key_cfg,
                    history_positions,
                )
        return final_key_bytes, final_value_key_bytes

    def _iter_args():
        for start, indices in row_chunks:
            yield (
                start,
                indices,
                binary_file,
                int(position_size),
                dict(cfg),
                dict(value_key_cfg),
                int(history_positions or 0),
            )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        task_iter = iter(_iter_args())
        max_pending = _soft_target_max_pending(cfg, workers, default_multiplier=1.0)
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(_soft_target_key_worker, next(task_iter)))
            except StopIteration:
                break
        with tqdm(total=len(final_indices_array), desc=f"  {label} soft_targets final keys", unit="pos") as pbar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    start, keys, value_keys = future.result()
                    end = start + len(keys)
                    final_key_bytes[start:end] = keys
                    if not shared_key_config:
                        final_value_key_bytes[start:end] = value_keys
                    pbar.update(end - start)
                    try:
                        pending.add(executor.submit(_soft_target_key_worker, next(task_iter)))
                    except StopIteration:
                        pass

    return final_key_bytes, final_value_key_bytes


def _save_soft_target_hash_filter(hashes, label, suffix):
    if hashes is None:
        return None
    handle = tempfile.NamedTemporaryFile(
        prefix=f"il_soft_{label}_{suffix}_",
        suffix=".npy",
        delete=False,
    )
    path = handle.name
    handle.close()
    np.save(path, np.asarray(hashes, dtype=np.uint64), allow_pickle=False)
    return path


def _aggregate_soft_targets_parallel(binary_file, position_size, source_indices, final_indices, cfg, label,
                                     history_positions, game_length_by_id,
                                     policy_enabled, value_enabled):
    source_indices_array = np.asarray(source_indices, dtype=np.uint32)
    final_indices_array = np.asarray(final_indices, dtype=np.uint32)
    if len(source_indices_array) == 0:
        return (
            defaultdict(lambda: defaultdict(float)),
            defaultdict(lambda: np.zeros(3, dtype=np.float64)),
            defaultdict(float),
            defaultdict(float),
            defaultdict(list),
            np.zeros((len(final_indices_array), 16), dtype=np.uint8),
            np.zeros((len(final_indices_array), 16), dtype=np.uint8),
        )

    preliminary_workers = _resolve_soft_target_workers(cfg, len(source_indices_array))
    chunk_size = _soft_target_chunk_size(cfg, len(source_indices_array), preliminary_workers)
    chunks = list(_soft_target_index_chunks(source_indices_array, chunk_size))
    workers = _resolve_soft_target_workers(cfg, len(chunks))

    value_key_cfg = _soft_target_value_key_config(cfg)
    final_key_workers = _resolve_soft_target_workers(cfg, len(final_indices_array))
    final_key_chunk_size = _soft_target_chunk_size(cfg, len(final_indices_array), final_key_workers)
    final_key_bytes, final_value_key_bytes = _build_final_soft_target_key_arrays(
        binary_file,
        position_size,
        final_indices_array,
        cfg,
        value_key_cfg,
        label,
        history_positions,
        final_key_workers,
        final_key_chunk_size,
    )
    policy_hash_filter = _sorted_unique_key_hashes(final_key_bytes) if policy_enabled else None
    shared_key_config = _soft_target_keys_equivalent(cfg, value_key_cfg)
    value_hash_filter = (
        policy_hash_filter if shared_key_config else
        (_sorted_unique_key_hashes(final_value_key_bytes) if value_enabled else None)
    )
    use_array_backend = bool(shared_key_config and policy_enabled and value_enabled)
    if not use_array_backend:
        raise ValueError(
            "Soft target aggregation now uses the array backend only. "
            "Enable both policy/value soft targets and use identical policy/value key settings "
            "(mode/include_turn/include_history) so raw aggregation can stay fast and memory-bounded."
        )
    print(
        f"  {label} soft_targets aggregate setup: "
        f"workers={workers}, chunks={len(chunks)}, chunk_size={chunk_size:,}, "
        f"pending={_soft_target_max_pending(cfg, workers, default_multiplier=1.0)}, "
        f"source={len(source_indices_array):,}, final={len(final_indices_array):,}, "
        f"shared_policy_value_key={'on' if shared_key_config else 'off'}, "
        f"backend=array"
    )

    final_key_ids = _key_ids_from_key_bytes(final_key_bytes, policy_hash_filter)
    aggregate = _aggregate_soft_targets_array_backend(
        binary_file,
        position_size,
        source_indices_array,
        chunks,
        workers,
        cfg,
        label,
        history_positions,
        policy_hash_filter,
        final_key_ids,
    )
    return aggregate, None, None, None, defaultdict(list), final_key_bytes, final_value_key_bytes


def _resolve_soft_target_materialize_workers(cfg, chunk_count):
    materialize_cfg = dict(cfg or {})
    return _resolve_positions_per_game_workers(materialize_cfg, chunk_count)


def _soft_target_array_materialize_chunk_size(cfg, final_count, workers):
    raw = cfg.get('materialize_chunk_size', cfg.get('chunk_size', 'auto'))
    if isinstance(raw, str) and raw.strip().lower() in {'auto', ''}:
        target_chunks = max(1, int(workers) * 64)
        return max(50000, min(200000, int(np.ceil(float(final_count) / float(target_chunks)))))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 150000


def _save_soft_target_array_materialize_files(aggregate, label):
    paths = {}
    for name in (
        'final_key_ids',
        'policy_total',
        'value_total',
        'wdl_counts',
        'policy_offsets',
        'policy_moves',
        'policy_move_counts',
    ):
        handle = tempfile.NamedTemporaryFile(
            prefix=f"il_soft_{label}_mat_{name}_",
            suffix=".npy",
            delete=False,
        )
        path = handle.name
        handle.close()
        np.save(path, np.asarray(aggregate[name]), allow_pickle=False)
        paths[name] = path
    return paths


def _cleanup_temp_paths(paths):
    for path in (paths or {}).values():
        if not path:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _init_soft_target_array_materialize_worker(binary_file, position_size, cfg, max_policy_moves,
                                               policy_mass_threshold, game_length_by_id, wdl_prior,
                                               aggregate_paths):
    global _SOFT_TARGET_MATERIALIZE_CONTEXT
    position_size = int(position_size)
    record_count = os.path.getsize(binary_file) // position_size
    record_dtype = np.dtype({
        'names': ['game_id', 'move_idx', 'move_target', 'outcome'],
        'formats': ['<u4', '<u2', '<u2', '<f4'],
        'offsets': [38, 42, 44, 46],
        'itemsize': position_size,
    })
    _SOFT_TARGET_MATERIALIZE_CONTEXT = {
        'binary_file': binary_file,
        'position_size': position_size,
        'cfg': dict(cfg or {}),
        'max_policy_moves': int(max_policy_moves),
        'policy_mass_threshold': float(policy_mass_threshold),
        'game_length_by_id': game_length_by_id,
        'wdl_prior': np.asarray(wdl_prior, dtype=np.float32),
        'records': np.memmap(binary_file, dtype=record_dtype, mode='r', shape=(record_count,)),
        'aggregate': {
            name: np.load(path, mmap_mode='r')
            for name, path in (aggregate_paths or {}).items()
        },
    }


def _materialize_soft_target_array_chunk(args):
    start, end, indices = args
    ctx = _SOFT_TARGET_MATERIALIZE_CONTEXT
    cfg = ctx['cfg']
    max_policy_moves = int(ctx['max_policy_moves'])
    policy_mass_threshold = float(ctx['policy_mass_threshold'])
    game_length_by_id = ctx.get('game_length_by_id')
    wdl_prior = ctx.get('wdl_prior', np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32))
    records = ctx['records']
    aggregate = ctx['aggregate']

    indices = np.asarray(indices, dtype=np.uint32)
    row_count = len(indices)
    policy_indices = np.full((row_count, max_policy_moves), -1, dtype=np.int16)
    policy_values = np.zeros((row_count, max_policy_moves), dtype=np.float16)
    value_wdl = np.zeros((row_count, 3), dtype=np.float32)
    occurrence_count = np.ones((row_count,), dtype=np.float32)
    value_occurrence_count = np.ones((row_count,), dtype=np.float32)
    sample_weight = np.ones((row_count,), dtype=np.float32)
    value_sample_weight = np.ones((row_count,), dtype=np.float32)
    moves_left_log = np.zeros((row_count,), dtype=np.float32)
    policy_mass_kept = np.ones((row_count,), dtype=np.float32)
    if row_count <= 0:
        return start, end, (
            policy_indices,
            policy_values,
            value_wdl,
            occurrence_count,
            value_occurrence_count,
            sample_weight,
            value_sample_weight,
            moves_left_log,
            policy_mass_kept,
        )

    rows = records[indices]
    game_ids = np.asarray(rows['game_id'], dtype=np.int64)
    move_indices = np.asarray(rows['move_idx'], dtype=np.int32)
    hard_moves = np.asarray(rows['move_target'], dtype=np.int32)
    outcomes = np.asarray(rows['outcome'], dtype=np.float32)

    az_mask = (hard_moves >= int(ACTION_SIZE)) & (hard_moves < int(AZ_ACTION_SIZE))
    if np.any(az_mask):
        _, az_to_policy = get_policy_index_maps()
        converted = np.asarray(az_to_policy[hard_moves[az_mask]], dtype=np.int32)
        if np.any(converted < 0):
            bad = int(hard_moves[az_mask][np.flatnonzero(converted < 0)[0]])
            raise ValueError(f"AZ index {bad} does not correspond to a compact policy move")
        hard_moves[az_mask] = converted

    total_moves = np.zeros((row_count,), dtype=np.int32)
    if game_length_by_id is not None:
        if isinstance(game_length_by_id, np.ndarray):
            valid_game = (game_ids >= 0) & (game_ids < len(game_length_by_id))
            if np.any(valid_game):
                total_moves[valid_game] = np.asarray(game_length_by_id[game_ids[valid_game]], dtype=np.int32)
        else:
            total_moves = np.asarray(
                [_lookup_game_length(game_length_by_id, int(game_id)) for game_id in game_ids],
                dtype=np.int32,
            )
    valid_total = total_moves > 0
    if np.any(valid_total):
        remaining_plies = np.maximum(
            0.0,
            total_moves[valid_total].astype(np.float32) - move_indices[valid_total].astype(np.float32),
        )
        moves_left_log[valid_total] = np.log1p(remaining_plies).astype(np.float32, copy=False)

    final_key_ids = np.asarray(aggregate['final_key_ids'], dtype=np.int32)
    key_ids = np.asarray(final_key_ids[int(start):int(end)], dtype=np.int64)
    policy_total = aggregate['policy_total']
    value_total = aggregate['value_total']
    wdl_counts = aggregate['wdl_counts']
    policy_offsets = aggregate['policy_offsets']
    policy_moves = aggregate['policy_moves']
    policy_move_counts = aggregate['policy_move_counts']

    valid_key = (key_ids >= 0) & (key_ids < len(policy_total))
    if np.any(valid_key):
        valid_ids = key_ids[valid_key]
        occurrence_count[valid_key] = np.maximum(1.0, np.asarray(policy_total[valid_ids], dtype=np.float32))
        value_occurrence_count[valid_key] = np.maximum(1.0, np.asarray(value_total[valid_ids], dtype=np.float32))

    # Most rows have a hard policy fallback.  Fill that whole path at once;
    # only well-observed keys need the relatively expensive distribution build.
    policy_indices[:, 0] = hard_moves.astype(np.int16, copy=False)
    policy_values[:, 0] = np.float16(1.0)

    valid_wdl = valid_key.copy()
    if np.any(valid_wdl):
        valid_rows = np.flatnonzero(valid_wdl)
        valid_wdl_counts = np.asarray(wdl_counts[key_ids[valid_rows]], dtype=np.float64)
        valid_wdl[valid_rows] = np.sum(valid_wdl_counts, axis=1) > 0.0
    if np.any(valid_wdl):
        rows_with_wdl = np.flatnonzero(valid_wdl)
        counts_with_wdl = np.asarray(wdl_counts[key_ids[rows_with_wdl]], dtype=np.float64)
        value_wdl[rows_with_wdl] = (
            (counts_with_wdl + wdl_prior.astype(np.float64, copy=False).reshape(1, 3))
            / (np.sum(counts_with_wdl, axis=1, keepdims=True) + _WDL_PRIOR_STRENGTH)
        ).astype(np.float32, copy=False)
    fallback_wdl = ~valid_wdl
    if np.any(fallback_wdl):
        fallback_rows = np.flatnonzero(fallback_wdl)
        value_wdl[fallback_rows] = (
            wdl_prior.astype(np.float64, copy=False).reshape(1, 3) / (1.0 + _WDL_PRIOR_STRENGTH)
        ).astype(np.float32, copy=False)
        outcome_class = np.where(outcomes[fallback_rows] > 0.9, 0, np.where(outcomes[fallback_rows] < -0.9, 2, 1))
        value_wdl[fallback_rows, outcome_class] += 1.0 / (1.0 + _WDL_PRIOR_STRENGTH)

    soft_candidate_rows = np.flatnonzero(valid_key & (occurrence_count >= _POLICY_TARGET_MIN_KEY_COUNT))
    if len(soft_candidate_rows):
        soft_key_ids, inverse = np.unique(key_ids[soft_candidate_rows], return_inverse=True)
        template_indices = np.full((len(soft_key_ids), max_policy_moves), -1, dtype=np.int16)
        template_values = np.zeros((len(soft_key_ids), max_policy_moves), dtype=np.float16)
        template_mass = np.ones((len(soft_key_ids),), dtype=np.float32)
        template_valid = np.zeros((len(soft_key_ids),), dtype=bool)
        for template_row, key_id in enumerate(soft_key_ids):
            move_start = int(policy_offsets[key_id])
            move_end = int(policy_offsets[key_id + 1])
            if move_end <= move_start:
                continue
            moves = {
                int(policy_moves[pos]): float(policy_move_counts[pos])
                for pos in range(move_start, move_end)
                if float(policy_move_counts[pos]) > 0.0
            }
            if not moves:
                continue
            # At this support level every non-empty move map resolves from the
            # shared key alone; hard_move_target is therefore intentionally unused.
            target_moves, kept_mass = _build_policy_target_distribution(
                moves,
                0,
                float(policy_total[key_id]),
                max_policy_moves=max_policy_moves,
                policy_mass_threshold=policy_mass_threshold,
            )
            for col, (target_move_idx, target_prob) in enumerate(target_moves[:max_policy_moves]):
                template_indices[template_row, col] = int(target_move_idx)
                template_values[template_row, col] = float(target_prob)
            template_mass[template_row] = float(kept_mass)
            template_valid[template_row] = True

        matching_templates = template_valid[inverse]
        if np.any(matching_templates):
            rows = soft_candidate_rows[matching_templates]
            template_rows = inverse[matching_templates]
            policy_indices[rows] = template_indices[template_rows]
            policy_values[rows] = template_values[template_rows]
            policy_mass_kept[rows] = template_mass[template_rows]

    return start, end, (
        policy_indices,
        policy_values,
        value_wdl,
        occurrence_count,
        value_occurrence_count,
        sample_weight,
        value_sample_weight,
        moves_left_log,
        policy_mass_kept,
    )


def _materialize_soft_targets_from_array_aggregate(binary_file, position_size, final_indices,
                                                   cfg, label, max_policy_moves,
                                                   policy_mass_threshold, aggregate,
                                                   game_length_by_id=None):
    final_indices = np.asarray(final_indices, dtype=np.uint32)
    row_count = len(final_indices)
    policy_indices = np.full((row_count, max_policy_moves), -1, dtype=np.int16)
    policy_values = np.zeros((row_count, max_policy_moves), dtype=np.float16)
    value_wdl = np.zeros((row_count, 3), dtype=np.float32)
    occurrence_count = np.zeros((row_count,), dtype=np.float32)
    value_occurrence_count = np.zeros((row_count,), dtype=np.float32)
    sample_weight = np.ones((row_count,), dtype=np.float32)
    value_sample_weight = np.ones((row_count,), dtype=np.float32)
    moves_left_log = np.zeros((row_count,), dtype=np.float32)
    policy_mass_kept = np.ones((row_count,), dtype=np.float32)
    if row_count <= 0:
        return policy_indices, policy_values, value_wdl, occurrence_count, value_occurrence_count, sample_weight, value_sample_weight, moves_left_log, policy_mass_kept

    final_key_ids = np.asarray(aggregate['final_key_ids'], dtype=np.int32)
    policy_total = np.asarray(aggregate['policy_total'])
    value_total = np.asarray(aggregate['value_total'])
    wdl_counts = np.asarray(aggregate['wdl_counts'])
    policy_offsets = np.asarray(aggregate['policy_offsets'], dtype=np.uint64)
    policy_moves = np.asarray(aggregate['policy_moves'])
    policy_move_counts = np.asarray(aggregate['policy_move_counts'])

    wdl_total = np.asarray(wdl_counts, dtype=np.float64).sum(axis=0)
    if float(np.sum(wdl_total)) > 0.0:
        wdl_prior = (wdl_total / float(np.sum(wdl_total))).astype(np.float32)
    else:
        wdl_prior = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

    preliminary_workers = _resolve_soft_target_materialize_workers(cfg, row_count)
    chunk_size = _soft_target_array_materialize_chunk_size(cfg, row_count, preliminary_workers)
    row_chunks = []
    for start in range(0, row_count, int(chunk_size)):
        end = min(row_count, start + int(chunk_size))
        row_chunks.append((start, end, final_indices[start:end]))
    workers = _resolve_soft_target_materialize_workers(cfg, len(row_chunks))

    if workers > 1 and len(row_chunks) > 1:
        print(
            f"  {label} soft_targets materialize setup: "
            f"workers={workers}, chunks={len(row_chunks)}, chunk_size={int(chunk_size):,}, backend=array"
        )
        aggregate_paths = _save_soft_target_array_materialize_files(aggregate, label)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_soft_target_array_materialize_worker,
                initargs=(
                    binary_file,
                    int(position_size),
                    dict(cfg),
                    int(max_policy_moves),
                    float(policy_mass_threshold),
                    game_length_by_id,
                    wdl_prior,
                    aggregate_paths,
                ),
            ) as executor:
                pending = {}
                task_iter = iter(row_chunks)
                max_pending = _soft_target_max_pending(cfg, workers, default_multiplier=1.0)

                def submit_next():
                    try:
                        start, end, indices = next(task_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(_materialize_soft_target_array_chunk, (start, end, indices))
                    pending[future] = (start, end)
                    return True

                for _ in range(min(max_pending, len(row_chunks))):
                    submit_next()

                with tqdm(total=row_count, desc=f"  {label} soft_targets materialize", unit="pos") as pbar:
                    while pending:
                        done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                        for future in done:
                            expected_start, expected_end = pending.pop(future)
                            start, end, result = future.result()
                            if int(start) != int(expected_start) or int(end) != int(expected_end):
                                raise RuntimeError(
                                    f"{label} soft target materialize chunk returned unexpected range "
                                    f"{start}:{end}, expected {expected_start}:{expected_end}"
                                )
                            (
                                policy_indices[start:end],
                                policy_values[start:end],
                                value_wdl[start:end],
                                occurrence_count[start:end],
                                value_occurrence_count[start:end],
                                sample_weight[start:end],
                                value_sample_weight[start:end],
                                moves_left_log[start:end],
                                policy_mass_kept[start:end],
                            ) = result
                            pbar.update(end - start)
                            submit_next()
        finally:
            _cleanup_temp_paths(aggregate_paths)

        return (
            policy_indices,
            policy_values,
            value_wdl,
            occurrence_count,
            value_occurrence_count,
            sample_weight,
            value_sample_weight,
            moves_left_log,
            policy_mass_kept,
        )

    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            iterator = tqdm(range(row_count), desc=f"  {label} soft_targets materialize", unit="pos")
            for row in iterator:
                idx = int(final_indices[row])
                offset = idx * int(position_size)
                record = mm[offset:offset + int(position_size)]
                if len(record) != int(position_size):
                    continue
                game_id = struct.unpack('I', record[38:42])[0]
                move_idx = struct.unpack('H', record[42:44])[0]
                hard_move_target = struct.unpack('H', record[44:46])[0]
                if hard_move_target >= ACTION_SIZE and hard_move_target < AZ_ACTION_SIZE:
                    hard_move_target = az_index_to_policy_index(hard_move_target)
                total_moves = _lookup_game_length(game_length_by_id, game_id) if game_length_by_id is not None else 0
                if total_moves > 0:
                    remaining_plies = max(0.0, float(total_moves) - float(move_idx))
                    moves_left_log[row] = float(np.log1p(remaining_plies))

                key_id = int(final_key_ids[row]) if row < len(final_key_ids) else -1
                if key_id >= 0 and key_id < len(policy_total):
                    policy_count = max(1.0, float(policy_total[key_id]))
                    value_count = max(1.0, float(value_total[key_id]))
                else:
                    policy_count = 1.0
                    value_count = 1.0
                occurrence_count[row] = policy_count
                value_occurrence_count[row] = value_count

                moves = None
                if key_id >= 0 and key_id + 1 < len(policy_offsets):
                    start = int(policy_offsets[key_id])
                    end = int(policy_offsets[key_id + 1])
                    if end > start:
                        moves = {
                            int(policy_moves[pos]): float(policy_move_counts[pos])
                            for pos in range(start, end)
                            if float(policy_move_counts[pos]) > 0.0
                        }
                if moves:
                    target_moves, kept_mass = _build_policy_target_distribution(
                        moves,
                        hard_move_target,
                        policy_count,
                        max_policy_moves=max_policy_moves,
                        policy_mass_threshold=policy_mass_threshold,
                    )
                    policy_mass_kept[row] = float(kept_mass)
                    for col, (target_move_idx, target_prob) in enumerate(target_moves[:max_policy_moves]):
                        policy_indices[row, col] = int(target_move_idx)
                        policy_values[row, col] = float(target_prob)
                else:
                    policy_indices[row, 0] = int(hard_move_target)
                    policy_values[row, 0] = 1.0

                if key_id >= 0 and key_id < len(wdl_counts) and float(np.sum(wdl_counts[key_id])) > 0.0:
                    value_wdl[row] = _shrink_wdl_counts(wdl_counts[key_id], value_count, wdl_prior)
                else:
                    outcome = float(struct.unpack('f', record[46:50])[0])
                    hard_wdl = np.zeros(3, dtype=np.float32)
                    hard_wdl[0 if outcome > 0.9 else 2 if outcome < -0.9 else 1] = 1.0
                    value_wdl[row] = _shrink_wdl_counts(hard_wdl, 1.0, wdl_prior)
        finally:
            mm.close()

    return (
        policy_indices,
        policy_values,
        value_wdl,
        occurrence_count,
        value_occurrence_count,
        sample_weight,
        value_sample_weight,
        moves_left_log,
        policy_mass_kept,
    )


def _materialize_soft_targets_parallel(binary_file, position_size, final_indices, final_key_bytes,
                                        final_value_key_bytes, cfg, label,
                                        max_policy_moves, policy_mass_threshold,
                                        policy_counts, wdl_counts, policy_total_counts,
                                        value_total_counts, moves_left_logs,
                                        game_length_by_id=None):
    if isinstance(policy_counts, dict) and policy_counts.get('backend') == 'array':
        return _materialize_soft_targets_from_array_aggregate(
            binary_file,
            position_size,
            final_indices,
            cfg,
            label,
            max_policy_moves,
            policy_mass_threshold,
            policy_counts,
            game_length_by_id=game_length_by_id,
        )
    raise ValueError("Soft target materialization expects the array aggregate backend.")


def _soft_target_count_histogram(values, chunk_size=500_000):
    total_len = len(values) if values is not None else 0
    if total_len == 0:
        return "empty"
    bin_counts = [
        ["1", 0],
        ["2-3", 0],
        ["4-7", 0],
        ["8-15", 0],
        ["16-31", 0],
        ["32-63", 0],
        ["64+", 0],
    ]
    for start in range(0, total_len, max(1, int(chunk_size))):
        chunk = np.asarray(values[start:start + chunk_size], dtype=np.float32)
        bin_counts[0][1] += int(np.count_nonzero(chunk == 1))
        bin_counts[1][1] += int(np.count_nonzero((chunk >= 2) & (chunk < 4)))
        bin_counts[2][1] += int(np.count_nonzero((chunk >= 4) & (chunk < 8)))
        bin_counts[3][1] += int(np.count_nonzero((chunk >= 8) & (chunk < 16)))
        bin_counts[4][1] += int(np.count_nonzero((chunk >= 16) & (chunk < 32)))
        bin_counts[5][1] += int(np.count_nonzero((chunk >= 32) & (chunk < 64)))
        bin_counts[6][1] += int(np.count_nonzero(chunk >= 64))
    parts = []
    total = float(total_len)
    for label, count in bin_counts:
        parts.append(f"{label}:{count:,} ({100.0 * count / total:.1f}%)")
    return ", ".join(parts)


def _soft_target_policy_stats(policy_indices, policy_values, policy_mass_kept, chunk_size=250_000,
                              quantile_sample_rows=1_000_000):
    total_len = len(policy_values) if policy_values is not None else 0
    if total_len == 0:
        return {
            'soft_rate': 0.0,
            'entropy_mean': 0.0,
            'effective_moves': 1.0,
            'entropy_p90': 0.0,
            'top1_mass_mean': 1.0,
            'kept_mass_mean': 1.0,
            'kept_mass_min': 1.0,
            'kept_mass_p01': 1.0,
        }
    chunk_size = max(1, int(chunk_size))
    sample_step = max(1, int(np.ceil(total_len / max(1, int(quantile_sample_rows)))))
    soft_rows = 0
    entropy_sum = 0.0
    top1_sum = 0.0
    kept_sum = 0.0
    kept_min = 1.0
    entropy_samples = []
    kept_samples = []
    for start in range(0, total_len, chunk_size):
        end = min(total_len, start + chunk_size)
        indices_chunk = np.asarray(policy_indices[start:end])
        values_chunk = np.asarray(policy_values[start:end], dtype=np.float32)
        valid = indices_chunk >= 0
        positive_mask = values_chunk > 0.0
        support = np.count_nonzero(valid & positive_mask, axis=1)
        safe_values = np.where(positive_mask, values_chunk, 1.0)
        entropy = -np.sum(
            np.where(positive_mask, values_chunk * np.log(safe_values), 0.0),
            axis=1,
        )
        kept = np.asarray(policy_mass_kept[start:end], dtype=np.float32)
        soft_rows += int(np.count_nonzero(support > 1))
        entropy_sum += float(np.sum(entropy, dtype=np.float64))
        top1_sum += float(np.sum(np.max(values_chunk, axis=1), dtype=np.float64))
        kept_sum += float(np.sum(kept, dtype=np.float64))
        if len(kept):
            kept_min = min(kept_min, float(np.min(kept)))
        local_offset = start % sample_step
        sample_start = (sample_step - local_offset) % sample_step
        if sample_start < len(entropy):
            entropy_samples.append(entropy[sample_start::sample_step].astype(np.float32, copy=False))
            kept_samples.append(kept[sample_start::sample_step].astype(np.float32, copy=False))
    if entropy_samples:
        entropy_sample = np.concatenate(entropy_samples)
    else:
        entropy_sample = np.asarray([], dtype=np.float32)
    if kept_samples:
        kept_sample = np.concatenate(kept_samples)
    else:
        kept_sample = np.asarray([], dtype=np.float32)
    entropy_mean = float(entropy_sum / total_len)
    return {
        'soft_rate': float(soft_rows / total_len),
        'entropy_mean': entropy_mean,
        'effective_moves': float(np.exp(entropy_mean)),
        'entropy_p90': float(np.percentile(entropy_sample, 90)) if len(entropy_sample) else 0.0,
        'top1_mass_mean': float(top1_sum / total_len),
        'kept_mass_mean': float(kept_sum / total_len),
        'kept_mass_min': float(kept_min),
        'kept_mass_p01': float(np.percentile(kept_sample, 1)) if len(kept_sample) else 1.0,
    }


def _print_soft_target_array_stats(label, arrays, policy_mass_threshold=1.0):
    if not arrays:
        return
    occurrence_count = arrays.get('occurrence_count')
    policy_indices = arrays.get('policy_indices')
    policy_values = arrays.get('policy_values')
    policy_mass_kept = arrays.get('policy_mass_kept')
    if occurrence_count is None or policy_indices is None or policy_values is None or policy_mass_kept is None:
        return
    policy_stats = _soft_target_policy_stats(policy_indices, policy_values, policy_mass_kept)
    occurrence_array = np.asarray(occurrence_count)
    occurrence_avg = float(occurrence_array.mean(dtype=np.float64)) if len(occurrence_array) else 0.0
    occurrence_max = float(occurrence_array.max()) if len(occurrence_array) else 0.0
    low_mass_rows = 0
    for start in range(0, len(policy_mass_kept), 500_000):
        kept_chunk = np.asarray(policy_mass_kept[start:start + 500_000], dtype=np.float32)
        low_mass_rows += int(np.count_nonzero(kept_chunk < policy_mass_threshold))
    print(
        f"    {label} occurrence: avg={occurrence_avg:.2f}, "
        f"max={occurrence_max:.0f}; count_hist: {_soft_target_count_histogram(occurrence_count)}"
    )
    print(
        f"    {label} policy_target: soft_rate={100.0 * policy_stats['soft_rate']:.1f}%, "
        f"entropy_mean={policy_stats['entropy_mean']:.3f}, "
        f"target_eff_moves={policy_stats['effective_moves']:.2f}, "
        f"top1_mass={policy_stats['top1_mass_mean']:.4f}, "
        f"entropy_p90={policy_stats['entropy_p90']:.3f}, "
        f"kept_mass_mean={policy_stats['kept_mass_mean']:.4f}, "
        f"kept_mass_min={policy_stats['kept_mass_min']:.4f}, "
        f"kept_mass_p01={policy_stats['kept_mass_p01']:.4f}, "
        f"below_threshold={low_mass_rows:,}"
    )


def _build_soft_targets(binary_file, position_size, source_indices, final_indices, cfg, label,
                        history_positions, paths, game_length_by_id=None, legacy_paths=None):
    if not cfg.get('enabled', False):
        return None

    final_indices_array = np.asarray(final_indices, dtype=np.uint32)
    policy_mass_threshold = float(cfg.get('policy_mass_threshold', 0.995) or 0.995)
    policy_mass_threshold = max(0.0, min(1.0, policy_mass_threshold))
    cached = _load_soft_target_arrays_with_legacy(
        paths,
        len(final_indices_array),
        cfg,
        label,
        legacy_paths=legacy_paths,
    )
    if cached is not None:
        print(f"  • {label} soft_targets: loaded cached arrays ({len(final_indices_array):,})")
        _print_soft_target_array_stats(label, cached, policy_mass_threshold)
        return cached

    policy_enabled = bool(cfg.get('policy', True))
    value_enabled = bool(cfg.get('value', True))
    max_policy_moves = max(1, int(cfg.get('max_policy_moves', 32)))

    (
        policy_counts,
        wdl_counts,
        policy_total_counts,
        value_total_counts,
        moves_left_logs,
        final_key_bytes,
        final_value_key_bytes,
    ) = _aggregate_soft_targets_parallel(
        binary_file,
        position_size,
        source_indices,
        final_indices_array,
        cfg,
        label,
        history_positions,
        game_length_by_id,
        policy_enabled,
        value_enabled,
    )

    (
        policy_indices,
        policy_values,
        value_wdl,
        occurrence_count,
        value_occurrence_count,
        sample_weight,
        value_sample_weight,
        moves_left_log,
        policy_mass_kept,
    ) = _materialize_soft_targets_parallel(
        binary_file,
        position_size,
        final_indices_array,
        final_key_bytes,
        final_value_key_bytes,
        cfg,
        label,
        max_policy_moves,
        policy_mass_threshold,
        policy_counts,
        wdl_counts,
        policy_total_counts,
        value_total_counts,
        moves_left_logs,
        game_length_by_id,
    )

    for path in paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(paths['policy_indices'], policy_indices, allow_pickle=False)
    np.save(paths['policy_values'], policy_values, allow_pickle=False)
    np.save(paths['value_wdl'], value_wdl, allow_pickle=False)
    np.save(paths['occurrence_count'], occurrence_count, allow_pickle=False)
    if paths.get('value_occurrence_count'):
        np.save(paths['value_occurrence_count'], value_occurrence_count, allow_pickle=False)
    np.save(paths['sample_weight'], sample_weight, allow_pickle=False)
    if paths.get('value_sample_weight'):
        np.save(paths['value_sample_weight'], value_sample_weight, allow_pickle=False)
    np.save(paths['moves_left_log'], moves_left_log, allow_pickle=False)
    np.save(paths['policy_mass_kept'], policy_mass_kept, allow_pickle=False)

    if isinstance(policy_counts, dict) and policy_counts.get('backend') == 'array':
        policy_total_for_stats = np.asarray(policy_counts.get('policy_total', []), dtype=np.float64)
        unique_positions = int(np.count_nonzero(policy_total_for_stats > 0.0))
        avg_count = (
            float(policy_total_for_stats.sum()) / max(1, unique_positions)
            if policy_total_for_stats.size else 0.0
        )
    else:
        unique_positions = len(policy_total_counts)
        avg_count = (sum(policy_total_counts.values()) / max(1, unique_positions)) if unique_positions else 0.0
    policy_stats = _soft_target_policy_stats(policy_indices, policy_values, policy_mass_kept)
    print(
        f"  {label} soft targets: {len(source_indices):,} source -> "
        f"{unique_positions:,} keys "
        f"(avg={avg_count:.2f}, soft={100.0 * policy_stats['soft_rate']:.1f}%, "
        f"target_eff={policy_stats['effective_moves']:.2f}, "
        f"top1={policy_stats['top1_mass_mean']:.4f}, "
        f"entropy={policy_stats['entropy_mean']:.3f}, kept_p01={policy_stats['kept_mass_p01']:.4f})"
    )

    del (
        policy_indices,
        policy_values,
        value_wdl,
        occurrence_count,
        value_occurrence_count,
        sample_weight,
        value_sample_weight,
        moves_left_log,
        policy_mass_kept,
        policy_counts,
        wdl_counts,
        policy_total_counts,
        value_total_counts,
        moves_left_logs,
        final_key_bytes,
        final_value_key_bytes,
    )
    gc.collect()

    return _load_soft_target_arrays(paths, len(final_indices_array))


def _ranges_to_index_array(ranges):
    ranges = list(ranges)
    total = sum(max(0, int(end) - int(start)) for start, end in ranges)
    dtype = np.uint32
    indices = np.empty(total, dtype=dtype)
    cursor = 0
    for start, end in ranges:
        count = int(end) - int(start)
        if count <= 0:
            continue
        indices[cursor:cursor + count] = np.arange(start, end, dtype=dtype)
        cursor += count
    return indices[:cursor]


def _resolve_target_positions(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "max", "all", "none"}:
            return None
        try:
            value = float(lowered.replace(",", "."))
            if 0 < value < 1_000:
                value *= 1_000_000
        except ValueError:
            return None
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _evenly_limit_indices(indices, target_count):
    target_count = int(target_count)
    if target_count <= 0 or len(indices) <= target_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=target_count)
    positions = np.rint(positions).astype(np.int64, copy=False)
    return np.asarray(indices[positions], dtype=np.uint32)


def _resolve_count_aware_selection_cfg(data_cfg):
    cfg = dict((data_cfg or {}).get('target_selection', {}) or {})
    mode = str(cfg.get('mode', 'even') or 'even').strip().lower()
    cfg['mode'] = mode
    cfg['enabled'] = mode in {'count_aware', 'count-aware', 'hybrid'}
    return cfg


def _resolve_soft_candidate_selection_cfg(selection_cfg, positions_per_game_cfg):
    cfg = dict((selection_cfg or {}).get('soft_candidate_selection', {}) or {})
    if not cfg.get('enabled', False):
        return None

    selector_cfg = dict(positions_per_game_cfg or {})
    selector_cfg['enabled'] = True
    selector_cfg['selection_mode'] = str(
        cfg.get('selection_mode', selector_cfg.get('selection_mode', 'smart')) or 'smart'
    )
    selector_cfg['min_distance'] = max(1, int(cfg.get('min_distance', 2) or 2))
    selector_cfg['max_total'] = max(1, int(cfg.get('max_total', selector_cfg.get('max_total', 40)) or 40))
    for key in ('opening_pct', 'middlegame_pct', 'endgame_pct', 'rare_or_eventful_pct'):
        if key in cfg:
            selector_cfg[key] = cfg[key]
    return selector_cfg


def _resolve_train_candidate_pool_count(target_train_count, available_train_count, selection_cfg):
    target_train_count = max(1, int(target_train_count))
    available_train_count = max(0, int(available_train_count))
    if available_train_count <= target_train_count:
        return available_train_count

    try:
        multiplier = float(selection_cfg.get('train_pool_multiplier', 1.0) or 1.0)
    except (TypeError, ValueError):
        multiplier = 1.0
    multiplier = max(1.0, multiplier)
    pool_count = int(round(target_train_count * multiplier))
    return max(target_train_count, min(available_train_count, pool_count))


def _resolve_soft_extra_candidate_count(target_train_count, base_train_count, available_extra_count, selection_cfg):
    soft_cfg = dict((selection_cfg or {}).get('soft_candidate_selection', {}) or {})
    target_train_count = max(1, int(target_train_count))
    base_train_count = max(0, int(base_train_count))
    available_extra_count = max(0, int(available_extra_count))
    if available_extra_count <= 0:
        return 0
    try:
        multiplier = float(soft_cfg.get('max_pool_multiplier', 1.0) or 1.0)
    except (TypeError, ValueError):
        multiplier = 1.0
    multiplier = max(0.0, multiplier)
    if multiplier <= 0.0:
        return 0
    hard_cap = int(round(float(target_train_count) * multiplier))
    try:
        train_pool_multiplier = max(1.0, float((selection_cfg or {}).get('train_pool_multiplier', 1.0) or 1.0))
    except (TypeError, ValueError):
        train_pool_multiplier = 1.0
    planned_pool = int(round(float(target_train_count) * train_pool_multiplier))
    missing_to_pool = max(0, planned_pool - base_train_count)
    try:
        buffer_fraction = max(0.0, float(soft_cfg.get('buffer_fraction', 0.15) or 0.0))
    except (TypeError, ValueError):
        buffer_fraction = 0.15
    buffer_rows = int(round(float(target_train_count) * buffer_fraction))
    target = missing_to_pool + buffer_rows
    if target <= 0:
        return 0
    return max(0, min(available_extra_count, hard_cap, target))


def _signature_hash_array_for_indices(binary_file, position_size, indices, cfg, history_positions):
    indices = np.asarray(indices, dtype=np.uint32)
    hashes = np.zeros((len(indices),), dtype=np.uint64)
    if len(indices) == 0:
        return hashes
    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for row, idx in enumerate(indices):
                key = _position_signature_for_index(mm, position_size, int(idx), cfg, history_positions)
                if key is not None:
                    hashes[row] = np.frombuffer(key[:8], dtype=np.uint64, count=1)[0]
        finally:
            mm.close()
    return hashes


def _signature_hash_worker(args):
    row_start, indices, binary_file, position_size, cfg, history_positions = args
    return int(row_start), _signature_hash_array_for_indices(
        binary_file,
        position_size,
        indices,
        cfg,
        history_positions,
    )


def _build_signature_hashes_parallel(binary_file, position_size, indices, cfg, label,
                                     history_positions, workers, chunk_size):
    indices = np.asarray(indices, dtype=np.uint32)
    hashes = np.zeros((len(indices),), dtype=np.uint64)
    if len(indices) == 0:
        return hashes

    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, len(indices), chunk_size):
        end = min(len(indices), start + chunk_size)
        chunks.append((start, np.asarray(indices[start:end], dtype=np.uint32)))

    if workers <= 1 or len(chunks) <= 1:
        iterator = tqdm(chunks, desc=f"  {label} count-aware candidate keys", unit="chunk")
        for start, chunk_indices in iterator:
            end = start + len(chunk_indices)
            hashes[start:end] = _signature_hash_array_for_indices(
                binary_file,
                position_size,
                chunk_indices,
                cfg,
                history_positions,
            )
        return hashes

    def _iter_args():
        for start, chunk_indices in chunks:
            yield (
                start,
                chunk_indices,
                binary_file,
                int(position_size),
                dict(cfg),
                int(history_positions or 0),
            )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        task_iter = iter(_iter_args())
        max_pending = max(1, int(workers) * 2)
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(_signature_hash_worker, next(task_iter)))
            except StopIteration:
                break
        with tqdm(total=len(indices), desc=f"  {label} count-aware candidate keys", unit="pos") as pbar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    start, chunk_hashes = future.result()
                    end = start + len(chunk_hashes)
                    hashes[start:end] = chunk_hashes
                    pbar.update(end - start)
                    try:
                        pending.add(executor.submit(_signature_hash_worker, next(task_iter)))
                    except StopIteration:
                        pass
    return hashes


def _init_count_aware_selection_worker(binary_file, position_size, cfg, history_positions,
                                       allowed_hashes_path):
    global _COUNT_AWARE_SELECTION_CONTEXT
    _COUNT_AWARE_SELECTION_CONTEXT = {
        'binary_file': binary_file,
        'position_size': int(position_size),
        'cfg': dict(cfg or {}),
        'history_positions': int(history_positions or 0),
        'allowed_hashes': np.load(allowed_hashes_path, mmap_mode='r'),
    }


def _count_matching_signature_hashes(binary_file, position_size, indices, cfg, history_positions,
                                     allowed_hashes):
    local = defaultdict(int)
    first_moves = {}
    diverse_hashes = set()
    local_move_counts = defaultdict(lambda: defaultdict(int))
    if allowed_hashes is None or len(allowed_hashes) == 0:
        return (
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.float32),
        )
    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for idx in np.asarray(indices, dtype=np.uint32):
                key = _position_signature_for_index(mm, position_size, int(idx), cfg, history_positions)
                if key is None:
                    continue
                key_hash = np.frombuffer(key[:8], dtype=np.uint64, count=1)[0]
                pos = int(np.searchsorted(allowed_hashes, key_hash))
                if pos < len(allowed_hashes) and allowed_hashes[pos] == key_hash:
                    key_hash_int = int(key_hash)
                    local[key_hash_int] += 1
                    offset = int(idx) * int(position_size)
                    move_target = struct.unpack('H', mm[offset + 44:offset + 46])[0]
                    if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
                        move_target = az_index_to_policy_index(move_target)
                    move_target = int(move_target)
                    previous = first_moves.get(key_hash_int)
                    if previous is None:
                        first_moves[key_hash_int] = move_target
                    elif previous != move_target:
                        diverse_hashes.add(key_hash_int)
                    local_move_counts[key_hash_int][move_target] += 1
        finally:
            mm.close()
    if not local:
        return (
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.float32),
        )
    key_list = list(local.keys())
    hashes = np.fromiter(key_list, dtype=np.uint64, count=len(local))
    counts = np.fromiter((local[key] for key in key_list), dtype=np.uint32, count=len(local))
    first_move_values = np.fromiter(
        (min(65534, int(first_moves.get(int(key), 65535))) for key in key_list),
        dtype=np.uint16,
        count=len(local),
    )
    diverse = np.fromiter(
        (1 if int(key) in diverse_hashes else 0 for key in key_list),
        dtype=np.uint8,
        count=len(local),
    )
    unique_moves = np.zeros((len(key_list),), dtype=np.uint8)
    top_moves = np.full((len(key_list),), 65535, dtype=np.uint16)
    top_counts = np.zeros((len(key_list),), dtype=np.uint32)
    entropy_terms = np.zeros((len(key_list),), dtype=np.float32)
    for row, key in enumerate(key_list):
        move_counts = local_move_counts.get(int(key), {})
        if not move_counts:
            continue
        unique_moves[row] = min(255, len(move_counts))
        best_move, best_count = max(move_counts.items(), key=lambda item: (int(item[1]), -int(item[0])))
        top_moves[row] = min(65534, int(best_move))
        top_counts[row] = int(best_count)
        entropy_terms[row] = float(sum(float(count) * math.log(max(1.0, float(count))) for count in move_counts.values()))
    order = np.argsort(hashes, kind='stable')
    return (
        hashes[order],
        counts[order],
        first_move_values[order],
        diverse[order],
        unique_moves[order],
        top_moves[order],
        top_counts[order],
        entropy_terms[order],
    )


def _count_aware_source_worker(args):
    chunk_id, indices = args
    ctx = _COUNT_AWARE_SELECTION_CONTEXT
    hashes, counts, first_moves, diverse, unique_moves, top_moves, top_counts, entropy_terms = _count_matching_signature_hashes(
        ctx['binary_file'],
        ctx['position_size'],
        indices,
        ctx['cfg'],
        ctx['history_positions'],
        ctx['allowed_hashes'],
    )
    return int(chunk_id), len(indices), hashes, counts, first_moves, diverse, unique_moves, top_moves, top_counts, entropy_terms


def _count_occurrences_for_candidate_hashes(binary_file, position_size, source_indices,
                                            candidate_hashes, cfg, label, history_positions,
                                            workers, chunk_size):
    source_indices = np.asarray(source_indices, dtype=np.uint32)
    candidate_hashes = np.asarray(candidate_hashes, dtype=np.uint64)
    valid_hashes = np.unique(candidate_hashes[candidate_hashes != np.uint64(0)])
    counts = np.zeros((len(valid_hashes),), dtype=np.uint32)
    first_moves = np.full((len(valid_hashes),), 65535, dtype=np.uint16)
    diverse = np.zeros((len(valid_hashes),), dtype=bool)
    unique_moves = np.ones((len(valid_hashes),), dtype=np.uint8)
    top_moves = np.full((len(valid_hashes),), 65535, dtype=np.uint16)
    top_counts = np.zeros((len(valid_hashes),), dtype=np.uint32)
    entropy_terms = np.zeros((len(valid_hashes),), dtype=np.float32)
    if len(source_indices) == 0 or len(valid_hashes) == 0:
        return valid_hashes, counts, diverse.astype(np.uint8), unique_moves, top_counts, entropy_terms

    chunk_size = max(1, int(chunk_size))
    chunks = list(_soft_target_index_chunks(source_indices, chunk_size))
    workers = max(1, int(workers))

    def _merge_hash_counts(hashes, chunk_counts, chunk_first_moves, chunk_diverse,
                           chunk_unique_moves, chunk_top_moves, chunk_top_counts,
                           chunk_entropy_terms):
        if len(hashes) == 0:
            return
        positions = np.searchsorted(valid_hashes, hashes)
        in_bounds = positions < len(valid_hashes)
        valid = np.zeros((len(hashes),), dtype=bool)
        if np.any(in_bounds):
            valid[in_bounds] = valid_hashes[positions[in_bounds]] == hashes[in_bounds]
        if np.any(valid):
            valid_positions = positions[valid]
            np.add.at(counts, valid_positions, chunk_counts[valid])
            incoming_first = np.asarray(chunk_first_moves[valid], dtype=np.uint16)
            known_incoming = incoming_first != np.uint16(65535)
            unset = first_moves[valid_positions] == np.uint16(65535)
            set_mask = unset & known_incoming
            if np.any(set_mask):
                first_moves[valid_positions[set_mask]] = incoming_first[set_mask]
            compare_mask = (~unset) & known_incoming
            if np.any(compare_mask):
                mismatch = first_moves[valid_positions[compare_mask]] != incoming_first[compare_mask]
                if np.any(mismatch):
                    diverse[valid_positions[compare_mask][mismatch]] = True
            diverse[valid_positions] |= np.asarray(chunk_diverse[valid], dtype=bool)
            incoming_unique = np.asarray(chunk_unique_moves[valid], dtype=np.uint8)
            unique_moves[valid_positions] = np.maximum(unique_moves[valid_positions], incoming_unique)
            incoming_top_moves = np.asarray(chunk_top_moves[valid], dtype=np.uint16)
            incoming_top_counts = np.asarray(chunk_top_counts[valid], dtype=np.uint32)
            existing_top = top_moves[valid_positions]
            unset_top = existing_top == np.uint16(65535)
            if np.any(unset_top):
                top_moves[valid_positions[unset_top]] = incoming_top_moves[unset_top]
                top_counts[valid_positions[unset_top]] = incoming_top_counts[unset_top]
            same_top = (~unset_top) & (existing_top == incoming_top_moves)
            if np.any(same_top):
                top_counts[valid_positions[same_top]] += incoming_top_counts[same_top]
            different_top = (~unset_top) & (existing_top != incoming_top_moves)
            if np.any(different_top):
                better = incoming_top_counts[different_top] > top_counts[valid_positions[different_top]]
                if np.any(better):
                    rows = valid_positions[different_top][better]
                    top_moves[rows] = incoming_top_moves[different_top][better]
                    top_counts[rows] = incoming_top_counts[different_top][better]
                diverse[valid_positions[different_top]] = True
                unique_moves[valid_positions[different_top]] = np.maximum(unique_moves[valid_positions[different_top]], np.uint8(2))
            entropy_terms[valid_positions] += np.asarray(chunk_entropy_terms[valid], dtype=np.float32)

    if workers <= 1 or len(chunks) <= 1:
        iterator = tqdm(chunks, desc=f"  {label} count-aware raw counts", unit="chunk")
        for _, chunk_indices in iterator:
            (
                hashes,
                chunk_counts,
                chunk_first_moves,
                chunk_diverse,
                chunk_unique_moves,
                chunk_top_moves,
                chunk_top_counts,
                chunk_entropy_terms,
            ) = _count_matching_signature_hashes(
                binary_file,
                position_size,
                chunk_indices,
                cfg,
                history_positions,
                valid_hashes,
            )
            _merge_hash_counts(
                hashes,
                chunk_counts,
                chunk_first_moves,
                chunk_diverse,
                chunk_unique_moves,
                chunk_top_moves,
                chunk_top_counts,
                chunk_entropy_terms,
            )
        return valid_hashes, counts, diverse.astype(np.uint8), unique_moves, top_counts, entropy_terms

    filter_path = _save_soft_target_hash_filter(valid_hashes, label, "count_filter")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_count_aware_selection_worker,
            initargs=(
                binary_file,
                int(position_size),
                dict(cfg),
                int(history_positions or 0),
                filter_path,
            ),
        ) as executor:
            task_iter = iter((chunk_id, indices) for chunk_id, indices in chunks)
            pending = set()
            max_pending = max(1, int(workers) * 2)
            for _ in range(max_pending):
                try:
                    pending.add(executor.submit(_count_aware_source_worker, next(task_iter)))
                except StopIteration:
                    break
            with tqdm(total=len(source_indices), desc=f"  {label} count-aware raw counts", unit="pos") as pbar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        (
                            _,
                            processed,
                            hashes,
                            chunk_counts,
                            chunk_first_moves,
                            chunk_diverse,
                            chunk_unique_moves,
                            chunk_top_moves,
                            chunk_top_counts,
                            chunk_entropy_terms,
                        ) = future.result()
                        _merge_hash_counts(
                            hashes,
                            chunk_counts,
                            chunk_first_moves,
                            chunk_diverse,
                            chunk_unique_moves,
                            chunk_top_moves,
                            chunk_top_counts,
                            chunk_entropy_terms,
                        )
                        pbar.update(int(processed))
                        try:
                            pending.add(executor.submit(_count_aware_source_worker, next(task_iter)))
                        except StopIteration:
                            pass
    finally:
        if filter_path:
            try:
                os.remove(filter_path)
            except OSError:
                pass
    return valid_hashes, counts, diverse.astype(np.uint8), unique_moves, top_counts, entropy_terms


def _candidate_occurrence_counts(binary_file, position_size, source_indices, candidate_indices,
                                 key_cfg, selection_cfg, label, history_positions):
    candidate_indices = np.asarray(candidate_indices, dtype=np.uint32)
    if len(candidate_indices) == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.float32),
            {
                'unique_moves': np.zeros(0, dtype=np.float32),
                'top1_mass': np.ones(0, dtype=np.float32),
                'entropy': np.zeros(0, dtype=np.float32),
                'effective_moves': np.ones(0, dtype=np.float32),
            },
        )
    workers = _resolve_soft_target_workers(selection_cfg, len(candidate_indices))
    chunk_size = _soft_target_chunk_size(selection_cfg, len(candidate_indices), workers)
    candidate_hashes = _build_signature_hashes_parallel(
        binary_file,
        position_size,
        candidate_indices,
        key_cfg,
        label,
        history_positions,
        workers,
        chunk_size,
    )
    source_workers = _resolve_soft_target_workers(selection_cfg, len(source_indices))
    source_chunk_size = _soft_target_chunk_size(selection_cfg, len(source_indices), source_workers)
    (
        unique_hashes,
        unique_counts,
        unique_diverse,
        unique_move_counts,
        unique_top_counts,
        unique_entropy_terms,
    ) = _count_occurrences_for_candidate_hashes(
        binary_file,
        position_size,
        source_indices,
        candidate_hashes,
        key_cfg,
        label,
        history_positions,
        source_workers,
        source_chunk_size,
    )
    occurrence = np.ones((len(candidate_indices),), dtype=np.float32)
    move_diversity = np.ones((len(candidate_indices),), dtype=np.float32)
    unique_moves = np.ones((len(candidate_indices),), dtype=np.float32)
    top1_mass = np.ones((len(candidate_indices),), dtype=np.float32)
    entropy = np.zeros((len(candidate_indices),), dtype=np.float32)
    effective_moves = np.ones((len(candidate_indices),), dtype=np.float32)
    if len(unique_hashes) > 0:
        positions = np.searchsorted(unique_hashes, candidate_hashes)
        in_bounds = (candidate_hashes != np.uint64(0)) & (positions < len(unique_hashes))
        valid = np.zeros((len(candidate_hashes),), dtype=bool)
        if np.any(in_bounds):
            valid[in_bounds] = unique_hashes[positions[in_bounds]] == candidate_hashes[in_bounds]
        occurrence[valid] = np.maximum(1, unique_counts[positions[valid]]).astype(np.float32, copy=False)
        mapped_unique = np.maximum(
            1,
            np.asarray(unique_move_counts[positions[valid]], dtype=np.uint16),
        ).astype(np.float32, copy=False)
        mapped_unique = np.maximum(
            mapped_unique,
            1.0 + np.asarray(unique_diverse[positions[valid]], dtype=np.float32),
        )
        unique_moves[valid] = mapped_unique
        move_diversity[valid] = mapped_unique
        mapped_counts = np.maximum(1.0, occurrence[valid])
        mapped_top = np.asarray(unique_top_counts[positions[valid]], dtype=np.float32)
        top1_mass[valid] = np.clip(mapped_top / mapped_counts, 0.0, 1.0)
        terms = np.asarray(unique_entropy_terms[positions[valid]], dtype=np.float32)
        ent = np.log(mapped_counts) - (terms / mapped_counts)
        ent = np.where(np.isfinite(ent) & (ent > 0.0), ent, 0.0)
        entropy[valid] = ent.astype(np.float32, copy=False)
        effective_moves[valid] = np.exp(np.minimum(ent, np.log(32.0))).astype(np.float32, copy=False)
    return occurrence, candidate_hashes, move_diversity, {
        'unique_moves': unique_moves,
        'top1_mass': top1_mass,
        'entropy': entropy,
        'effective_moves': effective_moves,
    }


def _top_rows_by_score(rows, scores, count):
    rows = np.asarray(rows, dtype=np.int64)
    if count <= 0 or len(rows) == 0:
        return np.zeros(0, dtype=np.int64)
    count = min(int(count), len(rows))
    row_scores = np.asarray(scores[rows], dtype=np.float32)
    if count >= len(rows):
        order = np.argsort(-row_scores, kind='stable')
        return rows[order]
    partition = np.argpartition(-row_scores, count - 1)[:count]
    chosen = rows[partition]
    order = np.argsort(-np.asarray(scores[chosen], dtype=np.float32), kind='stable')
    return chosen[order]


def _top_unique_hash_rows_by_score(rows, scores, key_hashes, count, oversample_factor=4.0):
    rows = np.asarray(rows, dtype=np.int64)
    if count <= 0 or len(rows) == 0:
        return np.zeros(0, dtype=np.int64)

    count = min(int(count), len(rows))
    oversample = int(np.ceil(float(count) * max(1.0, float(oversample_factor or 1.0))))
    oversample = max(count, min(len(rows), oversample))
    ordered = _top_rows_by_score(rows, scores, oversample)
    hashes = np.asarray(key_hashes[ordered], dtype=np.uint64)
    valid = hashes != np.uint64(0)
    if not np.any(valid):
        return ordered[:count]

    valid_positions = np.flatnonzero(valid)
    _, first_valid_positions = np.unique(hashes[valid], return_index=True)
    keep_positions = valid_positions[np.sort(first_valid_positions)]
    unique_rows = ordered[keep_positions]
    if len(unique_rows) >= count:
        return unique_rows[:count]

    selected = np.zeros((len(key_hashes),), dtype=bool)
    selected[unique_rows] = True
    fill_rows = ordered[~selected[ordered]]
    if len(fill_rows) == 0:
        return unique_rows
    return np.concatenate([unique_rows, fill_rows[:count - len(unique_rows)]])


def _select_count_bonus_rows(rows, scores, key_hashes, count):
    # One extra row per soft key keeps the count-aware portion diverse.
    return _top_unique_hash_rows_by_score(rows, scores, key_hashes, count, oversample_factor=2.0)


def _count_aware_limit_indices(binary_file, position_size, total_positions, source_indices, indices, target_count,
                               soft_targets_cfg, selection_cfg, label, history_positions,
                               game_length_by_id=None):
    target_count = int(target_count)
    indices = np.asarray(indices, dtype=np.uint32)
    if target_count <= 0 or len(indices) <= target_count:
        return indices
    if not selection_cfg.get('enabled', False):
        return _evenly_limit_indices(indices, target_count)

    key_cfg = dict(soft_targets_cfg or {})
    if not key_cfg.get('enabled', False):
        print(f"  - {label} target selection: count-aware disabled because soft_targets are off")
        return _evenly_limit_indices(indices, target_count)

    occurrence, candidate_hashes, move_diversity, policy_quality = _candidate_occurrence_counts(
        binary_file,
        position_size,
        source_indices,
        indices,
        key_cfg,
        selection_cfg,
        label,
        history_positions,
    )

    broad_fraction = float(selection_cfg.get('broad_fraction', 0.75) or 0.75)
    broad_fraction = max(0.0, min(1.0, broad_fraction))
    broad_count = int(round(target_count * broad_fraction))
    broad_count = max(0, min(target_count, broad_count))
    broad_rows = np.zeros(0, dtype=np.int64)
    selected = np.zeros((len(indices),), dtype=bool)
    if broad_count > 0:
        broad_positions = np.linspace(0, len(indices) - 1, num=broad_count)
        broad_rows = np.unique(np.rint(broad_positions).astype(np.int64, copy=False))
        selected[broad_rows] = True

    count_budget = target_count - int(np.count_nonzero(selected))
    if count_budget <= 0:
        return np.asarray(indices[np.sort(np.flatnonzero(selected))], dtype=np.uint32)

    clipped = np.minimum(np.maximum(occurrence, 1.0), _COUNT_AWARE_MAX_OCCURRENCE)
    scores = np.where(
        occurrence >= _COUNT_AWARE_MIN_OCCURRENCE,
        np.log1p(clipped) ** _COUNT_AWARE_OCCURRENCE_POWER,
        0.0,
    ).astype(np.float32)
    diverse_mask = move_diversity >= 2.0
    top1_mass = np.asarray(policy_quality.get('top1_mass'), dtype=np.float32)
    entropy = np.asarray(policy_quality.get('entropy'), dtype=np.float32)
    effective_moves = np.asarray(policy_quality.get('effective_moves'), dtype=np.float32)
    scores *= np.where(
        diverse_mask,
        _COUNT_AWARE_DIVERSITY_BONUS,
        _COUNT_AWARE_SINGLE_MOVE_WEIGHT,
    ).astype(np.float32)

    remaining = np.flatnonzero(~selected)
    score_positive = scores > 0.0
    positive_rows = remaining[score_positive[remaining]]

    def _choose_bonus_rows(candidate_rows, budget):
        candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
        budget = max(0, min(int(budget), len(candidate_rows)))
        if budget <= 0 or len(candidate_rows) == 0:
            return np.zeros(0, dtype=np.int64)
        return _select_count_bonus_rows(candidate_rows, scores, candidate_hashes, budget)

    single_budget_floor = int(round(count_budget * _COUNT_AWARE_MAX_SINGLE_FRACTION))
    diverse_budget = max(0, count_budget - single_budget_floor)
    diverse_rows = positive_rows[diverse_mask[positive_rows]]
    single_rows = positive_rows[~diverse_mask[positive_rows]]

    chosen_diverse = _choose_bonus_rows(diverse_rows, diverse_budget)
    selected[chosen_diverse] = True
    remaining_bonus = count_budget - len(chosen_diverse)
    if remaining_bonus > 0:
        chosen_single = _choose_bonus_rows(single_rows, remaining_bonus)
        selected[chosen_single] = True

    shortfall = target_count - int(np.count_nonzero(selected))
    if shortfall > 0:
        fill_rows = np.flatnonzero(~selected)
        fill = _top_rows_by_score(fill_rows, scores, shortfall)
        selected[fill] = True

    selected_rows = np.sort(np.flatnonzero(selected))
    if len(selected_rows) > target_count:
        selected_rows = selected_rows[:target_count]

    count_selected = selected_rows[~np.isin(selected_rows, broad_rows, assume_unique=False)]
    selected_occ = occurrence[selected_rows]
    selected_diverse = move_diversity[selected_rows] >= 2.0
    count_occ = occurrence[count_selected] if len(count_selected) else np.zeros(0, dtype=np.float32)
    count_diverse = move_diversity[count_selected] >= 2.0 if len(count_selected) else np.zeros(0, dtype=bool)
    bonus_avg_occ = float(np.mean(count_occ)) if len(count_occ) else 0.0
    positive_diverse = diverse_mask[positive_rows] if len(positive_rows) else np.zeros(0, dtype=bool)
    selected_eff = effective_moves[selected_rows] if len(selected_rows) else np.zeros(0, dtype=np.float32)
    selected_top1 = top1_mass[selected_rows] if len(selected_rows) else np.ones(0, dtype=np.float32)
    selected_entropy = entropy[selected_rows] if len(selected_rows) else np.zeros(0, dtype=np.float32)
    print(
        f"  - {label} target selection: count-aware "
        f"broad={len(broad_rows):,}, count_bonus={len(count_selected):,}, "
        f"avg_occ={float(np.mean(selected_occ)):.2f}, "
        f"bonus_avg_occ={bonus_avg_occ:.2f}, "
        f"count>=4={100.0 * float(np.mean(selected_occ >= 4.0)):.1f}%, "
        f"multi_move={100.0 * float(np.mean(selected_diverse)):.1f}%/"
        f"{100.0 * float(np.mean(count_diverse)) if len(count_diverse) else 0.0:.1f}% bonus, "
        f"available_multi={100.0 * float(np.mean(positive_diverse)) if len(positive_diverse) else 0.0:.1f}%, "
        f"sel_eff={float(np.mean(selected_eff)) if len(selected_eff) else 1.0:.2f}, "
        f"sel_top1={float(np.mean(selected_top1)) if len(selected_top1) else 1.0:.3f}, "
        f"sel_entropy={float(np.mean(selected_entropy)) if len(selected_entropy) else 0.0:.3f}"
    )
    return np.asarray(indices[selected_rows], dtype=np.uint32)


def _split_target_counts(train_count, val_count, target_total, train_split):
    train_count = int(train_count)
    val_count = int(val_count)
    available = train_count + val_count
    target_total = _resolve_target_positions(target_total)
    if target_total is None or available <= target_total:
        return train_count, val_count, False

    target_total = max(2, int(target_total))
    train_split = min(0.99, max(0.01, float(train_split or 0.85)))
    target_train = int(round(target_total * train_split))
    target_train = max(1, min(train_count, target_train))
    target_val = max(1, min(val_count, target_total - target_train))

    shortfall = target_total - (target_train + target_val)
    if shortfall > 0 and target_train < train_count:
        add = min(shortfall, train_count - target_train)
        target_train += add
        shortfall -= add
    if shortfall > 0 and target_val < val_count:
        add = min(shortfall, val_count - target_val)
        target_val += add

    return target_train, target_val, True


def _append_soft_candidate_train_pool(binary_file, position_size, total_positions, game_ranges, train_ranges,
                                      base_train_indices, positions_per_game_cfg, sample_dedup_cfg,
                                      selection_cfg, target_train_count, config, history_positions,
                                      cache_paths):
    selector_cfg = _resolve_soft_candidate_selection_cfg(selection_cfg, positions_per_game_cfg)
    if selector_cfg is None:
        return np.asarray(base_train_indices, dtype=np.uint32), 0

    base_train_indices = np.asarray(base_train_indices, dtype=np.uint32)
    if len(base_train_indices) == 0:
        return base_train_indices, 0

    selector_config = dict(config or {})
    selector_data = dict((selector_config.get('data', {}) or {}))
    selector_data['positions_per_game'] = dict(selector_cfg)
    selector_config['data'] = selector_data
    all_soft_candidates = _select_positions_per_game_all(
        binary_file,
        position_size,
        game_ranges,
        selector_cfg,
        selector_config,
        cache_path=cache_paths.get('soft_candidate_all_ppg'),
        legacy_cache_paths=[],
    )
    all_soft_candidates = np.asarray(all_soft_candidates, dtype=np.uint32)
    if len(all_soft_candidates) == 0:
        return base_train_indices, 0

    train_mask = _mask_sorted_indices_by_ranges(all_soft_candidates, train_ranges)
    soft_train_indices = np.asarray(all_soft_candidates[train_mask], dtype=np.uint32)
    if len(soft_train_indices) == 0:
        return base_train_indices, 0

    soft_train_indices = _dedupe_indices_by_signature(
        binary_file,
        position_size,
        soft_train_indices,
        sample_dedup_cfg,
        "Train soft candidates",
        history_positions,
        cache_path=cache_paths.get('soft_candidate_train_dedup'),
        legacy_cache_paths=[],
    )
    soft_train_indices = np.asarray(soft_train_indices, dtype=np.uint32)
    if len(soft_train_indices) == 0:
        return base_train_indices, 0

    workers = _resolve_soft_target_workers(selection_cfg, len(base_train_indices) + len(soft_train_indices))
    chunk_size = _soft_target_chunk_size(
        selection_cfg,
        len(base_train_indices) + len(soft_train_indices),
        workers,
    )
    existing_hashes = _build_signature_hashes_parallel(
        binary_file,
        position_size,
        base_train_indices,
        sample_dedup_cfg,
        "Train existing candidate",
        history_positions,
        workers,
        chunk_size,
    )
    extra_hashes = _build_signature_hashes_parallel(
        binary_file,
        position_size,
        soft_train_indices,
        sample_dedup_cfg,
        "Train soft extra",
        history_positions,
        workers,
        chunk_size,
    )
    existing_unique = np.unique(existing_hashes[existing_hashes != np.uint64(0)])
    valid_extra = extra_hashes != np.uint64(0)
    if len(existing_unique) > 0 and np.any(valid_extra):
        positions = np.searchsorted(existing_unique, extra_hashes[valid_extra])
        in_bounds = positions < len(existing_unique)
        duplicate = np.zeros(int(np.count_nonzero(valid_extra)), dtype=bool)
        if np.any(in_bounds):
            duplicate[in_bounds] = existing_unique[positions[in_bounds]] == extra_hashes[valid_extra][in_bounds]
        valid_rows = np.flatnonzero(valid_extra)
        valid_extra[valid_rows[duplicate]] = False

    extra_rows = np.flatnonzero(valid_extra)
    if len(extra_rows) > 0:
        _, first_positions = np.unique(extra_hashes[extra_rows], return_index=True)
        extra_rows = extra_rows[np.sort(first_positions)]
    soft_extra_indices = np.asarray(soft_train_indices[extra_rows], dtype=np.uint32)

    extra_cap = _resolve_soft_extra_candidate_count(
        target_train_count,
        len(base_train_indices),
        len(soft_extra_indices),
        selection_cfg,
    )
    if extra_cap <= 0:
        print(
            f"  - Train soft candidate pool: {len(soft_train_indices):,} candidates, "
            "0 added (cap=0)"
        )
        return base_train_indices, 0
    if len(soft_extra_indices) > extra_cap:
        soft_extra_indices = _evenly_limit_indices(soft_extra_indices, extra_cap)

    if len(soft_extra_indices) == 0:
        print(
            f"  - Train soft candidate pool: {len(soft_train_indices):,} candidates, "
            "0 new keys"
        )
        return base_train_indices, 0

    combined = np.concatenate([base_train_indices, np.asarray(soft_extra_indices, dtype=np.uint32)])
    combined = np.unique(combined).astype(np.uint32, copy=False)
    combined.sort()
    added = max(0, len(combined) - len(base_train_indices))
    print(
        f"  - Train soft candidate pool: {len(soft_train_indices):,} deduped candidates, "
        f"{len(soft_extra_indices):,} extra sampled, +{added:,} new train keys "
        f"(min_dist={selector_cfg.get('min_distance')}, max={selector_cfg.get('max_total')})"
    )
    return combined, added


def _selection_stage_label(name):
    labels = {
        'positions_per_game': 'positions/game',
        'sample_dedup': 'sample dedup',
        'soft_candidates': 'soft candidates',
        'candidate_pool': 'candidate pool',
        'epoch_samples': 'epoch samples',
        'target_positions': 'target positions',
    }
    return labels.get(str(name), str(name))


def _print_ascii_table(title, headers, rows, right_align=None):
    if not headers:
        return

    right_align = set(right_align or [])
    header_cells = [str(cell) for cell in headers]
    col_count = len(header_cells)
    normalized_rows = []
    for row in rows:
        cells = [str(cell) for cell in row]
        if len(cells) < col_count:
            cells.extend([""] * (col_count - len(cells)))
        elif len(cells) > col_count:
            cells = cells[:col_count]
        normalized_rows.append(cells)

    widths = [len(cell) for cell in header_cells]
    for row in normalized_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def border():
        return "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def line(cells, header=False):
        rendered = []
        for idx, cell in enumerate(cells):
            if not header and idx in right_align:
                rendered.append(f"{cell:>{widths[idx]}}")
            else:
                rendered.append(f"{cell:<{widths[idx]}}")
        return "| " + " | ".join(rendered) + " |"

    table_lines = [border(), line(header_cells, header=True), border()]
    table_lines.extend(line(row) for row in normalized_rows)
    table_lines.append(border())

    width = max(len(str(title)), max(len(row) for row in table_lines))
    print(f"\n{title}")
    print("-" * width)
    for row in table_lines:
        print(row)
    print()


def _print_il_selection_stage_report(stages):
    if not stages:
        return

    rows = []
    previous_total = None
    for name, train_count, val_count in stages:
        train_count = int(train_count or 0)
        val_count = int(val_count or 0)
        total = train_count + val_count
        if previous_total is None:
            kept_text = "100.00%"
            cut_text = "-"
        else:
            removed = max(0, previous_total - total)
            removed_pct = (float(removed) / float(previous_total) * 100.0) if previous_total > 0 else 0.0
            kept_pct = (float(total) / float(previous_total) * 100.0) if previous_total > 0 else 0.0
            kept_text = f"{kept_pct:.2f}%"
            cut_text = f"-{removed:,} ({removed_pct:.2f}%)"
        rows.append((
            _selection_stage_label(name),
            f"{train_count:,}",
            f"{val_count:,}",
            f"{total:,}",
            kept_text,
            cut_text,
        ))
        previous_total = total

    _print_ascii_table(
        "IL Selection Pipeline",
        ["Stage", "Train", "Val", "Total", "Kept", "Cut"],
        rows,
        right_align={1, 2, 3, 4, 5},
    )


def _selection_stages_payload(stages):
    payload = []
    previous_total = None
    for name, train_count, val_count in stages or ():
        train_count = int(train_count or 0)
        val_count = int(val_count or 0)
        total = train_count + val_count
        removed = 0 if previous_total is None else max(0, int(previous_total) - int(total))
        pct = (float(removed) / float(previous_total) * 100.0) if previous_total else 0.0
        payload.append({
            'stage': str(name),
            'train': train_count,
            'val': val_count,
            'total': total,
            'removed_from_previous': removed,
            'removed_from_previous_pct': pct,
        })
        previous_total = total
    return payload


def _positions_per_game_cache_config(cfg):
    cfg = dict(cfg or {})
    # Runtime-only knobs must not invalidate the selected-position cache.
    cfg.pop('workers', None)
    cfg.pop('chunk_size', None)
    return cfg


def _build_index_cache_paths(metadata, config):
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    encoder_id = str(model_cfg.get('input_encoder_id', 'planes_history_v1'))
    policy_codec_id = str(model_cfg.get('policy_codec_id', 'lc0_1858_v1'))
    # Preserve every existing v1 cache digest. Only non-default contracts add a
    # namespace, so a future transformer cannot reuse CNN tensors accidentally.
    contract_cache_payload = {}
    if (encoder_id, policy_codec_id) != ('planes_history_v1', 'lc0_1858_v1'):
        contract_cache_payload = {
            'input_encoder_id': encoder_id,
            'policy_codec_id': policy_codec_id,
        }
    binary_path = Path(metadata['binary_file'])
    try:
        binary_stat = binary_path.stat()
        binary_size = int(binary_stat.st_size)
        binary_mtime_ns = int(binary_stat.st_mtime_ns)
    except OSError:
        binary_size = 0
        binary_mtime_ns = 0

    binary_identity = {
        'binary_file': str(binary_path.resolve()),
        'binary_size': binary_size,
        'total_positions': int(metadata.get('total_positions', 0) or 0),
        'position_size': int(metadata.get('position_size', 0) or 0),
    }
    range_payload = {
        **binary_identity,
        'game_range_cache_schema': 1,
    }
    selector_payload = {
        **binary_identity,
        'selector_cache_schema': 4,
        'seed': int(config.get('seed', 0) or 0),
        'train_split': float(data_cfg.get('train_split', 0.85) or 0.85),
        'positions_per_game': _positions_per_game_cache_config(data_cfg.get('positions_per_game', {})),
    }
    ppg_all_payload = {
        **binary_identity,
        'ppg_all_cache_schema': 1,
        'positions_per_game': _positions_per_game_cache_config(data_cfg.get('positions_per_game', {})),
    }
    soft_candidate_selector_cfg = _resolve_soft_candidate_selection_cfg(
        data_cfg.get('target_selection', {}),
        data_cfg.get('positions_per_game', {}),
    )
    soft_candidate_payload = {
        **binary_identity,
        'soft_candidate_cache_schema': 1,
        'train_split': float(data_cfg.get('train_split', 0.85) or 0.85),
        'positions_per_game': _positions_per_game_cache_config(soft_candidate_selector_cfg or {}),
    }
    soft_candidate_dedup_payload = dict(soft_candidate_payload)
    soft_candidate_dedup_payload.update({
        'soft_candidate_dedup_schema': 1,
        **contract_cache_payload,
        'history_positions': int(model_cfg.get('history_positions', 0) or 0),
        'sample_dedup': data_cfg.get('sample_dedup', {}),
    })
    dedup_index_payload = dict(selector_payload)
    dedup_index_payload.update({
        'dedup_index_cache_schema': 1,
        **contract_cache_payload,
        'history_positions': int(model_cfg.get('history_positions', 0) or 0),
        'sample_dedup': data_cfg.get('sample_dedup', {}),
    })
    final_index_payload = dict(dedup_index_payload)
    final_index_payload.update({
        'index_cache_schema': 8,
        'history_positions': int(model_cfg.get('history_positions', 0) or 0),
        'sample_dedup': data_cfg.get('sample_dedup', {}),
        'target_positions': data_cfg.get('target_positions', 'max'),
        'target_selection': data_cfg.get('target_selection', {}),
    })
    soft_payload = dict(final_index_payload)
    soft_payload.update({
        'soft_target_cache_schema': 2,
        **contract_cache_payload,
        'action_size': int(ACTION_SIZE),
        'soft_targets': _soft_targets_cache_config(data_cfg.get('soft_targets', {})),
    })
    selector_digest = hashlib.sha1(
        json.dumps(selector_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    ppg_all_digest = hashlib.sha1(
        json.dumps(ppg_all_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    soft_candidate_digest = hashlib.sha1(
        json.dumps(soft_candidate_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    soft_candidate_dedup_digest = hashlib.sha1(
        json.dumps(soft_candidate_dedup_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    index_digest = hashlib.sha1(
        json.dumps(final_index_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    dedup_digest = hashlib.sha1(
        json.dumps(dedup_index_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    soft_digest = hashlib.sha1(
        json.dumps(soft_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    range_digest = hashlib.sha1(
        json.dumps(range_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]

    legacy_range_payload = dict(range_payload)
    legacy_range_payload['binary_mtime_ns'] = binary_mtime_ns
    legacy_selector_payload = dict(selector_payload)
    legacy_selector_payload['binary_mtime_ns'] = binary_mtime_ns
    legacy_ppg_all_payload = dict(ppg_all_payload)
    legacy_ppg_all_payload['binary_mtime_ns'] = binary_mtime_ns
    legacy_final_payload = dict(legacy_selector_payload)
    legacy_final_payload.update({
        'index_cache_schema': 4,
        'sample_dedup': data_cfg.get('sample_dedup', {}),
        'soft_targets': data_cfg.get('soft_targets', {}),
    })
    legacy_range_digest = hashlib.sha1(
        json.dumps(legacy_range_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    legacy_selector_digest = hashlib.sha1(
        json.dumps(legacy_selector_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    legacy_ppg_all_digest = hashlib.sha1(
        json.dumps(legacy_ppg_all_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    legacy_final_digest = hashlib.sha1(
        json.dumps(legacy_final_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    legacy_range_digests = [legacy_range_digest]
    legacy_selector_digests = [legacy_selector_digest]
    legacy_ppg_all_digests = [legacy_ppg_all_digest]
    legacy_final_digests = [legacy_final_digest]
    for legacy_binary_file in (str(binary_path), str(binary_path.resolve())):
        legacy_identity = {
            'binary_file': legacy_binary_file,
            'binary_size': binary_size,
            'binary_mtime_ns': binary_mtime_ns,
            'total_positions': int(metadata.get('total_positions', 0) or 0),
            'position_size': int(metadata.get('position_size', 0) or 0),
        }
        old_range_payload = {
            **legacy_identity,
            'game_range_cache_schema': 1,
        }
        old_selector_payload = {
            **legacy_identity,
            'selector_cache_schema': 4,
            'seed': int(config.get('seed', 0) or 0),
            'train_split': float(data_cfg.get('train_split', 0.85) or 0.85),
            'positions_per_game': data_cfg.get('positions_per_game', {}),
        }
        old_ppg_all_payload = {
            **legacy_identity,
            'ppg_all_cache_schema': 1,
            'positions_per_game': data_cfg.get('positions_per_game', {}),
        }
        old_final_payload = dict(old_selector_payload)
        old_final_payload.update({
            'index_cache_schema': 4,
            'sample_dedup': data_cfg.get('sample_dedup', {}),
            'soft_targets': data_cfg.get('soft_targets', {}),
        })
        for target, payload in (
            (legacy_range_digests, old_range_payload),
            (legacy_selector_digests, old_selector_payload),
            (legacy_ppg_all_digests, old_ppg_all_payload),
            (legacy_final_digests, old_final_payload),
        ):
            digest = hashlib.sha1(
                json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest()[:16]
            if digest not in target:
                target.append(digest)

    cache_dir = binary_path.parent / "index_cache"
    return {
        'cache_dir': cache_dir,
        'range_digest': range_digest,
        'selector_digest': selector_digest,
        'ppg_all_digest': ppg_all_digest,
        'soft_candidate_digest': soft_candidate_digest,
        'soft_candidate_dedup_digest': soft_candidate_dedup_digest,
        'dedup_digest': dedup_digest,
        'index_digest': index_digest,
        'soft_digest': soft_digest,
        'manifest': cache_dir / f"il_prepare_{index_digest}_{soft_digest}.json",
        'game_ranges': cache_dir / f"il_game_ranges_{range_digest}.npz",
        'legacy_game_ranges': [cache_dir / f"il_game_ranges_{digest}.npz" for digest in legacy_range_digests],
        'train': cache_dir / f"il_indices_{index_digest}_train.npy",
        'val': cache_dir / f"il_indices_{index_digest}_val.npy",
        'train_dedup': cache_dir / f"il_indices_{dedup_digest}_train_dedup.npy",
        'val_dedup': cache_dir / f"il_indices_{dedup_digest}_val_dedup.npy",
        'legacy_train': [cache_dir / f"il_indices_{digest}_train.npy" for digest in legacy_final_digests],
        'legacy_val': [cache_dir / f"il_indices_{digest}_val.npy" for digest in legacy_final_digests],
        'train_ppg': cache_dir / f"il_indices_{selector_digest}_train_ppg.npy",
        'val_ppg': cache_dir / f"il_indices_{selector_digest}_val_ppg.npy",
        'all_ppg': cache_dir / f"il_indices_{ppg_all_digest}_all_ppg.npy",
        'soft_candidate_all_ppg': cache_dir / f"il_indices_{soft_candidate_digest}_soft_candidate_all_ppg.npy",
        'soft_candidate_train_dedup': cache_dir / f"il_indices_{soft_candidate_dedup_digest}_soft_candidate_train_dedup.npy",
        'legacy_train_ppg': [cache_dir / f"il_indices_{digest}_train_ppg.npy" for digest in legacy_selector_digests],
        'legacy_val_ppg': [cache_dir / f"il_indices_{digest}_val_ppg.npy" for digest in legacy_selector_digests],
        'legacy_all_ppg': [cache_dir / f"il_indices_{digest}_all_ppg.npy" for digest in legacy_ppg_all_digests],
        'train_soft_policy_indices': cache_dir / f"il_soft_{soft_digest}_train_policy_indices.npy",
        'train_soft_policy_values': cache_dir / f"il_soft_{soft_digest}_train_policy_values.npy",
        'train_soft_value_wdl': cache_dir / f"il_soft_{soft_digest}_train_value_wdl.npy",
        'train_soft_occurrence_count': cache_dir / f"il_soft_{soft_digest}_train_occurrence_count.npy",
        'train_soft_value_occurrence_count': cache_dir / f"il_soft_{soft_digest}_train_value_occurrence_count.npy",
        'train_soft_sample_weight': cache_dir / f"il_soft_{soft_digest}_train_sample_weight.npy",
        'train_soft_value_sample_weight': cache_dir / f"il_soft_{soft_digest}_train_value_sample_weight.npy",
        'train_soft_moves_left_log': cache_dir / f"il_soft_{soft_digest}_train_moves_left_log.npy",
        'train_soft_policy_mass_kept': cache_dir / f"il_soft_{soft_digest}_train_policy_mass_kept.npy",
        'val_soft_policy_indices': cache_dir / f"il_soft_{soft_digest}_val_policy_indices.npy",
        'val_soft_policy_values': cache_dir / f"il_soft_{soft_digest}_val_policy_values.npy",
        'val_soft_value_wdl': cache_dir / f"il_soft_{soft_digest}_val_value_wdl.npy",
        'val_soft_occurrence_count': cache_dir / f"il_soft_{soft_digest}_val_occurrence_count.npy",
        'val_soft_value_occurrence_count': cache_dir / f"il_soft_{soft_digest}_val_value_occurrence_count.npy",
        'val_soft_sample_weight': cache_dir / f"il_soft_{soft_digest}_val_sample_weight.npy",
        'val_soft_value_sample_weight': cache_dir / f"il_soft_{soft_digest}_val_value_sample_weight.npy",
        'val_soft_moves_left_log': cache_dir / f"il_soft_{soft_digest}_val_moves_left_log.npy",
        'val_soft_policy_mass_kept': cache_dir / f"il_soft_{soft_digest}_val_policy_mass_kept.npy",
        'legacy_train_soft_digest': legacy_final_digests[0],
        'legacy_val_soft_digest': legacy_final_digests[0],
        'legacy_soft_digests': legacy_final_digests,
    }


def _soft_target_paths(index_cache_paths, split, legacy=False):
    keys = {
        'policy_indices': 'policy_indices',
        'policy_values': 'policy_values',
        'value_wdl': 'value_wdl',
        'occurrence_count': 'occurrence_count',
        'value_occurrence_count': 'value_occurrence_count',
        'sample_weight': 'sample_weight',
        'value_sample_weight': 'value_sample_weight',
        'moves_left_log': 'moves_left_log',
        'policy_mass_kept': 'policy_mass_kept',
    }
    if legacy:
        digest = index_cache_paths.get(f'legacy_{split}_soft_digest')
        cache_dir = Path(index_cache_paths['cache_dir'])
        return {
            key: cache_dir / f"il_soft_{digest}_{split}_{suffix}.npy"
            for key, suffix in keys.items()
        }
    prefix = f'{split}_soft'
    return {
        key: index_cache_paths[f'{prefix}_{suffix}']
        for key, suffix in keys.items()
    }


def _legacy_soft_target_paths(index_cache_paths, split):
    cache_dir = Path(index_cache_paths['cache_dir'])
    legacy_sets = []
    keys = {
        'policy_indices': 'policy_indices',
        'policy_values': 'policy_values',
        'value_wdl': 'value_wdl',
        'occurrence_count': 'occurrence_count',
        'sample_weight': 'sample_weight',
        'moves_left_log': 'moves_left_log',
        'policy_mass_kept': 'policy_mass_kept',
    }
    for digest in index_cache_paths.get('legacy_soft_digests', []):
        legacy_sets.append({
            key: cache_dir / f"il_soft_{digest}_{split}_{suffix}.npy"
            for key, suffix in keys.items()
        })
    return legacy_sets


def _save_prepare_manifest(path, metadata, config, index_cache_paths, train_indices, val_indices,
                           game_count, train_game_count, val_game_count, train_soft_targets,
                           val_soft_targets):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    payload = {
        'cache_schema': 1,
        'binary_file': str(Path(metadata['binary_file']).resolve()),
        'total_positions': int(metadata.get('total_positions', 0) or 0),
        'position_size': int(metadata.get('position_size', 0) or 0),
        'seed': int(config.get('seed', 0) or 0),
        'train_split': float(data_cfg.get('train_split', 0.85) or 0.85),
        'target_positions': data_cfg.get('target_positions', 'max'),
        'input_encoder_id': str(model_cfg.get('input_encoder_id', 'planes_history_v1')),
        'policy_codec_id': str(model_cfg.get('policy_codec_id', 'lc0_1858_v1')),
        'history_positions': int(model_cfg.get('history_positions', 0) or 0),
        'digests': {
            'game_ranges': index_cache_paths.get('range_digest'),
            'selector': index_cache_paths.get('selector_digest'),
            'ppg_all': index_cache_paths.get('ppg_all_digest'),
            'indices': index_cache_paths.get('index_digest'),
            'soft_targets': index_cache_paths.get('soft_digest'),
        },
        'counts': {
            'games': int(game_count or 0),
            'train_games': int(train_game_count or 0),
            'val_games': int(val_game_count or 0),
            'train_indices': int(len(train_indices)),
            'val_indices': int(len(val_indices)),
        },
        'enabled': {
            'positions_per_game': bool((data_cfg.get('positions_per_game', {}) or {}).get('enabled', False)),
            'sample_dedup': bool((data_cfg.get('sample_dedup', {}) or {}).get('enabled', False)),
            'soft_targets': bool((data_cfg.get('soft_targets', {}) or {}).get('enabled', False)),
        },
        'soft_targets_loaded': {
            'train': train_soft_targets is not None,
            'val': val_soft_targets is not None,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


class BinaryChessDataset(Dataset):
    """
    Memory-mapped dataset with POV and Dynamic Sliding Window support
    
    🆕 v4.2 KEY FEATURES:
    - POV (Point of View): All boards from current player's perspective
    - Dynamic Sliding Window: History assembled at load time using mmap
    - Per-game selected indices are supplied by create_dataloaders
    - GameID tracking: Efficient history reconstruction
    """
    
    def __init__(self, binary_file, indices, position_size,
                 history_positions=0, game_length_by_id=None,
                 index_cache_path=None, use_index_mmap=True,
                 soft_targets=None, soft_target_paths=None, board_dtype='float32',
                 return_numpy=False, filter_stats=None):
        """
        Args:
            binary_file: Path to binary file
            indices: List of position indices to use
            position_size: Size of each position in bytes
            history_positions: Number of history positions to include (dynamic)
            game_length_by_id: Optional dict {game_id: total game plies} for MLH targets
        """
        # ✅ ADDED v4.2.1: Input validation
        if history_positions < 0:
            raise ValueError(f"history_positions must be >= 0, got {history_positions}")
        if position_size < 52:  # Board + game/move/outcome metadata + ActorElo
            raise ValueError(f"position_size too small: {position_size} (minimum 52 bytes)")
        
        self.binary_file = binary_file
        self.position_size = position_size
        self.history_positions = history_positions
        self.game_length_by_id = game_length_by_id
        board_dtype = str(board_dtype or 'float32').strip().lower()
        self.board_np_dtype = np.float16 if board_dtype in {'float16', 'fp16', 'half'} else np.float32
        self.return_numpy = bool(return_numpy)
        self._empty_history_tensor = np.zeros((16, 8, 8), dtype=self.board_np_dtype)
        self.index_cache_path = str(index_cache_path) if index_cache_path else None
        self.use_index_mmap = bool(use_index_mmap and self.index_cache_path)
        self.soft_targets = soft_targets or None
        self.soft_target_paths = (
            {key: str(path) for key, path in soft_target_paths.items()}
            if soft_target_paths else None
        )
        self._mmap = None
        self._file = None
        input_count = len(indices)
        self.indices = self._as_index_array(indices)

        self.indices = self._finalize_indices(self.indices)
        final_count = len(self.indices)
        self.filter_stats = {
            'input_count': input_count,
            'final_count': final_count,
        }
        if filter_stats:
            self.filter_stats.update(dict(filter_stats))
            self.filter_stats['input_count'] = input_count
            self.filter_stats['final_count'] = final_count

    @staticmethod
    def _as_index_array(indices):
        if isinstance(indices, np.ndarray):
            if indices.dtype == np.uint32:
                return indices
            return indices.astype(np.uint32, copy=False)
        return np.asarray(indices, dtype=np.uint32)

    def _finalize_indices(self, indices):
        indices = self._as_index_array(indices)
        if not self.use_index_mmap:
            return indices

        cache_path = Path(self.index_cache_path)
        if cache_path.exists():
            cached = None
            try:
                cached = np.load(cache_path, mmap_mode='r')
                if len(cached) == len(indices) and cached.dtype == np.uint32:
                    return cached
            except (OSError, ValueError):
                pass
            finally:
                if cached is not None:
                    _close_numpy_mmap(cached)

        try:
            _save_npy_atomic(cache_path, indices)
            return np.load(cache_path, mmap_mode='r')
        except OSError as exc:
            print(f"  - index mmap cache disabled for this dataset: {exc}")
            self.use_index_mmap = False
            return indices

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mmap'] = None
        state['_file'] = None
        if self.use_index_mmap and self.index_cache_path:
            state['indices'] = None
        if self.soft_target_paths:
            state['soft_targets'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap = None
        self._file = None
        if not hasattr(self, 'return_numpy'):
            self.return_numpy = False
        if self.indices is None and self.use_index_mmap and self.index_cache_path:
            self.indices = np.load(self.index_cache_path, mmap_mode='r')
        if self.soft_targets is None and self.soft_target_paths:
            self.soft_targets = _load_soft_target_arrays(self.soft_target_paths, len(self.indices))
    
    def _ensure_mmap(self):
        """Open memory-mapped file (per worker)"""
        if self._mmap is None:
            self._file = open(self.binary_file, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get item with POV and dynamic sliding window history
        
        🔧 v4.3 FIXES:
        - Binary offsets updated for GameID uint32 (4 bytes instead of 2)
        - History walk iterates over raw file positions (position_idx-1, -2, …)
          which is correct: the binary file stores positions in game order, so
          adjacent positions in the file that share the same GameID belong to the
          same game. Per-game index selection only affects which positions are *returned*
          as training samples — history is still assembled from every position in
          the file, exactly as intended by the sliding-window design.
        """
        self._ensure_mmap()
        
        position_idx = int(self.indices[idx])
        offset = position_idx * self.position_size
        data = self._mmap[offset:offset + self.position_size]
        
        # 🔧 v4.5 FIXED Layout:
        # [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [ActorElo (2B)]
        compact_board = data[:38]  # 🔧 FIXED: 38 bytes (was 36)
        game_id    = struct.unpack('I', data[38:42])[0]   # 🔧 FIXED: offset +2
        move_idx   = struct.unpack('H', data[42:44])[0]   # 🔧 FIXED: offset +2
        move_target = struct.unpack('H', data[44:46])[0]  # 🔧 FIXED: offset +2
        outcome    = struct.unpack('f', data[46:50])[0]   # 🔧 FIXED: offset +2
        if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
            move_target = az_index_to_policy_index(move_target)
        total_moves = None
        if self.game_length_by_id is not None:
            total_moves = _lookup_game_length(self.game_length_by_id, game_id)
        
        # 🔧 v4.5: Validation
        if len(compact_board) != 38:
            raise ValueError(f"Invalid compact board size: {len(compact_board)} (expected 38)")
        
        # Determine whose turn it is
        turn = get_turn_from_move_idx(move_idx)
        is_black_turn = turn == chess.BLACK

        # Convert current board to tensor with POV
        board_tensor = compact_to_tensor(compact_board, flip_perspective=is_black_turn)
        
        # 🔧 v4.5: Validate tensor shape (should be 16 planes now)
        if board_tensor.shape[0] != 16:
            raise ValueError(f"Invalid board tensor shape: {board_tensor.shape} (expected (16, 8, 8))")
        
        # DYNAMIC SLIDING WINDOW: Build history by walking backwards through raw file.
        # We walk position_idx-1, position_idx-2, … and stop as soon as the GameID
        # changes (= different game) or we run out of file.  This correctly assembles
        # history regardless of which positions the per-game selector returned.
        history_tensors = []
        
        if self.history_positions > 0:
            current_offset = position_idx - 1
            collected_history = 0
            
            while collected_history < self.history_positions and current_offset >= 0:
                hist_offset = current_offset * self.position_size
                hist_data = self._mmap[hist_offset:hist_offset + self.position_size]
                
                # 🔧 v4.5 FIXED: GameID is uint32 at [38:42]
                hist_game_id = struct.unpack('I', hist_data[38:42])[0]
                
                # Stop if different game
                if hist_game_id != game_id:
                    break
                
                hist_compact_board = hist_data[:38]  # 🔧 FIXED: 38 bytes
                
                # All history boards use the CURRENT player's POV for consistency
                hist_tensor = compact_to_tensor(hist_compact_board, flip_perspective=is_black_turn)
                
                history_tensors.append(hist_tensor)  # newest first, reversed below
                collected_history += 1
                current_offset -= 1
            
            # Reverse to get oldest-first order
            history_tensors.reverse()
            
            # Pad with zeros at the front if not enough history available
            while len(history_tensors) < self.history_positions:
                history_tensors.insert(0, self._empty_history_tensor)
        
        # Stack: [oldest_history, …, newest_history, current]  →  (16*(H+1), 8, 8)
        if history_tensors:
            all_tensors = history_tensors + [board_tensor]
            stacked_board = np.concatenate(all_tensors, axis=0)
        else:
            stacked_board = board_tensor
        
        # 🔧 CRITICAL FIX: .copy() to avoid mmap non-resizable storage issue
        # DataLoader collate requires resizable tensors
        stacked_board = stacked_board.astype(self.board_np_dtype, copy=True)
        if getattr(self, 'return_numpy', False):
            result = {
                'board': stacked_board,
                'move': np.int16(move_target),
                'value': np.asarray([outcome], dtype=np.float32),
                'move_idx': np.int16(move_idx),
            }
            if total_moves is not None:
                result['total_moves'] = np.int16(total_moves)
            if self.soft_targets is not None:
                result['policy_indices'] = np.array(
                    self.soft_targets['policy_indices'][idx], dtype=np.int16, copy=True
                )
                result['policy_values'] = np.array(
                    self.soft_targets['policy_values'][idx], dtype=np.float16, copy=True
                )
                result['value_wdl'] = np.array(
                    self.soft_targets['value_wdl'][idx], dtype=np.float32, copy=True
                )
                result['occurrence_count'] = np.float32(self.soft_targets['occurrence_count'][idx])
                if 'value_occurrence_count' in self.soft_targets:
                    result['value_occurrence_count'] = np.float32(
                        self.soft_targets['value_occurrence_count'][idx]
                    )
                result['sample_weight'] = np.float32(self.soft_targets['sample_weight'][idx])
                if 'value_sample_weight' in self.soft_targets:
                    result['value_sample_weight'] = np.float32(
                        self.soft_targets['value_sample_weight'][idx]
                    )
                result['moves_left_log'] = np.float32(self.soft_targets['moves_left_log'][idx])
                result['policy_mass_kept'] = np.float32(self.soft_targets['policy_mass_kept'][idx])
            return result
        
        # 🆕 v4.4 FIX: Dodano move_idx dla move-weighted BCE loss
        if total_moves is not None:
            result = {
                'board': torch.from_numpy(stacked_board),
                'move': torch.tensor(move_target, dtype=torch.int16),
                'value': torch.tensor([outcome], dtype=torch.float32),
                'move_idx': torch.tensor(move_idx, dtype=torch.int16),
                'total_moves': torch.tensor(total_moves, dtype=torch.int16),
            }
        else:
            result = {
                'board': torch.from_numpy(stacked_board),
                'move': torch.tensor(move_target, dtype=torch.int16),
                'value': torch.tensor([outcome], dtype=torch.float32),
                'move_idx': torch.tensor(move_idx, dtype=torch.int16),
            }
        if self.soft_targets is not None:
            result['policy_indices'] = torch.from_numpy(
                np.array(self.soft_targets['policy_indices'][idx], dtype=np.int16, copy=True)
            )
            result['policy_values'] = torch.from_numpy(
                np.array(self.soft_targets['policy_values'][idx], dtype=np.float32, copy=True)
            )
            result['value_wdl'] = torch.from_numpy(
                np.array(self.soft_targets['value_wdl'][idx], dtype=np.float32, copy=True)
            )
            result['occurrence_count'] = torch.tensor(
                float(self.soft_targets['occurrence_count'][idx]),
                dtype=torch.float32,
            )
            if 'value_occurrence_count' in self.soft_targets:
                result['value_occurrence_count'] = torch.tensor(
                    float(self.soft_targets['value_occurrence_count'][idx]),
                    dtype=torch.float32,
                )
            result['sample_weight'] = torch.tensor(
                float(self.soft_targets['sample_weight'][idx]),
                dtype=torch.float32,
            )
            if 'value_sample_weight' in self.soft_targets:
                result['value_sample_weight'] = torch.tensor(
                    float(self.soft_targets['value_sample_weight'][idx]),
                    dtype=torch.float32,
                )
            result['moves_left_log'] = torch.tensor(
                float(self.soft_targets['moves_left_log'][idx]),
                dtype=torch.float32,
            )
            result['policy_mass_kept'] = torch.tensor(
                float(self.soft_targets['policy_mass_kept'][idx]),
                dtype=torch.float32,
            )
        return result
    
    def __del__(self):
        mmap_obj = getattr(self, '_mmap', None)
        file_obj = getattr(self, '_file', None)
        if mmap_obj is not None:
            mmap_obj.close()
        if file_obj is not None:
            file_obj.close()


def _il_collate_value(values, key):
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)

    if isinstance(first, np.ndarray):
        arr = np.stack(values, axis=0)
    else:
        dtype = None
        if key in {'move', 'move_idx', 'total_moves', 'policy_indices'}:
            dtype = np.int16
        elif key in {
            'value',
            'value_wdl',
            'occurrence_count',
            'value_occurrence_count',
            'sample_weight',
            'value_sample_weight',
            'moves_left_log',
            'policy_mass_kept',
        }:
            dtype = np.float32
        arr = np.asarray(values, dtype=dtype)

    if key == 'policy_values' and arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    elif key in {
        'value',
        'value_wdl',
        'occurrence_count',
        'value_occurrence_count',
        'sample_weight',
        'value_sample_weight',
        'moves_left_log',
        'policy_mass_kept',
    } and arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    elif key in {'move', 'move_idx', 'total_moves', 'policy_indices'} and arr.dtype != np.int16:
        arr = arr.astype(np.int16, copy=False)

    return torch.from_numpy(np.ascontiguousarray(arr))


def il_collate_fn(batch):
    """Fast dict collate for large IL batches with sparse soft targets."""
    if not batch or not isinstance(batch[0], dict):
        return default_collate(batch)

    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = _il_collate_value([item[key] for item in batch], key)
    return result


def il_worker_init_fn(_worker_id):
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


class NumpyRandomSampler(Sampler):
    """Random sampler that avoids PyTorch RandomSampler's large Python int list."""

    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = int(seed or 0)
        self.epoch = 0

    def __iter__(self):
        n = len(self.data_source)
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        dtype = np.uint32 if n <= np.iinfo(np.uint32).max else np.int64
        permutation = np.arange(n, dtype=dtype)
        rng.shuffle(permutation)
        for idx in permutation:
            yield int(idx)

    def __len__(self):
        return len(self.data_source)


class NumpyBlockShuffleSampler(Sampler):
    """Shuffle file-local blocks to reduce random mmap/disk seeks during IL."""

    def __init__(self, data_source, seed=0, block_size=65536, sort_indices=False):
        self.data_source = data_source
        self.seed = int(seed or 0)
        self.epoch = 0
        self.block_size = max(1, int(block_size or 65536))
        self.n = len(data_source)
        self.file_order = None
        self.order_mode = "identity"
        if sort_indices and self.n > 1:
            indices = getattr(data_source, 'indices', None)
            if indices is not None:
                order_dtype = np.uint32 if self.n <= np.iinfo(np.uint32).max else np.int64
                self.file_order = np.argsort(
                    np.asarray(indices, dtype=np.uint32),
                    kind='stable',
                ).astype(order_dtype, copy=False)
                self.order_mode = "argsort"

    def __iter__(self):
        n = self.n
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        block_count = int(np.ceil(n / float(self.block_size)))
        block_ids = np.arange(block_count, dtype=np.uint32)
        rng.shuffle(block_ids)
        for block_id in block_ids:
            start = int(block_id) * self.block_size
            end = min(n, start + self.block_size)
            if self.file_order is None:
                yield from range(start, end)
            else:
                for idx in self.file_order[start:end]:
                    yield int(idx)

    def __len__(self):
        return len(self.data_source)


def _informative_soft_rows_cache_path(data_source, min_moves, min_entropy):
    paths = getattr(data_source, 'soft_target_paths', None) or {}
    policy_path = paths.get('policy_values')
    if not policy_path:
        return None
    entropy_milli = int(round(float(min_entropy) * 1000.0))
    base = Path(policy_path)
    return base.with_name(
        f"{base.stem}_informative_m{int(min_moves)}_e{entropy_milli:04d}.npy"
    )


def _load_informative_soft_rows(path, total_rows, dtype):
    if path is None or not path.exists():
        return None
    try:
        rows = np.load(path, mmap_mode='r')
        if rows.ndim != 1 or len(rows) and (int(rows[0]) < 0 or int(rows[-1]) >= int(total_rows)):
            return None
        return np.asarray(rows, dtype=dtype)
    except (OSError, ValueError):
        return None


class SmartEpochSampler(Sampler):
    """Rotate a larger IL candidate pool into a fixed-size epoch mix.

    The pool keeps cached soft targets for many rows; each epoch reserves a
    bounded share for informative soft positions and fills the rest broadly.
    """

    def __init__(self, data_source, cfg, seed=0, num_samples=None):
        self.data_source = data_source
        self.cfg = dict(cfg or {})
        self.seed = int(seed or 0)
        self.epoch = 0
        self.n = len(data_source)
        if self.n <= 0:
            raise ValueError("SmartEpochSampler requires a non-empty data source.")
        self.num_samples = int(num_samples if num_samples is not None else self.n)
        self.num_samples = max(1, min(self.num_samples, self.n))
        self.index_dtype = np.uint32 if self.n <= np.iinfo(np.uint32).max else np.int64
        self.block_size = 65_536

        soft_targets = getattr(data_source, 'soft_targets', None) or {}
        self.soft_row_count = 0
        self.soft_rows = np.zeros(0, dtype=self.index_dtype)
        policy_values = soft_targets.get('policy_values') if isinstance(soft_targets, dict) else None
        if policy_values is not None and len(policy_values) == self.n:
            min_moves = max(2, int(self.cfg.get('min_soft_moves', 2) or 2))
            try:
                min_entropy = max(0.0, float(self.cfg.get('min_soft_entropy', 0.12) or 0.0))
            except (TypeError, ValueError):
                min_entropy = 0.12
            cache_path = _informative_soft_rows_cache_path(data_source, min_moves, min_entropy)
            cached_rows = _load_informative_soft_rows(cache_path, self.n, self.index_dtype)
            if cached_rows is not None:
                self.soft_rows = cached_rows
            else:
                row_chunks = []
                chunk_size = 250_000
                for start in range(0, self.n, chunk_size):
                    chunk = np.asarray(policy_values[start:start + chunk_size], dtype=np.float32)
                    if chunk.size == 0:
                        continue
                    support = np.count_nonzero(chunk > 0.0, axis=1)
                    safe = np.clip(chunk, 1.0e-12, 1.0)
                    entropy = -np.sum(np.where(chunk > 0.0, chunk * np.log(safe), 0.0), axis=1)
                    mask = (support >= min_moves) & (entropy >= min_entropy)
                    if np.any(mask):
                        row_chunks.append((np.flatnonzero(mask) + start).astype(self.index_dtype, copy=False))
                self.soft_rows = (
                    np.concatenate(row_chunks).astype(self.index_dtype, copy=False)
                    if row_chunks else np.zeros(0, dtype=self.index_dtype)
                )
                if cache_path is not None:
                    _save_npy_atomic(cache_path, self.soft_rows)
            self.soft_row_count = len(self.soft_rows)
        self.soft_epoch_fraction = self._soft_epoch_fraction()

    def _soft_epoch_fraction(self):
        if self.soft_row_count <= 0:
            return 0.0
        epoch_soft_capacity = self.soft_row_count / max(1.0, float(self.num_samples))
        try:
            max_soft_fraction = max(0.0, float(self.cfg.get('soft_max_fraction', 0.28)))
        except (TypeError, ValueError):
            max_soft_fraction = 0.28

        # Every informative soft row gets one chance per epoch.  The hard cap
        # prevents soft labels from taking over when the candidate pool grows.
        return max(0.0, min(max_soft_fraction, epoch_soft_capacity))

    def _draw_rows(self, rng, rows, count, excluded=None):
        count = int(count)
        rows = np.asarray(rows, dtype=self.index_dtype)
        if excluded is not None and len(rows):
            rows = rows[~excluded[rows]]
        if count <= 0 or len(rows) == 0:
            return np.zeros(0, dtype=self.index_dtype)
        count = min(count, len(rows))
        return rng.choice(rows, size=count, replace=False).astype(self.index_dtype, copy=False)

    def _broad_rows(self, rng, count, excluded=None):
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=self.index_dtype)
        unavailable = int(np.count_nonzero(excluded)) if excluded is not None else 0
        available = max(0, self.n - unavailable)
        count = min(count, available)
        if count <= 0:
            return np.zeros(0, dtype=self.index_dtype)
        if self.n == 1:
            return np.zeros(1, dtype=self.index_dtype)

        # One coprime step gives a permutation of the pool, unlike the old small
        # random offset over a fixed grid which revisited nearly the same rows.
        start = int(rng.integers(0, self.n))
        step = max(1, int(rng.integers(1, self.n)))
        while math.gcd(step, self.n) != 1:
            step = 1 if step >= self.n - 1 else step + 1

        proposal_count = count
        if unavailable:
            proposal_count = min(
                self.n,
                max(count, int(math.ceil(count * self.n / max(1, available) * 1.05))),
            )
        positions = np.arange(proposal_count, dtype=np.uint64)
        rows = ((np.uint64(start) + np.uint64(step) * positions) % np.uint64(self.n)).astype(
            self.index_dtype,
            copy=False,
        )
        if excluded is not None:
            rows = rows[~excluded[rows]]
        return rows[:count]

    def _block_order(self, rng, rows):
        rows = np.asarray(rows, dtype=self.index_dtype)
        if len(rows) <= 1:
            return rows
        rows = np.sort(rows, kind='stable')
        block_count = int(np.ceil(len(rows) / float(self.block_size)))
        block_ids = np.arange(block_count, dtype=np.uint32)
        rng.shuffle(block_ids)
        ordered = []
        for block_id in block_ids:
            start = int(block_id) * self.block_size
            end = min(len(rows), start + self.block_size)
            block = rows[start:end].copy()
            rng.shuffle(block)
            ordered.append(block)
        return np.concatenate(ordered) if ordered else rows

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        soft_count = int(round(self.num_samples * self.soft_epoch_fraction))
        broad_count = self.num_samples - soft_count
        if broad_count < 0:
            broad_count = 0

        # Broad means non-soft coverage.  Excluding the entire informative-soft
        # set makes the configured quota exact instead of letting broad draws
        # silently increase it.
        excluded = np.zeros((self.n,), dtype=bool)
        if len(self.soft_rows):
            excluded[self.soft_rows] = True
        soft_rows = self._draw_rows(rng, self.soft_rows, soft_count)
        broad_rows = self._broad_rows(rng, broad_count, excluded=excluded)
        if len(broad_rows):
            excluded[broad_rows] = True
        draws = [soft_rows, broad_rows]
        sampled = np.concatenate([item for item in draws if len(item)]) if any(len(item) for item in draws) else self._broad_rows(rng, self.num_samples)
        if len(sampled) < self.num_samples:
            fill = self._broad_rows(rng, self.num_samples - len(sampled), excluded=excluded)
            sampled = np.concatenate([sampled, fill])
        elif len(sampled) > self.num_samples:
            sampled = sampled[:self.num_samples]
        sampled = self._block_order(rng, sampled.astype(self.index_dtype, copy=False))
        for idx in sampled:
            yield int(idx)

    def __len__(self):
        return self.num_samples

    def summary(self):
        return {
            'pool': self.n,
            'epoch_samples': self.num_samples,
            'soft_rows': int(self.soft_row_count),
            'soft_pool_fraction': float(self.soft_row_count / max(1, self.n)),
            'soft_epoch_fraction': float(self.soft_epoch_fraction),
            'block_size': int(self.block_size),
        }


# ==============================================================================
# DATALOADER CREATION
# ==============================================================================

def _compute_progress_histogram(binary_file, position_size, indices, game_length_by_id,
                                bins=10, sample_max=200000, seed=0):
    """
    Compute histogram of position progress (move_idx / total_moves) for given indices.
    Uses sampling for speed if indices list is large.
    """
    if len(indices) == 0:
        return [0] * bins, 0, 0
    
    total_count = len(indices)
    sampled = False
    sample_indices = indices
    if sample_max and total_count > sample_max:
        rng = random.Random(seed)
        sample_positions = rng.sample(range(total_count), sample_max)
        sample_indices = [int(indices[pos]) for pos in sample_positions]
        sampled = True
    
    counts = [0] * bins
    with open(binary_file, 'rb') as f:
        for idx in sample_indices:
            offset = idx * position_size
            f.seek(offset + 38)
            header = f.read(6)
            if len(header) < 6:
                continue
            
            game_id = struct.unpack('I', header[:4])[0]
            move_idx = struct.unpack('H', header[4:6])[0]
            total_moves = _lookup_game_length(game_length_by_id, game_id)
            
            if total_moves <= 1:
                progress = 1.0
            else:
                progress = move_idx / (total_moves - 1)
                if progress < 0.0:
                    progress = 0.0
                elif progress > 1.0:
                    progress = 1.0
            
            bin_idx = int(progress * bins)
            if bin_idx >= bins:
                bin_idx = bins - 1
            counts[bin_idx] += 1
    
    sample_count = len(sample_indices)
    return counts, sample_count, total_count if sampled else sample_count


def _print_progress_histogram(label, counts, sample_count, total_count, bins=10):
    if sample_count == 0:
        print(f"  🔍 {label}: no samples")
        return
    sampled_note = ""
    if total_count != sample_count:
        sampled_note = f" (sampled from {total_count:,})"
    print(f"  🔍 {label} progress histogram{sampled_note}:")
    for i in range(bins):
        lo = int(100 * i / bins)
        hi = int(100 * (i + 1) / bins)
        pct = (counts[i] * 100.0) / sample_count if sample_count else 0.0
        print(f"     {lo:02d}-{hi:02d}%: {counts[i]:,} ({pct:.1f}%)")

def _build_game_ranges_fast(binary_file, position_size, start_position, total_positions):
    """Vectorized range scan with bounded memory for very large binaries."""
    start_position = int(start_position)
    total_positions = int(total_positions)
    position_size = int(position_size)
    count = total_positions - start_position
    if count <= 0:
        return []

    record_dtype = np.dtype({
        'names': ['game_id'],
        'formats': ['<u4'],
        'offsets': [38],
        'itemsize': position_size,
    })
    records = np.memmap(
        binary_file,
        dtype=record_dtype,
        mode='r',
        offset=start_position * position_size,
        shape=(count,),
    )
    ranges = []
    previous_game_id = None
    # Avoid a full 329M-row temporary boolean array: a few MiB per chunk is
    # enough and keeps the fast path available under normal RAM pressure.
    chunk_size = 2_000_000
    try:
        for local_start in range(0, count, chunk_size):
            local_end = min(count, local_start + chunk_size)
            game_ids = np.asarray(records['game_id'][local_start:local_end], dtype=np.uint32)
            if not len(game_ids):
                continue
            changes = np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1
            starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
            ends = np.concatenate((changes, np.asarray([len(game_ids)], dtype=np.int64)))
            for begin, end in zip(starts, ends):
                game_id = int(game_ids[int(begin)])
                absolute_start = start_position + local_start + int(begin)
                absolute_end = start_position + local_start + int(end)
                if previous_game_id == game_id and ranges:
                    ranges[-1] = (game_id, ranges[-1][1], absolute_end)
                else:
                    ranges.append((game_id, absolute_start, absolute_end))
                previous_game_id = game_id
    finally:
        del records
    return ranges


def _build_game_ranges(binary_file, position_size, total_positions):
    """
    Build contiguous (game_id, start_idx, end_idx) ranges by scanning the binary file.
    Assumes positions are stored in game order and GameID is at offset 38.
    """
    total_positions = int(total_positions)
    position_size = int(position_size)
    if total_positions <= 0:
        return []

    try:
        return _build_game_ranges_fast(binary_file, position_size, 0, total_positions)
    except Exception as exc:
        print(f"  • Fast game-range scan failed ({exc}); falling back to Python scan.")

    ranges = []
    with open(binary_file, 'rb') as f:
        prev_game_id = None
        start_idx = 0
        last_idx = -1

        iterator = tqdm(range(total_positions), desc="  Game ranges", unit="pos")
        for idx in iterator:
            record = f.read(position_size)
            if len(record) < position_size:
                break

            last_idx = idx
            game_id = struct.unpack('I', record[38:42])[0]

            if prev_game_id is None:
                prev_game_id = game_id
                start_idx = idx
                continue

            if game_id != prev_game_id:
                ranges.append((prev_game_id, start_idx, idx))
                prev_game_id = game_id
                start_idx = idx

        if prev_game_id is not None and last_idx >= start_idx:
            ranges.append((prev_game_id, start_idx, last_idx + 1))

    return ranges


def _build_game_ranges_slice(binary_file, position_size, start_position, total_positions):
    start_position = int(start_position)
    total_positions = int(total_positions)
    if start_position <= 0:
        return _build_game_ranges(binary_file, position_size, total_positions)
    if start_position >= total_positions:
        return []
    try:
        return _build_game_ranges_fast(binary_file, position_size, start_position, total_positions)
    except Exception as exc:
        print(f"  • Fast tail game-range scan failed ({exc}); falling back to Python scan.")

    ranges = []
    with open(binary_file, 'rb') as f:
        f.seek(start_position * int(position_size))
        prev_game_id = None
        start_idx = start_position
        for idx in range(start_position, total_positions):
            record = f.read(position_size)
            if len(record) < position_size:
                break
            game_id = struct.unpack('I', record[38:42])[0]
            if prev_game_id is None:
                prev_game_id = game_id
                start_idx = idx
                continue
            if game_id != prev_game_id:
                ranges.append((prev_game_id, start_idx, idx))
                prev_game_id = game_id
                start_idx = idx
        if prev_game_id is not None:
            ranges.append((prev_game_id, start_idx, total_positions))
    return ranges


def _load_npz_game_ranges(path):
    try:
        cached = np.load(path, mmap_mode='r')
        game_ids = cached['game_id']
        starts = cached['start']
        ends = cached['end']
        if not (len(game_ids) == len(starts) == len(ends)):
            return None
        return [
            (int(game_id), int(start), int(end))
            for game_id, start, end in zip(game_ids, starts, ends)
        ]
    except (OSError, ValueError, KeyError):
        return None


def _try_build_game_ranges_from_append_cache(binary_file, position_size, total_positions, cache_path):
    if cache_path is None:
        return None
    cache_dir = Path(cache_path).parent
    candidates = sorted(
        cache_dir.glob("il_game_ranges_*.npz"),
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    best_ranges = None
    best_total = 0
    for candidate in candidates:
        ranges = _load_npz_game_ranges(candidate)
        if not ranges:
            continue
        old_total = int(ranges[-1][2])
        if old_total <= best_total or old_total >= int(total_positions):
            continue
        if not _validate_cached_ranges_prefix(binary_file, position_size, total_positions, ranges):
            continue
        best_ranges = ranges
        best_total = old_total
    if not best_ranges:
        return None
    print(
        f"  - Game ranges append cache hit: reused {len(best_ranges):,} games "
        f"for first {best_total:,} records; scanning tail only"
    )
    tail_ranges = _build_game_ranges_slice(binary_file, position_size, best_total, total_positions)
    if tail_ranges and int(tail_ranges[0][0]) == int(best_ranges[-1][0]):
        last_game, last_start, _ = best_ranges[-1]
        _, _, first_tail_end = tail_ranges[0]
        best_ranges[-1] = (last_game, last_start, first_tail_end)
        tail_ranges = tail_ranges[1:]
    return best_ranges + tail_ranges


def _load_game_ranges_cache(cache_path, total_positions, legacy_cache_paths=None):
    candidates = _cache_candidates(cache_path, legacy_cache_paths)
    if not candidates:
        return None
    primary = candidates[0]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            cached = np.load(candidate, mmap_mode='r')
            game_ids = cached['game_id']
            starts = cached['start']
            ends = cached['end']
            if not (len(game_ids) == len(starts) == len(ends)):
                continue
            if len(ends) and int(ends[-1]) > int(total_positions):
                continue
            ranges = [
                (int(game_id), int(start), int(end))
                for game_id, start, end in zip(game_ids, starts, ends)
            ]
            if candidate != primary:
                _save_game_ranges_cache(primary, ranges)
                print(f"  - Migrated cached game ranges ({len(ranges):,} games) -> {primary.name}")
            return ranges
        except (OSError, ValueError, KeyError):
            continue
    return None


def _save_game_ranges_cache(cache_path, game_ranges):
    if cache_path is None or not game_ranges:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    game_ids = np.asarray([int(game_id) for game_id, _, _ in game_ranges], dtype=np.uint32)
    starts = np.asarray([int(start) for _, start, _ in game_ranges], dtype=np.uint32)
    ends = np.asarray([int(end) for _, _, end in game_ranges], dtype=np.uint32)
    np.savez(cache_path, game_id=game_ids, start=starts, end=ends)


def _split_indices_by_game(metadata, config, return_game_ranges=False, materialize_indices=True,
                           game_ranges_cache_path=None, legacy_game_ranges_cache_paths=None):
    """
    Split dataset by GameID to prevent val leakage from shared game history.
    Returns train/val indices and game counts for logging.
    """
    import random
    
    binary_file = metadata['binary_file']
    position_size = metadata['position_size']
    total_positions = metadata['total_positions']
    
    print(f"  Game ranges: {total_positions:,} binary records")
    game_ranges = _load_game_ranges_cache(
        game_ranges_cache_path,
        total_positions,
        legacy_cache_paths=legacy_game_ranges_cache_paths,
    )
    if game_ranges is None:
        game_ranges = _try_build_game_ranges_from_append_cache(
            binary_file,
            position_size,
            total_positions,
            game_ranges_cache_path,
        )
    if game_ranges is None:
        game_ranges = _build_game_ranges(binary_file, position_size, total_positions)
        _save_game_ranges_cache(game_ranges_cache_path, game_ranges)
    elif game_ranges_cache_path is not None and not Path(game_ranges_cache_path).exists():
        _save_game_ranges_cache(game_ranges_cache_path, game_ranges)
    if not game_ranges:
        raise ValueError("No games found for per-game split.")
    print(f"  Games found: {len(game_ranges):,}")
    
    rng = random.Random(config['seed'])
    rng.shuffle(game_ranges)
    
    target_train_positions = int(total_positions * config['data']['train_split'])
    
    train_ranges = []
    val_ranges = []
    remaining_train = target_train_positions
    remaining_total = total_positions
    
    # Probabilistic per-game assignment to avoid tail bias and keep length distribution similar.
    # For each game, assign to train with probability proportional to remaining target positions.
    for game_id, start, end in game_ranges:
        count = end - start
        if remaining_total <= 0:
            val_ranges.append((game_id, start, end))
            continue
        
        if remaining_train <= 0:
            val_ranges.append((game_id, start, end))
            remaining_total -= count
            continue
        
        if remaining_train >= remaining_total:
            train_ranges.append((game_id, start, end))
            remaining_train -= count
            remaining_total -= count
            continue
        
        p_train = remaining_train / remaining_total
        if rng.random() < p_train:
            train_ranges.append((game_id, start, end))
            remaining_train -= count
        else:
            val_ranges.append((game_id, start, end))
        remaining_total -= count
    
    # Ensure both splits are non-empty
    if not val_ranges and train_ranges:
        val_ranges.append(train_ranges.pop())
    if not train_ranges and val_ranges:
        train_ranges.append(val_ranges.pop())

    # Keep the random game-level split, but process each split in binary order.
    # This turns later mmap reads for sampling/dedup/soft-targets into mostly
    # sequential disk access instead of millions of random seeks.
    train_ranges.sort(key=lambda item: int(item[1]))
    val_ranges.sort(key=lambda item: int(item[1]))

    if materialize_indices:
        train_indices = _ranges_to_index_array((start, end) for _, start, end in train_ranges)
        val_indices = _ranges_to_index_array((start, end) for _, start, end in val_ranges)
    else:
        train_indices = None
        val_indices = None
    
    if return_game_ranges:
        return (
            train_indices,
            val_indices,
            len(game_ranges),
            len(train_ranges),
            len(val_ranges),
            game_ranges,
            train_ranges,
            val_ranges,
        )
    return train_indices, val_indices, len(game_ranges), len(train_ranges), len(val_ranges)


def create_dataloaders(metadata, config):
    config = normalize_data_config(config)
    """Create dataloaders with POV, MTL, and Dynamic Sliding Window support"""
    
    if metadata['total_positions'] == 0:
        raise ValueError("Cannot create dataloaders with 0 positions.")
    
    total_positions = metadata['total_positions']
    
    split_by_game = True
    game_count = train_game_count = val_game_count = None
    data_cfg = config.get('data', {}) or {}
    positions_per_game_cfg = dict(data_cfg.get('positions_per_game', {}) or {})
    sample_dedup_cfg = dict(data_cfg.get('sample_dedup', {}) or {})
    soft_targets_cfg = dict(data_cfg.get('soft_targets', {}) or {})
    target_selection_cfg = _resolve_count_aware_selection_cfg(data_cfg)
    if soft_targets_cfg.get('enabled', False):
        soft_targets_cfg.setdefault('workers', data_cfg.get('soft_target_workers', 'auto'))
        soft_targets_cfg.setdefault('chunk_size', data_cfg.get('soft_target_chunk_size', 'auto'))
        soft_targets_cfg.pop('materialize_workers', None)
        soft_targets_cfg.pop('materialize_chunk_size', None)
    target_selection_cfg.setdefault('workers', data_cfg.get('soft_target_workers', 'auto'))
    target_selection_cfg.setdefault('chunk_size', data_cfg.get('soft_target_chunk_size', 'auto'))
    train_sampling_cfg = dict(data_cfg.get('train_sampling', {}) or {})
    positions_per_game_enabled = bool(positions_per_game_cfg.get('enabled', False))
    train_soft_targets = None
    val_soft_targets = None
    selection_stages = []
    epoch_train_count = None
    epoch_val_count = None
    game_length_by_id = None
    need_game_length = True
    index_cache_paths = _build_index_cache_paths(metadata, config)
    history_positions = config['model'].get('history_positions', 0)
    hw_cfg = config.get('hardware', {})
    
    if split_by_game:
        if need_game_length:
            print(
                "\nPreparing IL dataset "
                f"(positions={total_positions:,}, "
                f"positions/game={'on' if positions_per_game_enabled else 'off'}, "
                f"dedup={'on' if sample_dedup_cfg.get('enabled', False) else 'off'}, "
                f"soft={'on' if soft_targets_cfg.get('enabled', False) else 'off'})"
            )
            (train_indices, val_indices, game_count, train_game_count,
             val_game_count, game_ranges, train_ranges, val_ranges) = _split_indices_by_game(
                metadata,
                config,
                return_game_ranges=True,
                materialize_indices=not positions_per_game_enabled,
                game_ranges_cache_path=index_cache_paths['game_ranges'],
                legacy_game_ranges_cache_paths=index_cache_paths['legacy_game_ranges'],
            )
            game_length_by_id = _build_game_length_lookup(game_ranges)
            raw_train_count = sum(max(0, int(end) - int(start)) for _, start, end in train_ranges)
            raw_val_count = sum(max(0, int(end) - int(start)) for _, start, end in val_ranges)
            selection_stages = [("Raw split", raw_train_count, raw_val_count)]
            if positions_per_game_enabled:
                all_ppg_indices = _select_positions_per_game_all(
                    metadata['binary_file'],
                    metadata['position_size'],
                    game_ranges,
                    positions_per_game_cfg,
                    config,
                    cache_path=index_cache_paths['all_ppg'],
                    legacy_cache_paths=index_cache_paths['legacy_all_ppg'],
                    train_cache_path=index_cache_paths['train_ppg'],
                    val_cache_path=index_cache_paths['val_ppg'],
                )
                all_ppg_indices = np.asarray(all_ppg_indices, dtype=np.uint32)
                train_mask = _mask_sorted_indices_by_ranges(all_ppg_indices, train_ranges)
                train_indices = np.asarray(all_ppg_indices[train_mask], dtype=np.uint32)
                val_indices = np.asarray(all_ppg_indices[~train_mask], dtype=np.uint32)
                if index_cache_paths.get('train_ppg') is not None and not Path(index_cache_paths['train_ppg']).exists():
                    _save_npy_atomic(index_cache_paths['train_ppg'], train_indices)
                if index_cache_paths.get('val_ppg') is not None and not Path(index_cache_paths['val_ppg']).exists():
                    _save_npy_atomic(index_cache_paths['val_ppg'], val_indices)
            selection_stages.append(("positions_per_game", len(train_indices), len(val_indices)))
            soft_source_mode = str(soft_targets_cfg.get('source', 'positions_per_game') or 'positions_per_game').strip().lower()
            if soft_source_mode in {'raw', 'raw_split', 'split', 'all_split'}:
                train_soft_source_indices = _ranges_to_index_array((start, end) for _, start, end in train_ranges)
                val_soft_source_indices = _ranges_to_index_array((start, end) for _, start, end in val_ranges)
            else:
                train_soft_source_indices = train_indices
                val_soft_source_indices = val_indices
            train_indices = _dedupe_indices_by_signature(
                metadata['binary_file'],
                metadata['position_size'],
                train_indices,
                sample_dedup_cfg,
                "Train",
                history_positions,
                cache_path=index_cache_paths['train_dedup'],
                legacy_cache_paths=[],
            )
            val_indices = _dedupe_indices_by_signature(
                metadata['binary_file'],
                metadata['position_size'],
                val_indices,
                sample_dedup_cfg,
                "Val",
                history_positions,
                cache_path=index_cache_paths['val_dedup'],
                legacy_cache_paths=[],
            )
            selection_stages.append(("sample_dedup", len(train_indices), len(val_indices)))
            target_train_count, target_val_count, target_limited = _split_target_counts(
                len(train_indices),
                len(val_indices),
                data_cfg.get('target_positions', 'max'),
                data_cfg.get('train_split', 0.85),
            )
            epoch_train_count = int(target_train_count)
            epoch_val_count = int(target_val_count)
            if target_limited:
                train_indices, soft_added = _append_soft_candidate_train_pool(
                    metadata['binary_file'],
                    metadata['position_size'],
                    metadata['total_positions'],
                    game_ranges,
                    train_ranges,
                    train_indices,
                    positions_per_game_cfg,
                    sample_dedup_cfg,
                    target_selection_cfg,
                    target_train_count,
                    config,
                    history_positions,
                    index_cache_paths,
                )
                if soft_added > 0:
                    selection_stages.append(("soft_candidates", len(train_indices), len(val_indices)))
                train_pool_count = _resolve_train_candidate_pool_count(
                    target_train_count,
                    len(train_indices),
                    target_selection_cfg,
                )
                train_indices = _count_aware_limit_indices(
                    metadata['binary_file'],
                    metadata['position_size'],
                    metadata['total_positions'],
                    train_soft_source_indices,
                    train_indices,
                    train_pool_count,
                    soft_targets_cfg,
                    target_selection_cfg,
                    "Train",
                    history_positions,
                    game_length_by_id=game_length_by_id,
                )
                val_indices = _evenly_limit_indices(val_indices, target_val_count)
                if len(train_indices) > int(target_train_count):
                    print(
                        f"  - Train epoch rotation: pool={len(train_indices):,}, "
                        f"epoch_samples={int(target_train_count):,}, "
                        f"pool_multiplier={len(train_indices) / max(1, int(target_train_count)):.2f}x"
                    )
            else:
                target_requested = _resolve_target_positions(data_cfg.get('target_positions', 'max'))
                available_after_dedup = len(train_indices) + len(val_indices)
                if target_requested is not None and available_after_dedup < int(target_requested):
                    print(
                        f"  ! IL target_positions shortfall: requested {int(target_requested):,}, "
                        f"available {available_after_dedup:,}. Add more PGNs/data, reduce dedup, "
                        "or lower positions_per_game spacing."
                    )
            selection_stages.append(("candidate_pool", len(train_indices), len(val_indices)))
            if epoch_train_count is not None and (
                int(epoch_train_count) != len(train_indices) or int(epoch_val_count) != len(val_indices)
            ):
                selection_stages.append(("epoch_samples", int(epoch_train_count), int(epoch_val_count)))
            _print_il_selection_stage_report(selection_stages)
            train_soft_paths = _soft_target_paths(index_cache_paths, 'train')
            val_soft_paths = _soft_target_paths(index_cache_paths, 'val')
            train_soft_targets = _build_soft_targets(
                metadata['binary_file'],
                metadata['position_size'],
                train_soft_source_indices,
                train_indices,
                soft_targets_cfg,
                "Train",
                history_positions,
                paths=train_soft_paths,
                game_length_by_id=game_length_by_id,
                legacy_paths=_legacy_soft_target_paths(index_cache_paths, 'train'),
            )
            val_soft_targets = _build_soft_targets(
                metadata['binary_file'],
                metadata['position_size'],
                val_soft_source_indices,
                val_indices,
                soft_targets_cfg,
                "Val",
                history_positions,
                paths=val_soft_paths,
                game_length_by_id=game_length_by_id,
                legacy_paths=_legacy_soft_target_paths(index_cache_paths, 'val'),
            )
            _save_prepare_manifest(
                index_cache_paths['manifest'],
                metadata,
                config,
                index_cache_paths,
                train_indices,
                val_indices,
                game_count,
                train_game_count,
                val_game_count,
                train_soft_targets,
                val_soft_targets,
            )
            try:
                del all_ppg_indices
            except NameError:
                pass
            try:
                del train_mask
            except NameError:
                pass
            try:
                del train_soft_source_indices
            except NameError:
                pass
            try:
                del val_soft_source_indices
            except NameError:
                pass
            try:
                del game_ranges
            except NameError:
                pass
            try:
                del train_ranges
            except NameError:
                pass
            try:
                del val_ranges
            except NameError:
                pass
            gc.collect()
        else:
            train_indices, val_indices, game_count, train_game_count, val_game_count = _split_indices_by_game(metadata, config)
    else:
        all_indices = np.arange(total_positions, dtype=np.uint32)
        
        # 🔧 FIXED: Random shuffle before split to balance Win/Draw/Loss distribution
        # Sequential split causes validation bias (last 10% may have different outcome distribution)
        import random
        random.seed(config['seed'])
        rng = np.random.default_rng(config['seed'])
        rng.shuffle(all_indices)
        
        # Split
        split_idx = int(len(all_indices) * config['data']['train_split'])
        split_idx = max(1, min(split_idx, len(all_indices) - 1))
        
        train_indices = all_indices[:split_idx]
        val_indices = all_indices[split_idx:]
        if need_game_length:
            game_ranges = _build_game_ranges(
                metadata['binary_file'],
                metadata['position_size'],
                total_positions
            )
            game_length_by_id = _build_game_length_lookup(game_ranges)
    
    # Get configuration
    position_size = metadata.get('position_size')
    
    # 🔧 v4.5 FIXED: Calculate actual input planes (16 per position with metadata)
    input_planes = 16 * (1 + history_positions)  # 16 planes (12 pieces + 4 metadata)
    
    selected_total = int(len(train_indices) + len(val_indices))
    epoch_train_print = int(epoch_train_count) if epoch_train_count is not None else int(len(train_indices))
    epoch_val_print = int(epoch_val_count) if epoch_val_count is not None else int(len(val_indices))
    print("\nIL Dataset Ready")
    print("-" * 70)
    print(
        f"  Pool     : {selected_total:,} "
        f"(train={len(train_indices):,}, val={len(val_indices):,}) "
        f"from {total_positions:,} binary positions"
    )
    if epoch_train_print != len(train_indices) or epoch_val_print != len(val_indices):
        print(
            f"  Epoch    : {epoch_train_print + epoch_val_print:,} "
            f"(train={epoch_train_print:,}, val={epoch_val_print:,}; val is fixed)"
        )
    print(f"  Input    : {position_size}B records, {input_planes} planes, history={history_positions}")
    if positions_per_game_cfg.get('enabled', False):
        phase_budgets = _positions_per_game_budgets(positions_per_game_cfg)
        print(
            f"  Select   : {positions_per_game_cfg.get('selection_mode', 'even')}, "
            f"max={positions_per_game_cfg.get('max_total', 32)}, "
            f"min_distance={positions_per_game_cfg.get('min_distance', 1)}, "
            f"O/M/E/R="
            f"{phase_budgets.get('opening', 0)}/"
            f"{phase_budgets.get('middlegame', 0)}/"
            f"{phase_budgets.get('endgame', 0)}/"
            f"{phase_budgets.get('rare_or_eventful', 0)}"
        )
    else:
        print("  Select   : positions/game off")
    if sample_dedup_cfg.get('enabled', False):
        print(
            f"  Dedup    : {sample_dedup_cfg.get('mode', 'position_plus_move')}, "
            f"max_count={sample_dedup_cfg.get('max_count', 4)}, "
            f"turn/history={sample_dedup_cfg.get('include_turn', True)}/"
            f"{sample_dedup_cfg.get('include_history', True)}"
        )
    else:
        print("  Dedup    : off")
    if target_selection_cfg.get('enabled', False):
        soft_candidate_cfg = _resolve_soft_candidate_selection_cfg(target_selection_cfg, positions_per_game_cfg)
        soft_candidate_text = ""
        if soft_candidate_cfg is not None:
            soft_candidate_text = (
                f", soft_extra=min_dist{soft_candidate_cfg.get('min_distance')}"
                f"/max{soft_candidate_cfg.get('max_total')}"
            )
        print(
            f"  Target   : count-aware train, "
            f"broad={100.0 * float(target_selection_cfg.get('broad_fraction', 0.75)):.0f}%, "
            f"occurrence>=3, multi-move preference, "
            f"single-move cap={100.0 * _COUNT_AWARE_MAX_SINGLE_FRACTION:.0f}%"
            f"{soft_candidate_text}"
        )
    else:
        print("  Target   : even")
    if soft_targets_cfg.get('enabled', False):
        print(
            f"  Soft     : {soft_targets_cfg.get('mode', 'fen')}, "
            f"source={soft_targets_cfg.get('source', 'positions_per_game')}, "
            f"top_moves={soft_targets_cfg.get('max_policy_moves', 32)}"
        )
        rating_cfg = soft_targets_cfg.get('policy_rating_weight', {}) or {}
        print(
            f"  Policy   : min_key_count={_POLICY_TARGET_MIN_KEY_COUNT}, "
            f"min_move_count={_POLICY_TARGET_MIN_MOVE_COUNT}, "
            f"power={_POLICY_TARGET_POWER}, "
            f"elo_weight={'on' if rating_cfg.get('enabled', False) else 'off'}"
        )
    else:
        print("  Soft     : off")
    if train_sampling_cfg.get('enabled', False):
        print(
            f"  Sampler  : smart_epoch, "
            f"soft_once_per_epoch=on, "
            f"soft_cap={100.0 * float(train_sampling_cfg.get('soft_max_fraction', 0.28)):.0f}%"
        )
    else:
        print("  Sampler  : default shuffle")
    if split_by_game and game_count is not None:
        print(f"  Games    : {game_count:,} (train={train_game_count:,}, val={val_game_count:,})")
    print("-" * 70)
    
    # Create datasets
    print("Creating dataset views...")
    use_index_mmap = bool(config.get('hardware', {}).get('dataloader_index_mmap', True))
    dataloader_board_dtype = str(hw_cfg.get('dataloader_board_dtype', 'float32') or 'float32')
    dataloader_return_numpy = bool(hw_cfg.get('dataloader_return_numpy', True))
    train_soft_target_paths = _soft_target_paths(index_cache_paths, 'train') if train_soft_targets is not None else None
    val_soft_target_paths = _soft_target_paths(index_cache_paths, 'val') if val_soft_targets is not None else None
    selection_stage_payload = _selection_stages_payload(selection_stages)
    train_filter_stats = {
        'selection_stages': selection_stage_payload,
        'binary_positions': int(total_positions),
    }
    val_filter_stats = {
        'selection_stages': selection_stage_payload,
        'binary_positions': int(total_positions),
    }
    train_dataset = BinaryChessDataset(
        metadata['binary_file'], 
        train_indices, 
        position_size=position_size,
        history_positions=history_positions,
        game_length_by_id=game_length_by_id,
        index_cache_path=index_cache_paths['train'],
        use_index_mmap=use_index_mmap,
        soft_targets=train_soft_targets,
        soft_target_paths=train_soft_target_paths,
        board_dtype=dataloader_board_dtype,
        return_numpy=dataloader_return_numpy,
        filter_stats=train_filter_stats,
    )
    if epoch_train_count is not None:
        train_dataset.epoch_sample_count = max(1, min(len(train_dataset), int(epoch_train_count)))
        train_dataset.filter_stats['epoch_sample_count'] = int(train_dataset.epoch_sample_count)
    val_dataset = BinaryChessDataset(
        metadata['binary_file'], 
        val_indices,
        position_size=position_size,
        history_positions=history_positions,
        game_length_by_id=game_length_by_id,
        index_cache_path=index_cache_paths['val'],
        use_index_mmap=use_index_mmap,
        soft_targets=val_soft_targets,
        soft_target_paths=val_soft_target_paths,
        board_dtype=dataloader_board_dtype,
        return_numpy=dataloader_return_numpy,
        filter_stats=val_filter_stats,
    )

    # Debug: show progress histogram after per-game selection
    debug_enabled = config.get('debug', {}).get('enabled', False)
    if debug_enabled and positions_per_game_cfg.get('enabled', False):
        if game_length_by_id is None:
            print("  ⚠️ Debug histogram skipped: game_length_by_id missing")
        else:
            counts, sample_count, total_count = _compute_progress_histogram(
                metadata['binary_file'],
                position_size,
                train_dataset.indices,
                game_length_by_id,
                bins=10,
                sample_max=200000,
                seed=config.get('seed', 0)
            )
            _print_progress_histogram("Train", counts, sample_count, total_count, bins=10)
            counts, sample_count, total_count = _compute_progress_histogram(
                metadata['binary_file'],
                position_size,
                val_dataset.indices,
                game_length_by_id,
                bins=10,
                sample_max=200000,
                seed=(config.get('seed', 0) + 1)
            )
            _print_progress_histogram("Val", counts, sample_count, total_count, bins=10)
    
    # Dataloaders
    print("Creating DataLoaders...")
    il_cfg = config.get('imitation_learning', {})
    train_batch_size = int(il_cfg['batch_size'])
    eval_batch_size = int(il_cfg.get('eval_batch_size', 0) or 0)
    if eval_batch_size <= 0:
        eval_multiplier = float(il_cfg.get('eval_batch_size_multiplier', 1.0) or 1.0)
        eval_batch_size = max(1, int(round(train_batch_size * max(1.0, eval_multiplier))))
    train_num_workers = int(hw_cfg.get('num_workers', 0) or 0)
    val_num_workers = int(hw_cfg.get('val_num_workers', train_num_workers) or 0)
    train_prefetch_factor = int(hw_cfg.get('train_prefetch_factor', hw_cfg.get('prefetch_factor', 1)) or 1)
    val_prefetch_factor = int(hw_cfg.get('val_prefetch_factor', hw_cfg.get('prefetch_factor', 1)) or 1)
    train_pin_memory = bool(hw_cfg.get('pin_memory', True))
    val_pin_memory = bool(hw_cfg.get('val_pin_memory', train_pin_memory))
    train_persistent_workers = bool(hw_cfg.get('persistent_workers', True)) and train_num_workers > 0
    val_persistent_workers = bool(hw_cfg.get('val_persistent_workers', False)) and val_num_workers > 0
    use_numpy_shuffle = bool(hw_cfg.get('dataloader_numpy_shuffle', True))
    use_block_shuffle = bool(hw_cfg.get('dataloader_block_shuffle', False))
    block_shuffle_size = max(1, int(hw_cfg.get('dataloader_block_shuffle_size', 65536) or 65536))
    train_in_order = bool(hw_cfg.get('dataloader_train_in_order', True))
    val_in_order = bool(hw_cfg.get('dataloader_val_in_order', True))
    train_sampler = None
    train_sampling_label = "shuffle"
    train_sampling_mode = str(train_sampling_cfg.get('mode', 'smart_epoch') or 'smart_epoch').strip().lower()
    print("Creating train sampler...")
    if bool(train_sampling_cfg.get('enabled', False)) and train_sampling_mode == 'smart_epoch':
        try:
            epoch_samples = int(getattr(train_dataset, 'epoch_sample_count', len(train_dataset)) or len(train_dataset))
            train_sampler = SmartEpochSampler(
                train_dataset,
                train_sampling_cfg,
                seed=config.get('seed', 0),
                num_samples=epoch_samples,
            )
            summary = train_sampler.summary()
            train_sampling_label = "smart_epoch"
            print(
                f"Train sampler: smart_epoch, "
                f"pool={summary['pool']:,}, epoch_samples={summary['epoch_samples']:,}, "
                f"soft_rows={summary['soft_rows']:,}, soft_pool={100.0 * summary['soft_pool_fraction']:.1f}%"
            )
            print(
                f"  epoch mix broad/soft="
                f"{1.0 - float(summary.get('soft_epoch_fraction', 0.0)):.3f}/"
                f"{float(summary.get('soft_epoch_fraction', 0.0)):.3f}, "
                f"soft_once_per_epoch=on, "
                f"blocks={summary['block_size']:,}"
            )
        except Exception as exc:
            train_sampler = None
            train_sampling_label = "shuffle"
            print(
                f"  ! Train sampler: smart_epoch failed ({type(exc).__name__}: {exc}); "
                "falling back to stable shuffle."
            )
    if train_sampler is None:
        if use_block_shuffle:
            print(f"  - block shuffle sampler: block_size={block_shuffle_size:,}, order=identity")
            train_sampler = NumpyBlockShuffleSampler(
                train_dataset,
                seed=config.get('seed', 0),
                block_size=block_shuffle_size,
            )
            order_mode = getattr(train_sampler, 'order_mode', 'unknown')
            train_sampling_label = f"block_shuffle:{block_shuffle_size:,}/{order_mode}"
        else:
            train_sampler = (
                NumpyRandomSampler(train_dataset, seed=config.get('seed', 0))
                if use_numpy_shuffle
                else None
            )
            train_sampling_label = "numpy_shuffle" if train_sampler is not None else "torch_shuffle"
        print(f"Train sampler: {train_sampling_label}")
    
    print("Creating train DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_num_workers,
        pin_memory=train_pin_memory,
        persistent_workers=train_persistent_workers,
        prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
        in_order=train_in_order if train_num_workers > 0 else True,
        collate_fn=il_collate_fn,
        worker_init_fn=il_worker_init_fn if train_num_workers > 0 else None,
    )
    
    print("Creating val DataLoader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=val_pin_memory,
        persistent_workers=val_persistent_workers,
        prefetch_factor=val_prefetch_factor if val_num_workers > 0 else None,
        in_order=val_in_order if val_num_workers > 0 else True,
        collate_fn=il_collate_fn,
        worker_init_fn=il_worker_init_fn if val_num_workers > 0 else None,
    )
    
    print(
        f"DataLoaders: batch={train_batch_size}/{eval_batch_size}, "
        f"workers={train_num_workers}/{val_num_workers}, "
        f"prefetch={train_prefetch_factor}/{val_prefetch_factor}, "
        f"dtype={dataloader_board_dtype}, numpy={'on' if dataloader_return_numpy else 'off'}, "
        f"collate=fast, mmap={'on' if use_index_mmap else 'off'}"
    )
    
    return train_loader, val_loader
