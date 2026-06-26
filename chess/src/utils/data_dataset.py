"""
Chess dataset and dataloader utilities.
Split from src.data to keep dataset logic separate from preprocessing.
"""

import chess
from collections import defaultdict
import gc
import hashlib
import json
import mmap
import os
import random
import struct
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.utils.data_helpers import (
    ACTION_SIZE,
    AZ_ACTION_SIZE,
    az_index_to_policy_index,
    compact_to_board,
    compact_to_tensor,
    get_turn_from_move_idx,
    index_to_move,
)


_SOFT_TARGET_AGGREGATE_CONTEXT = {}
_SOFT_TARGET_MATERIALIZE_CONTEXT = {}


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
    if move == chess.Move.null() or move not in board.legal_moves:
        return False

    if move.promotion is not None:
        return True
    if board.is_en_passant(move) or board.is_castling(move):
        return True
    if board.is_check() or board.gives_check(move):
        return True
    if board.legal_moves.count() <= 2:
        return True
    if len(board.piece_map()) <= 7:
        return True
    if int(board.halfmove_clock) >= 80:
        return True

    after = board.copy(stack=False)
    after.push(move)
    return after.is_game_over(claim_draw=True)


def _compact_piece_stats(compact_board):
    piece_count = 0
    material = 0
    pawns = 0
    values = {
        1: 1, 2: 3, 3: 3, 4: 5, 5: 9,
        7: 1, 8: 3, 9: 3, 10: 5, 11: 9,
    }
    for byte in compact_board[:32]:
        for code in ((int(byte) >> 4) & 0x0F, int(byte) & 0x0F):
            if code == 0:
                continue
            piece_count += 1
            material += values.get(code, 0)
            if code in (1, 7):
                pawns += 1
    return piece_count, material, pawns


def _quick_record_features(record):
    compact_board = record[:38]
    piece_count, material, pawns = _compact_piece_stats(compact_board)
    return {
        'piece_count': piece_count,
        'material': material,
        'pawns': pawns,
        'castling': int(compact_board[32]),
        'ep_square': int(compact_board[33]),
        'halfmove_clock': int(struct.unpack('>H', compact_board[34:36])[0]),
        'fullmove_number': int(struct.unpack('>H', compact_board[36:38])[0]),
        'outcome': float(struct.unpack('f', record[46:50])[0]),
    }


def _record_at(mm, position_size, index):
    if mm is None:
        return None
    offset = int(index) * int(position_size)
    record = mm[offset:offset + int(position_size)]
    if len(record) != int(position_size):
        return None
    return record


def _smart_position_score(mm, position_size, abs_idx, start, end):
    record = _record_at(mm, position_size, abs_idx)
    if record is None:
        return 0.0

    features = _quick_record_features(record)
    length = max(1, int(end) - int(start))
    rel = int(abs_idx) - int(start)
    progress = float(rel) / float(max(1, length - 1))
    fullmove = int(features['fullmove_number'])
    piece_count = int(features['piece_count'])
    material = int(features['material'])
    pawns = int(features['pawns'])

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
    if int(features['ep_square']) != 255:
        score += 0.30
    if int(features['halfmove_clock']) == 0 and fullmove > 8:
        score += 0.20

    prev_record = _record_at(mm, position_size, int(abs_idx) - 1) if int(abs_idx) > int(start) else None
    next_record = _record_at(mm, position_size, int(abs_idx) + 1) if int(abs_idx) + 1 < int(end) else None
    for neighbor in (prev_record, next_record):
        if neighbor is None:
            continue
        n_piece_count, n_material, _ = _compact_piece_stats(neighbor[:38])
        if n_piece_count != piece_count:
            score += 0.50
        if n_material != material:
            score += 0.35
        if int(neighbor[32]) != int(features['castling']):
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

    scored = {
        rel: _smart_position_score(mm, position_size, int(start) + rel, start, end)
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
            workers = max(1, int(os.cpu_count() or 1) - 1)
        elif lowered in {'off', 'false', 'no'}:
            workers = 1
        else:
            try:
                workers = int(lowered)
            except ValueError:
                workers = max(1, int(os.cpu_count() or 1) - 1)
    else:
        try:
            workers = int(raw)
        except (TypeError, ValueError):
            workers = max(1, int(os.cpu_count() or 1) - 1)
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


def _select_positions_per_game_worker(args):
    chunk_id, binary_file, position_size, ranges, cfg, needs_records = args
    selected = []
    if needs_records:
        with open(binary_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
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
            f"  - {label} positions_per_game: {before:,} -> {len(indices):,} "
            f"(mode={selection_mode}, max_total={int(cfg.get('max_total', 32))}, "
            f"workers={_resolve_positions_per_game_workers(cfg, len(game_ranges))}, "
            f"min_distance={int(cfg.get('min_distance', 1))}, budgets={_positions_per_game_budgets(cfg)})"
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
            f"  - {label} positions_per_game: {before:,} -> {len(indices):,} "
            f"(mode={selection_mode}, max_total={int(cfg.get('max_total', 32))}, "
            f"min_distance={int(cfg.get('min_distance', 1))}, budgets={_positions_per_game_budgets(cfg)})"
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
        f"  • {label} positions_per_game: {before:,} -> {len(indices):,} "
        f"(mode={selection_mode}, max_total={int(cfg.get('max_total', 32))}, "
        f"min_distance={int(cfg.get('min_distance', 1))}, budgets={_positions_per_game_budgets(cfg)})"
    )
    return indices


def _filter_sorted_indices_by_ranges(indices, ranges):
    indices = np.asarray(indices, dtype=np.uint32)
    if len(indices) == 0 or not ranges:
        return np.asarray([], dtype=np.uint32)
    if len(indices) > 1 and np.any(indices[1:] < indices[:-1]):
        indices = np.sort(indices)
    mask = _mask_sorted_indices_by_ranges(indices, ranges)
    return np.asarray(indices[mask], dtype=np.uint32)


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

    board = record[:38]
    game_id = struct.unpack('I', record[38:42])[0]
    move_idx = struct.unpack('H', record[42:44])[0]

    if mode == "fen":
        signature_board_bytes = 38
        sig_bytes = bytes(board[:signature_board_bytes])
    elif mode == "fen_no_counters":
        signature_board_bytes = 34
        sig_bytes = bytes(board[:signature_board_bytes])
    elif mode == "pieces":
        signature_board_bytes = 32
        sig_bytes = bytes(board[:signature_board_bytes])
    else:
        signature_board_bytes = 34
        move_target = struct.unpack('H', record[44:46])[0]
        if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
            move_target = az_index_to_policy_index(move_target)
        sig_bytes = bytes(board[:34]) + struct.pack('H', int(move_target))

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

    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            iterator = tqdm(indices_array, desc=f"  {label} sample_dedup", unit="pos")
            for idx in iterator:
                sig = _sample_signature_for_index(mm, position_size, int(idx), cfg, history_positions)
                if sig is not None:
                    sig_to_indices[sig].append(int(idx))
        finally:
            mm.close()

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
        f"  • {label} sample_dedup: {before:,} -> {len(final_indices):,} "
        f"(removed={removed:,}, mode={mode}, max_count={max_count})"
    )
    return final_indices


def _position_signature_for_index(mm, position_size, index, cfg, history_positions):
    mode = cfg.get('mode', 'fen')
    valid_modes = {"fen", "fen_no_counters", "pieces"}
    if mode not in valid_modes:
        raise ValueError(f"soft_targets.mode must be one of {sorted(valid_modes)}, got: {mode}")

    offset = int(index) * int(position_size)
    record = mm[offset:offset + position_size]
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
    if float(cfg.get('value_wdl_shrinkage_alpha', 0.0) or 0.0) > 0.0:
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


def _sample_weight_from_count(count, cfg, move_idx=None, total_moves=None):
    weight_cfg = cfg.get('sample_weight', {}) or {}
    if not weight_cfg.get('enabled', True):
        return 1.0
    mode = str(weight_cfg.get('mode', 'sqrt_count')).lower()
    count = max(1.0, float(count))
    if mode in {'none', 'off', 'disabled'}:
        weight = 1.0
    elif mode in {'sqrt', 'sqrt_count'}:
        weight = float(np.sqrt(count))
    elif mode in {'log', 'log_count', 'log1p'}:
        weight = float(np.log1p(count))
    elif mode in {'occurrence_count', 'count', 'raw'}:
        weight = count
    else:
        raise ValueError(
            "soft_targets.sample_weight.mode must be one of: "
            "occurrence_count, sqrt_count, log_count, none"
        )
    max_weight = float(weight_cfg.get('max', 32.0) or 0.0)
    if mode in {'occurrence_count', 'count', 'raw'} and max_weight <= 0.0:
        raise ValueError(
            "Uncapped soft_targets.sample_weight occurrence_count is unsafe. "
            "Use sqrt_count/log_count or set sample_weight.max > 0."
        )
    if max_weight > 0.0:
        weight = min(weight, max_weight)

    try:
        min_boost_count = max(1.0, float(weight_cfg.get('opening_boost_min_count', 2.0)))
    except (TypeError, ValueError):
        min_boost_count = 2.0
    if count >= min_boost_count:
        progress = None
        try:
            total_moves_f = float(total_moves)
            move_idx_f = float(move_idx)
            if total_moves_f > 1.0:
                progress = max(0.0, min(1.0, move_idx_f / total_moves_f))
        except (TypeError, ValueError):
            progress = None

        if progress is not None:
            try:
                opening_progress_max = float(weight_cfg.get('opening_progress_max', 0.25))
            except (TypeError, ValueError):
                opening_progress_max = 0.25
            try:
                opening_multiplier = float(weight_cfg.get('opening_multiplier', 1.0))
            except (TypeError, ValueError):
                opening_multiplier = 1.0
            try:
                opening_max = float(weight_cfg.get('opening_max', max_weight or 0.0) or 0.0)
            except (TypeError, ValueError):
                opening_max = max_weight or 0.0

            if opening_multiplier > 1.0 and progress <= max(0.0, min(1.0, opening_progress_max)):
                weight *= opening_multiplier
                if opening_max > 0.0:
                    weight = min(weight, opening_max)
    return max(1.0e-6, float(weight))


def _soft_target_value_key_config(cfg):
    cfg = dict(cfg or {})
    value_key_cfg = dict(cfg.get('value_key', {}) or {})
    key_cfg = dict(cfg)
    key_cfg.pop('value_key', None)
    key_cfg.update(value_key_cfg)
    return key_cfg


def _value_sample_weight_from_context(value_count, move_idx, total_moves, cfg):
    weight_cfg = cfg.get('value_sample_weight', {}) or {}
    if not weight_cfg.get('enabled', True):
        return 1.0

    count = max(1.0, float(value_count))
    try:
        confidence_min = float(weight_cfg.get('confidence_min', 0.60))
    except (TypeError, ValueError):
        confidence_min = 0.60
    confidence_min = max(0.0, min(1.0, confidence_min))
    try:
        confidence_k = max(0.0, float(weight_cfg.get('confidence_k', 3.0)))
    except (TypeError, ValueError):
        confidence_k = 3.0
    confidence = confidence_min + (1.0 - confidence_min) * (count / max(count + confidence_k, 1.0e-8))

    try:
        progress_min = float(weight_cfg.get('progress_min', 0.25))
    except (TypeError, ValueError):
        progress_min = 0.25
    progress_min = max(0.0, min(1.0, progress_min))
    try:
        progress_power = max(0.05, float(weight_cfg.get('progress_power', 1.5)))
    except (TypeError, ValueError):
        progress_power = 1.5
    progress = 1.0
    try:
        total_moves = float(total_moves)
        move_idx = float(move_idx)
        if total_moves > 1.0:
            progress = max(0.0, min(1.0, move_idx / total_moves))
    except (TypeError, ValueError):
        progress = 1.0
    progress_weight = progress_min + (1.0 - progress_min) * (progress ** progress_power)

    weight = confidence * progress_weight
    max_weight = float(weight_cfg.get('max', 1.0) or 0.0)
    if max_weight > 0.0:
        weight = min(weight, max_weight)
    return max(1.0e-6, float(weight))


def _shrink_wdl_counts(wdl_counts, count, cfg, prior):
    wdl = np.asarray(wdl_counts, dtype=np.float64)
    total = float(count if count is not None else np.sum(wdl))
    if total <= 0.0 or float(np.sum(wdl)) <= 0.0:
        return np.asarray(prior, dtype=np.float32)
    alpha = max(0.0, float(cfg.get('value_wdl_shrinkage_alpha', 0.0) or 0.0))
    if alpha <= 0.0:
        return (wdl / float(np.sum(wdl))).astype(np.float32)
    prior = np.asarray(prior, dtype=np.float64)
    prior_sum = float(np.sum(prior))
    if prior_sum <= 0.0:
        prior = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    else:
        prior = prior / prior_sum
    smoothed = (wdl + alpha * prior) / (float(np.sum(wdl)) + alpha)
    return smoothed.astype(np.float32)


def _sample_weight_config_max(cfg):
    weight_cfg = (cfg.get('sample_weight', {}) or {}) if cfg else {}
    if not weight_cfg.get('enabled', True):
        return 1.0
    base_max = float(weight_cfg.get('max', 32.0) or 0.0)
    opening_max = float(weight_cfg.get('opening_max', base_max) or 0.0)
    return max(base_max, opening_max)


def _validate_soft_target_arrays(arrays, cfg, label):
    if arrays is None:
        return None
    max_weight = _sample_weight_config_max(cfg)
    if max_weight > 0.0:
        try:
            observed_max = float(np.max(arrays['sample_weight'])) if len(arrays['sample_weight']) else 1.0
        except (KeyError, ValueError):
            return None
        if observed_max > max_weight + 1.0e-4:
            print(
                f"  • {label} soft_targets cache ignored: sample_weight max "
                f"{observed_max:.3f} exceeds configured cap {max_weight:.3f}"
            )
            return None
    return arrays


def _resolve_soft_target_workers(cfg, chunk_count):
    return _resolve_positions_per_game_workers(cfg, chunk_count)


def _soft_target_chunk_size(cfg, source_count, workers):
    raw = cfg.get('chunk_size', 500000)
    if isinstance(raw, str) and raw.strip().lower() in {'auto', ''}:
        target_chunks = max(1, int(workers) * 8)
        return max(50000, min(1000000, int(np.ceil(float(source_count) / float(target_chunks)))))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 500000


def _soft_target_index_chunks(indices, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for chunk_id, start in enumerate(range(0, len(indices), chunk_size)):
        yield chunk_id, np.asarray(indices[start:start + chunk_size], dtype=np.uint32)


def _soft_targets_cache_config(cfg):
    cfg = dict(cfg or {})
    cfg.pop('workers', None)
    cfg.pop('chunk_size', None)
    cfg.pop('materialize_workers', None)
    cfg.pop('materialize_chunk_size', None)
    return cfg


def _init_soft_target_aggregate_worker(binary_file, position_size, cfg, history_positions,
                                       game_length_by_id, policy_enabled, value_enabled):
    global _SOFT_TARGET_AGGREGATE_CONTEXT
    _SOFT_TARGET_AGGREGATE_CONTEXT = {
        'binary_file': binary_file,
        'position_size': int(position_size),
        'cfg': dict(cfg or {}),
        'value_key_cfg': _soft_target_value_key_config(cfg),
        'history_positions': int(history_positions or 0),
        'game_length_by_id': game_length_by_id,
        'policy_enabled': bool(policy_enabled),
        'value_enabled': bool(value_enabled),
    }


def _aggregate_soft_target_indices(binary_file, position_size, indices, cfg, history_positions,
                                   game_length_by_id, policy_enabled, value_enabled):
    value_key_cfg = _soft_target_value_key_config(cfg)
    policy_counts = defaultdict(lambda: defaultdict(float))
    wdl_counts = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    policy_total_counts = defaultdict(float)
    value_total_counts = defaultdict(float)
    moves_left_logs = defaultdict(list)

    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for idx in np.asarray(indices, dtype=np.uint32):
                idx = int(idx)
                policy_key = _position_signature_for_index(mm, position_size, idx, cfg, history_positions)
                value_key = _position_signature_for_index(mm, position_size, idx, value_key_cfg, history_positions)
                if policy_key is None and value_key is None:
                    continue
                offset = idx * int(position_size)
                record = mm[offset:offset + int(position_size)]
                game_id = struct.unpack('I', record[38:42])[0]
                move_idx = struct.unpack('H', record[42:44])[0]
                move_target = struct.unpack('H', record[44:46])[0]
                if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
                    move_target = az_index_to_policy_index(move_target)
                outcome = float(struct.unpack('f', record[46:50])[0])
                total_moves = _lookup_game_length(game_length_by_id, game_id)
                if policy_key is not None and total_moves > 0:
                    remaining_plies = max(0.0, float(total_moves) - float(move_idx))
                    moves_left_logs[policy_key].append(float(np.log1p(remaining_plies)))

                if policy_key is not None:
                    if policy_enabled:
                        policy_counts[policy_key][int(move_target)] += 1.0
                    policy_total_counts[policy_key] += 1.0
                if value_key is not None and value_enabled:
                    if outcome > 0.9:
                        wdl_counts[value_key][0] += 1.0
                    elif outcome < -0.9:
                        wdl_counts[value_key][2] += 1.0
                    else:
                        wdl_counts[value_key][1] += 1.0
                    value_total_counts[value_key] += 1.0
        finally:
            mm.close()

    return (
        {key: dict(value) for key, value in policy_counts.items()},
        {key: value.tolist() for key, value in wdl_counts.items()},
        dict(policy_total_counts),
        dict(value_total_counts),
        dict(moves_left_logs),
    )


def _signature_key_array_for_indices(binary_file, position_size, indices, cfg, history_positions):
    indices = np.asarray(indices, dtype=np.uint32)
    keys = np.zeros((len(indices), 16), dtype=np.uint8)
    if len(indices) == 0:
        return keys
    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for row, idx in enumerate(indices):
                key = _position_signature_for_index(mm, position_size, int(idx), cfg, history_positions)
                if key is not None:
                    keys[row] = np.frombuffer(key, dtype=np.uint8)
        finally:
            mm.close()
    return keys


def _soft_target_aggregate_worker(args):
    chunk_id, indices, final_row_start, final_indices = args
    ctx = _SOFT_TARGET_AGGREGATE_CONTEXT
    final_keys = _signature_key_array_for_indices(
        ctx['binary_file'],
        ctx['position_size'],
        final_indices,
        ctx['cfg'],
        ctx['history_positions'],
    )
    final_value_keys = _signature_key_array_for_indices(
        ctx['binary_file'],
        ctx['position_size'],
        final_indices,
        ctx['value_key_cfg'],
        ctx['history_positions'],
    )
    return (
        chunk_id,
        len(indices),
        int(final_row_start),
        final_keys,
        final_value_keys,
        _aggregate_soft_target_indices(
            ctx['binary_file'],
            ctx['position_size'],
            indices,
            ctx['cfg'],
            ctx['history_positions'],
            ctx['game_length_by_id'],
            ctx['policy_enabled'],
            ctx['value_enabled'],
        ),
    )


def _merge_soft_target_aggregate(result, policy_counts, wdl_counts, policy_total_counts,
                                 value_total_counts, moves_left_logs):
    result_policy, result_wdl, result_policy_total, result_value_total, result_moves_left = result
    for key, moves in result_policy.items():
        target_moves = policy_counts[key]
        for move_idx, count in moves.items():
            target_moves[int(move_idx)] += float(count)
    for key, wdl in result_wdl.items():
        wdl_counts[key] += np.asarray(wdl, dtype=np.float64)
    for key, count in result_policy_total.items():
        policy_total_counts[key] += float(count)
    for key, count in result_value_total.items():
        value_total_counts[key] += float(count)
    for key, values in result_moves_left.items():
        moves_left_logs[key].extend(values)


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

    policy_counts = defaultdict(lambda: defaultdict(float))
    wdl_counts = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    policy_total_counts = defaultdict(float)
    value_total_counts = defaultdict(float)
    moves_left_logs = defaultdict(list)
    final_key_bytes = np.zeros((len(final_indices_array), 16), dtype=np.uint8)
    final_value_key_bytes = np.zeros((len(final_indices_array), 16), dtype=np.uint8)
    value_key_cfg = _soft_target_value_key_config(cfg)

    def _final_slice_for_source_chunk(indices):
        if len(final_indices_array) == 0 or len(indices) == 0:
            return 0, final_indices_array[:0]
        start_idx = int(np.min(indices))
        end_idx = int(np.max(indices))
        start_row = int(np.searchsorted(final_indices_array, start_idx, side='left'))
        end_row = int(np.searchsorted(final_indices_array, end_idx, side='right'))
        return start_row, final_indices_array[start_row:end_row]

    if workers <= 1 or len(chunks) <= 1:
        iterator = tqdm(chunks, desc=f"  {label} soft_targets aggregate", unit="chunk")
        for _, indices in iterator:
            final_row_start, final_slice = _final_slice_for_source_chunk(indices)
            if len(final_slice) > 0:
                final_key_bytes[final_row_start:final_row_start + len(final_slice)] = _signature_key_array_for_indices(
                    binary_file,
                    position_size,
                    final_slice,
                    cfg,
                    history_positions,
                )
                final_value_key_bytes[final_row_start:final_row_start + len(final_slice)] = _signature_key_array_for_indices(
                    binary_file,
                    position_size,
                    final_slice,
                    value_key_cfg,
                    history_positions,
                )
            result = _aggregate_soft_target_indices(
                binary_file,
                position_size,
                indices,
                cfg,
                history_positions,
                game_length_by_id,
                policy_enabled,
                value_enabled,
            )
            _merge_soft_target_aggregate(
                result,
                policy_counts,
                wdl_counts,
                policy_total_counts,
                value_total_counts,
                moves_left_logs,
            )
        return policy_counts, wdl_counts, policy_total_counts, value_total_counts, moves_left_logs, final_key_bytes, final_value_key_bytes

    task_args = []
    for chunk_id, indices in chunks:
        final_row_start, final_slice = _final_slice_for_source_chunk(indices)
        task_args.append((chunk_id, indices, final_row_start, final_slice))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_soft_target_aggregate_worker,
        initargs=(
            binary_file,
            int(position_size),
            dict(cfg),
            int(history_positions or 0),
            game_length_by_id,
            bool(policy_enabled),
            bool(value_enabled),
        ),
    ) as executor:
        futures = [executor.submit(_soft_target_aggregate_worker, args) for args in task_args]
        with tqdm(total=len(source_indices_array), desc=f"  {label} soft_targets aggregate", unit="pos") as pbar:
            for future in as_completed(futures):
                _, processed, final_row_start, final_keys, final_value_keys, result = future.result()
                if len(final_keys) > 0:
                    final_key_bytes[final_row_start:final_row_start + len(final_keys)] = final_keys
                if len(final_value_keys) > 0:
                    final_value_key_bytes[final_row_start:final_row_start + len(final_value_keys)] = final_value_keys
                _merge_soft_target_aggregate(
                    result,
                    policy_counts,
                    wdl_counts,
                    policy_total_counts,
                    value_total_counts,
                    moves_left_logs,
                )
                pbar.update(int(processed))

    return policy_counts, wdl_counts, policy_total_counts, value_total_counts, moves_left_logs, final_key_bytes, final_value_key_bytes


def _resolve_soft_target_materialize_workers(cfg, chunk_count):
    materialize_cfg = dict(cfg or {})
    if 'materialize_workers' in materialize_cfg:
        materialize_cfg['workers'] = materialize_cfg.get('materialize_workers')
    return _resolve_positions_per_game_workers(materialize_cfg, chunk_count)


def _soft_target_materialize_chunk_size(cfg, final_count, workers):
    raw = cfg.get('materialize_chunk_size', 250000)
    if isinstance(raw, str) and raw.strip().lower() in {'auto', ''}:
        target_chunks = max(1, int(workers) * 8)
        return max(50000, min(500000, int(np.ceil(float(final_count) / float(target_chunks)))))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 250000


def _build_policy_soft_target_subset(keys, policy_counts, total_counts, moves_left_logs):
    unique_keys = {bytes(key) for key in keys if any(key)}
    policy_subset = {}
    total_subset = {}
    moves_left_subset = {}
    for key in unique_keys:
        count = float(total_counts.get(key, 0.0) or 0.0)
        if count <= 1.0:
            continue
        moves = policy_counts.get(key)
        if moves:
            policy_subset[key] = dict(moves)
        total_subset[key] = float(count)
        moves_left = moves_left_logs.get(key)
        if moves_left:
            moves_left_subset[key] = list(moves_left)
    return policy_subset, total_subset, moves_left_subset


def _build_value_soft_target_subset(keys, wdl_counts, total_counts):
    unique_keys = {bytes(key) for key in keys if any(key)}
    wdl_subset = {}
    total_subset = {}
    for key in unique_keys:
        count = float(total_counts.get(key, 0.0) or 0.0)
        if count <= 1.0:
            continue
        wdl = wdl_counts.get(key)
        if wdl is not None:
            wdl_subset[key] = np.asarray(wdl, dtype=np.float32)
            total_subset[key] = float(count)
    return wdl_subset, total_subset


def _init_soft_target_materialize_worker(binary_file, position_size, cfg, max_policy_moves,
                                         policy_mass_threshold, game_length_by_id, wdl_prior):
    global _SOFT_TARGET_MATERIALIZE_CONTEXT
    _SOFT_TARGET_MATERIALIZE_CONTEXT = {
        'binary_file': binary_file,
        'position_size': int(position_size),
        'cfg': dict(cfg or {}),
        'max_policy_moves': int(max_policy_moves),
        'policy_mass_threshold': float(policy_mass_threshold),
        'game_length_by_id': game_length_by_id,
        'wdl_prior': np.asarray(wdl_prior, dtype=np.float32),
    }


def _materialize_soft_target_chunk(args):
    (
        indices,
        policy_key_bytes,
        value_key_bytes,
        policy_subset,
        policy_total_subset,
        moves_left_subset,
        wdl_subset,
        value_total_subset,
    ) = args
    ctx = _SOFT_TARGET_MATERIALIZE_CONTEXT
    binary_file = ctx['binary_file']
    position_size = int(ctx['position_size'])
    cfg = ctx['cfg']
    max_policy_moves = int(ctx['max_policy_moves'])
    policy_mass_threshold = float(ctx['policy_mass_threshold'])
    game_length_by_id = ctx.get('game_length_by_id')
    wdl_prior = ctx.get('wdl_prior', np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32))
    indices = np.asarray(indices, dtype=np.uint32)
    policy_key_bytes = np.asarray(policy_key_bytes, dtype=np.uint8)
    value_key_bytes = np.asarray(value_key_bytes, dtype=np.uint8)
    row_count = len(indices)

    policy_indices = np.full((row_count, max_policy_moves), -1, dtype=np.int16)
    policy_values = np.zeros((row_count, max_policy_moves), dtype=np.float16)
    value_wdl = np.zeros((row_count, 3), dtype=np.float32)
    occurrence_count = np.zeros((row_count,), dtype=np.float32)
    value_occurrence_count = np.zeros((row_count,), dtype=np.float32)
    sample_weight = np.ones((row_count,), dtype=np.float32)
    value_sample_weight = np.ones((row_count,), dtype=np.float32)
    moves_left_log = np.zeros((row_count,), dtype=np.float32)
    policy_mass_kept = np.ones((row_count,), dtype=np.float32)

    fallback_file = None
    fallback_mm = None
    def _ensure_fallback_mm():
        nonlocal fallback_file, fallback_mm
        if fallback_mm is None:
            fallback_file = open(binary_file, 'rb')
            fallback_mm = mmap.mmap(fallback_file.fileno(), 0, access=mmap.ACCESS_READ)
        return fallback_mm

    try:
        for row, idx in enumerate(indices):
            policy_key = policy_key_bytes[row].tobytes() if row < len(policy_key_bytes) else None
            value_key = value_key_bytes[row].tobytes() if row < len(value_key_bytes) else None
            policy_count = max(1.0, float(policy_total_subset.get(policy_key, 0.0))) if policy_key is not None else 1.0
            value_count = max(1.0, float(value_total_subset.get(value_key, 0.0))) if value_key is not None else 1.0
            occurrence_count[row] = policy_count
            value_occurrence_count[row] = value_count

            ml_values = moves_left_subset.get(policy_key) if policy_key is not None else None
            move_idx = None
            total_moves = None
            if ml_values:
                if len(ml_values) == 1:
                    moves_left_log[row] = float(ml_values[0])
                else:
                    moves_left_log[row] = float(np.median(np.asarray(ml_values, dtype=np.float32)))
            elif game_length_by_id is not None:
                mm = _ensure_fallback_mm()
                offset = int(idx) * position_size
                game_id = struct.unpack('I', mm[offset + 38:offset + 42])[0]
                move_idx = struct.unpack('H', mm[offset + 42:offset + 44])[0]
                total_moves = _lookup_game_length(game_length_by_id, game_id)
                if total_moves > 0:
                    remaining_plies = max(0.0, float(total_moves) - float(move_idx))
                    moves_left_log[row] = float(np.log1p(remaining_plies))

            if move_idx is None or total_moves is None:
                mm = _ensure_fallback_mm()
                offset = int(idx) * position_size
                game_id = struct.unpack('I', mm[offset + 38:offset + 42])[0]
                move_idx = struct.unpack('H', mm[offset + 42:offset + 44])[0]
                total_moves = _lookup_game_length(game_length_by_id, game_id) if game_length_by_id is not None else 0
            sample_weight[row] = _sample_weight_from_count(policy_count, cfg, move_idx, total_moves)
            value_sample_weight[row] = _value_sample_weight_from_context(value_count, move_idx, total_moves, cfg)

            moves = policy_subset.get(policy_key) if policy_key is not None else None
            if moves:
                if len(moves) == 1:
                    move_idx, move_count = next(iter(moves.items()))
                    policy_indices[row, 0] = int(move_idx)
                    policy_values[row, 0] = 1.0
                    policy_mass_kept[row] = 1.0
                else:
                    all_moves = sorted(
                        moves.items(),
                        key=lambda item: (-float(item[1]), int(item[0])),
                    )
                    total_move_mass = sum(float(v) for _, v in all_moves)
                    selected_moves = []
                    cumulative_mass = 0.0
                    for move_idx, move_count in all_moves:
                        if len(selected_moves) >= max_policy_moves:
                            break
                        selected_moves.append((move_idx, move_count))
                        cumulative_mass += float(move_count)
                        if total_move_mass > 0.0 and cumulative_mass / total_move_mass >= policy_mass_threshold:
                            break
                    move_mass = sum(float(v) for _, v in selected_moves)
                    if move_mass > 0.0:
                        policy_mass_kept[row] = float(move_mass) / float(total_move_mass or move_mass)
                        for col, (move_idx, move_count) in enumerate(selected_moves):
                            policy_indices[row, col] = int(move_idx)
                            policy_values[row, col] = float(move_count) / float(move_mass)
            else:
                mm = _ensure_fallback_mm()
                offset = int(idx) * position_size
                move_target = struct.unpack('H', mm[offset + 44:offset + 46])[0]
                if move_target >= ACTION_SIZE and move_target < AZ_ACTION_SIZE:
                    move_target = az_index_to_policy_index(move_target)
                policy_indices[row, 0] = int(move_target)
                policy_values[row, 0] = 1.0

            wdl = wdl_subset.get(value_key) if value_key is not None else None
            if wdl is not None and float(np.sum(wdl)) > 0.0:
                value_wdl[row] = _shrink_wdl_counts(wdl, value_count, cfg, wdl_prior)
            else:
                mm = _ensure_fallback_mm()
                offset = int(idx) * position_size
                outcome = float(struct.unpack('f', mm[offset + 46:offset + 50])[0])
                hard_wdl = np.zeros(3, dtype=np.float32)
                hard_wdl[0 if outcome > 0.9 else 2 if outcome < -0.9 else 1] = 1.0
                value_wdl[row] = _shrink_wdl_counts(hard_wdl, 1.0, cfg, wdl_prior)
    finally:
        if fallback_mm is not None:
            fallback_mm.close()
        if fallback_file is not None:
            fallback_file.close()

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
    final_indices = np.asarray(final_indices, dtype=np.uint32)
    final_key_bytes = np.asarray(final_key_bytes, dtype=np.uint8)
    final_value_key_bytes = np.asarray(final_value_key_bytes, dtype=np.uint8)
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

    wdl_total = np.zeros(3, dtype=np.float64)
    for wdl in wdl_counts.values():
        wdl_total += np.asarray(wdl, dtype=np.float64)
    if float(np.sum(wdl_total)) > 0.0:
        wdl_prior = (wdl_total / float(np.sum(wdl_total))).astype(np.float32)
    else:
        wdl_prior = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

    preliminary_workers = _resolve_soft_target_materialize_workers(cfg, row_count)
    chunk_size = _soft_target_materialize_chunk_size(cfg, row_count, preliminary_workers)
    chunks = list(_soft_target_index_chunks(np.arange(row_count, dtype=np.uint32), chunk_size))
    workers = _resolve_soft_target_materialize_workers(cfg, len(chunks))
    _init_soft_target_materialize_worker(
        binary_file,
        int(position_size),
        dict(cfg),
        int(max_policy_moves),
        float(policy_mass_threshold),
        game_length_by_id,
        wdl_prior,
    )

    def _build_args(row_positions):
        start = int(row_positions[0])
        end = int(row_positions[-1]) + 1
        policy_key_slice = final_key_bytes[start:end]
        value_key_slice = final_value_key_bytes[start:end]
        policy_subset, policy_total_subset, moves_left_subset = _build_policy_soft_target_subset(
            policy_key_slice,
            policy_counts,
            policy_total_counts,
            moves_left_logs,
        )
        wdl_subset, value_total_subset = _build_value_soft_target_subset(
            value_key_slice,
            wdl_counts,
            value_total_counts,
        )
        return (
            final_indices[start:end],
            policy_key_slice,
            value_key_slice,
            policy_subset,
            policy_total_subset,
            moves_left_subset,
            wdl_subset,
            value_total_subset,
        ), start, end

    if workers <= 1 or len(chunks) <= 1:
        iterator = tqdm(chunks, desc=f"  {label} soft_targets materialize", unit="chunk")
        for _, row_positions in iterator:
            args, start, end = _build_args(row_positions)
            result = _materialize_soft_target_chunk(args)
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
        return policy_indices, policy_values, value_wdl, occurrence_count, value_occurrence_count, sample_weight, value_sample_weight, moves_left_log, policy_mass_kept

    max_in_flight = max(1, workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_soft_target_materialize_worker,
        initargs=(
            binary_file,
            int(position_size),
            dict(cfg),
            int(max_policy_moves),
            float(policy_mass_threshold),
            game_length_by_id,
            wdl_prior,
        ),
    ) as executor:
        pending = {}
        chunk_iter = iter(chunks)

        def submit_next():
            try:
                _, row_positions = next(chunk_iter)
            except StopIteration:
                return False
            args, start, end = _build_args(row_positions)
            future = executor.submit(_materialize_soft_target_chunk, args)
            pending[future] = (start, end)
            return True

        for _ in range(min(max_in_flight, len(chunks))):
            submit_next()

        with tqdm(total=row_count, desc=f"  {label} soft_targets materialize", unit="pos") as pbar:
            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    start, end = pending.pop(future)
                    result = future.result()
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

    return policy_indices, policy_values, value_wdl, occurrence_count, value_occurrence_count, sample_weight, value_sample_weight, moves_left_log, policy_mass_kept


def _build_soft_targets(binary_file, position_size, source_indices, final_indices, cfg, label,
                        history_positions, paths, game_length_by_id=None, legacy_paths=None):
    if not cfg.get('enabled', False):
        return None

    final_indices_array = np.asarray(final_indices, dtype=np.uint32)
    cached = _load_soft_target_arrays_with_legacy(
        paths,
        len(final_indices_array),
        cfg,
        label,
        legacy_paths=legacy_paths,
    )
    if cached is not None:
        print(f"  • {label} soft_targets: loaded cached arrays ({len(final_indices_array):,})")
        return cached

    policy_enabled = bool(cfg.get('policy', True))
    value_enabled = bool(cfg.get('value', True))
    max_policy_moves = max(1, int(cfg.get('max_policy_moves', 16)))
    policy_mass_threshold = float(cfg.get('policy_mass_threshold', 1.0) or 1.0)
    policy_mass_threshold = max(0.0, min(1.0, policy_mass_threshold))

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

    unique_positions = len(policy_total_counts)
    avg_count = (sum(policy_total_counts.values()) / max(1, unique_positions)) if unique_positions else 0.0
    unique_value_positions = len(value_total_counts)
    avg_value_count = (sum(value_total_counts.values()) / max(1, unique_value_positions)) if unique_value_positions else 0.0
    mean_mass = float(policy_mass_kept.mean()) if len(policy_mass_kept) else 1.0
    min_mass = float(policy_mass_kept.min()) if len(policy_mass_kept) else 1.0
    low_mass_rows = int((policy_mass_kept < policy_mass_threshold).sum()) if len(policy_mass_kept) else 0
    print(
        f"  • {label} soft_targets: {len(source_indices):,} samples -> "
        f"{unique_positions:,} policy keys (avg_count={avg_count:.2f}), "
        f"{unique_value_positions:,} value keys (avg_count={avg_value_count:.2f}, "
        f"top_moves={max_policy_moves}, kept_mass_mean={mean_mass:.4f}, "
        f"kept_mass_min={min_mass:.4f}, below_threshold={low_mass_rows:,})"
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


def _selection_stage_label(name):
    labels = {
        'positions_per_game': 'positions/game',
        'sample_dedup': 'sample dedup',
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


def _estimate_positions_per_game_count(game_ranges, cfg):
    if not game_ranges:
        return 0
    max_total = max(1, int(cfg.get('max_total', 32)))
    lengths = np.fromiter(
        (max(0, int(end) - int(start)) for _, start, end in game_ranges),
        dtype=np.int64,
        count=len(game_ranges),
    )
    if lengths.size == 0:
        return 0

    selection_mode = str(cfg.get('selection_mode', 'even')).strip().lower()
    if selection_mode == 'smart':
        min_distance = max(1, int(cfg.get('min_distance', 4)))
        spaced_cap = (lengths + min_distance - 1) // min_distance
        per_game = np.minimum(np.minimum(lengths, max_total), spaced_cap)
    else:
        per_game = np.minimum(lengths, max_total)
    return int(np.sum(per_game, dtype=np.int64))


def _maybe_expand_positions_per_game_for_target(cfg, train_ranges, val_ranges, target_total, sample_dedup_cfg):
    if not cfg.get('enabled', False):
        return cfg
    target_total = _resolve_target_positions(target_total)
    if target_total is None:
        return cfg

    ranges = list(train_ranges or []) + list(val_ranges or [])
    if not ranges:
        return cfg

    cfg = dict(cfg)
    current_max_total = max(1, int(cfg.get('max_total', 32)))
    original_min_distance = max(1, int(cfg.get('min_distance', 1)))
    current_estimate = _estimate_positions_per_game_count(ranges, cfg)
    if current_estimate >= target_total:
        return cfg

    raw_total = sum(max(0, int(end) - int(start)) for _, start, end in ranges)
    desired_pre_dedup = min(raw_total, int(target_total))
    if current_estimate >= desired_pre_dedup:
        return cfg

    max_game_len = max(max(0, int(end) - int(start)) for _, start, end in ranges)
    try:
        max_total_before_distance = int(cfg.get('auto_expand_max_total_before_distance', 50))
    except (TypeError, ValueError):
        max_total_before_distance = 50
    max_total_before_distance = max(current_max_total, min(max_game_len, max(1, max_total_before_distance)))

    def _fmt(value):
        return f"{int(value):,}"

    def _delta(before, after):
        value = int(after) - int(before)
        return f"+{value:,}" if value >= 0 else f"{value:,}"

    print(
        f"  - positions_per_game target check: probowalo wybrac {_fmt(current_estimate)} pozycji "
        f"(max_total={current_max_total}, min_distance={original_min_distance}), "
        f"za malo do targetu {_fmt(target_total)}, brakuje {_fmt(int(target_total) - int(current_estimate))}."
    )
    print(
        f"    Step 1: powiekszam max_total do {max_total_before_distance} "
        f"zanim rusze min_distance."
    )

    def _find_max_total(probe_base, upper_limit=None):
        low = max(1, int(probe_base.get('max_total', current_max_total)))
        if _estimate_positions_per_game_count(ranges, probe_base) < desired_pre_dedup:
            low += 1
        limit = max_game_len if upper_limit is None else max(1, min(max_game_len, int(upper_limit)))
        high = min(limit, max(low, int(probe_base.get('max_total', current_max_total))))
        while high < limit:
            high = min(limit, max(high + 1, high * 2))
            probe_cfg = dict(probe_base)
            probe_cfg['max_total'] = high
            if _estimate_positions_per_game_count(ranges, probe_cfg) >= desired_pre_dedup:
                break

        best = high
        probe_cfg = dict(cfg)
        probe_cfg.update(probe_base)
        while low <= high:
            mid = (low + high) // 2
            probe_cfg = dict(probe_base)
            probe_cfg['max_total'] = mid
            estimate = _estimate_positions_per_game_count(ranges, probe_cfg)
            if estimate >= desired_pre_dedup:
                best = mid
                high = mid - 1
            else:
                low = mid + 1

        probe_cfg = dict(probe_base)
        probe_cfg['max_total'] = int(best)
        return int(best), _estimate_positions_per_game_count(ranges, probe_cfg)

    best, final_estimate = _find_max_total(cfg, upper_limit=max_total_before_distance)
    print(
        f"    Step 1 wynik: max_total {current_max_total} -> {int(best)}, "
        f"estimate={_fmt(final_estimate)} ({_delta(current_estimate, final_estimate)})."
    )
    selection_mode = str(cfg.get('selection_mode', 'even')).strip().lower()
    if (
        final_estimate < desired_pre_dedup
        and selection_mode == 'smart'
        and original_min_distance > 3
        and bool(cfg.get('auto_relax_min_distance_for_target', True))
    ):
        print(
            f"    Step 2: nadal za malo; rozluzniam min_distance "
            f"{original_min_distance} -> 3 i szukam max_total do {max_total_before_distance}."
        )
        relaxed_cfg = dict(cfg)
        relaxed_cfg['min_distance'] = 3
        relaxed_best, relaxed_estimate = _find_max_total(
            relaxed_cfg,
            upper_limit=max_total_before_distance,
        )
        if relaxed_estimate > final_estimate:
            cfg = relaxed_cfg
            best = relaxed_best
            previous_estimate = final_estimate
            final_estimate = relaxed_estimate
            print(
                f"    Step 2 wynik: min_distance {original_min_distance} -> 3, "
                f"max_total={int(best)}, estimate={_fmt(final_estimate)} "
                f"({_delta(previous_estimate, final_estimate)})."
            )
        else:
            print(
                f"    Step 2 wynik: brak sensownego zysku "
                f"(estimate bylby {_fmt(relaxed_estimate)})."
            )

    if final_estimate < desired_pre_dedup and int(best) >= max_total_before_distance:
        print(
            f"    Step 3: nadal za malo; powiekszam max_total ponad "
            f"{max_total_before_distance}, do limitu dlugosci gry {max_game_len}."
        )
        previous_best = int(best)
        previous_estimate = final_estimate
        expanded_best, expanded_estimate = _find_max_total(cfg, upper_limit=max_game_len)
        if expanded_estimate > final_estimate:
            best = expanded_best
            final_estimate = expanded_estimate
            print(
                f"    Step 3 wynik: max_total {previous_best} -> {int(best)}, "
                f"estimate={_fmt(final_estimate)} ({_delta(previous_estimate, final_estimate)})."
            )
        else:
            print(
                f"    Step 3 wynik: brak sensownego zysku "
                f"(estimate bylby {_fmt(expanded_estimate)})."
            )

    cfg['max_total'] = int(best)
    final_min_distance = max(1, int(cfg.get('min_distance', original_min_distance)))
    distance_note = ""
    if final_min_distance != original_min_distance:
        distance_note = f", min_distance {original_min_distance} -> {final_min_distance}"
    staged_note = f", max_total-before-distance={max_total_before_distance}"
    print(
        f"  • positions_per_game auto-expanded: max_total {current_max_total} -> {int(best)} "
        f"(target={int(target_total):,}, pre-dedup estimate={final_estimate:,}, "
        f"no duplicate reserve{distance_note}{staged_note})"
    )
    if final_estimate < int(target_total):
        print(
            f"  ! positions_per_game cannot reach target with current data/spacing: "
            f"estimate={final_estimate:,}, target={int(target_total):,}. "
            "Use more PGNs or lower min_distance manually."
        )
    return cfg


def _positions_per_game_cache_config(cfg):
    cfg = dict(cfg or {})
    # Runtime-only knobs must not invalidate the selected-position cache.
    cfg.pop('workers', None)
    cfg.pop('chunk_size', None)
    return cfg


def _build_index_cache_paths(metadata, config):
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
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
    final_index_payload = dict(selector_payload)
    final_index_payload.update({
        'index_cache_schema': 6,
        'history_positions': int(model_cfg.get('history_positions', 0) or 0),
        'sample_dedup': data_cfg.get('sample_dedup', {}),
        'target_positions': data_cfg.get('target_positions', 'max'),
    })
    soft_payload = dict(final_index_payload)
    soft_payload.update({
        'soft_target_cache_schema': 2,
        'action_size': int(ACTION_SIZE),
        'soft_targets': _soft_targets_cache_config(data_cfg.get('soft_targets', {})),
    })
    selector_digest = hashlib.sha1(
        json.dumps(selector_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    ppg_all_digest = hashlib.sha1(
        json.dumps(ppg_all_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:16]
    index_digest = hashlib.sha1(
        json.dumps(final_index_payload, sort_keys=True, default=str).encode('utf-8')
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
        'index_digest': index_digest,
        'soft_digest': soft_digest,
        'manifest': cache_dir / f"il_prepare_{index_digest}_{soft_digest}.json",
        'game_ranges': cache_dir / f"il_game_ranges_{range_digest}.npz",
        'legacy_game_ranges': [cache_dir / f"il_game_ranges_{digest}.npz" for digest in legacy_range_digests],
        'train': cache_dir / f"il_indices_{index_digest}_train.npy",
        'val': cache_dir / f"il_indices_{index_digest}_val.npy",
        'legacy_train': [cache_dir / f"il_indices_{digest}_train.npy" for digest in legacy_final_digests],
        'legacy_val': [cache_dir / f"il_indices_{digest}_val.npy" for digest in legacy_final_digests],
        'train_ppg': cache_dir / f"il_indices_{selector_digest}_train_ppg.npy",
        'val_ppg': cache_dir / f"il_indices_{selector_digest}_val_ppg.npy",
        'all_ppg': cache_dir / f"il_indices_{ppg_all_digest}_all_ppg.npy",
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
                 filter_stats=None):
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
        if position_size < 50:  # 🔧 FIXED: Minimum size with metadata (38B board + 12B metadata)
            raise ValueError(f"position_size too small: {position_size} (minimum 50 bytes)")
        
        self.binary_file = binary_file
        self.position_size = position_size
        self.history_positions = history_positions
        self.game_length_by_id = game_length_by_id
        board_dtype = str(board_dtype or 'float32').strip().lower()
        self.board_np_dtype = np.float16 if board_dtype in {'float16', 'fp16', 'half'} else np.float32
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
        # [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [MTL (12B)]
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


class NumpyWeightedSampler(Sampler):
    """Replacement sampler with NumPy probabilities and optional opening floor."""

    def __init__(self, data_source, weights, seed=0, num_samples=None,
                 phase_labels=None, opening_floor=0.0):
        self.data_source = data_source
        self.seed = int(seed or 0)
        self.epoch = 0
        n = len(data_source)
        if n <= 0:
            raise ValueError("NumpyWeightedSampler requires a non-empty data source.")
        self.num_samples = int(num_samples if num_samples is not None else n)
        self.num_samples = max(1, self.num_samples)

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n:
            raise ValueError(
                f"Sampler weights length mismatch: got {weights.shape[0]:,}, expected {n:,}"
            )
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            raise ValueError("NumpyWeightedSampler requires at least one positive weight.")

        self.probabilities = None
        self.opening_floor = max(0.0, min(1.0, float(opening_floor or 0.0)))
        self.opening_indices = None
        self.opening_probabilities = None
        self.other_indices = None
        self.other_probabilities = None
        self.natural_opening_probability = 0.0

        if phase_labels is not None and self.opening_floor > 0.0:
            phase_labels = np.asarray(phase_labels, dtype=np.uint8)
            if phase_labels.shape[0] == n:
                opening_indices = np.flatnonzero(phase_labels == 0)
                other_indices = np.flatnonzero(phase_labels != 0)
                if len(opening_indices) > 0 and len(other_indices) > 0:
                    opening_weights = weights[opening_indices]
                    other_weights = weights[other_indices]
                    opening_sum = float(opening_weights.sum())
                    other_sum = float(other_weights.sum())
                    if opening_sum > 0.0 and other_sum > 0.0:
                        index_dtype = np.uint32 if n <= np.iinfo(np.uint32).max else np.int64
                        self.opening_indices = opening_indices.astype(index_dtype, copy=False)
                        self.other_indices = other_indices.astype(index_dtype, copy=False)
                        self.opening_probabilities = opening_weights / opening_sum
                        self.other_probabilities = other_weights / other_sum
                        self.natural_opening_probability = opening_sum / total_weight
        if self.opening_indices is None or self.other_indices is None:
            self.probabilities = weights / total_weight

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        if self.opening_indices is not None and self.other_indices is not None:
            opening_share = max(self.opening_floor, self.natural_opening_probability)
            opening_count = int(round(self.num_samples * opening_share))
            opening_count = max(0, min(self.num_samples, opening_count))
            other_count = self.num_samples - opening_count

            draws = []
            if opening_count > 0:
                draws.append(
                    rng.choice(
                        self.opening_indices,
                        size=opening_count,
                        replace=True,
                        p=self.opening_probabilities,
                    )
                )
            if other_count > 0:
                draws.append(
                    rng.choice(
                        self.other_indices,
                        size=other_count,
                        replace=True,
                        p=self.other_probabilities,
                    )
                )
            sampled = draws[0] if len(draws) == 1 else np.concatenate(draws)
            rng.shuffle(sampled)
        else:
            sampled = rng.choice(
                len(self.data_source),
                size=self.num_samples,
                replace=True,
                p=self.probabilities,
            )

        for idx in sampled:
            yield int(idx)

    def __len__(self):
        return self.num_samples


def _resolve_weighted_epoch_size(dataset_size, cfg):
    multiplier = cfg.get('epoch_size_multiplier', 1.0)
    try:
        multiplier = float(multiplier)
    except (TypeError, ValueError):
        multiplier = 1.0
    multiplier = max(0.05, multiplier)
    raw_num_samples = cfg.get('num_samples', None)
    if raw_num_samples is None:
        return max(1, int(round(int(dataset_size) * multiplier)))
    if isinstance(raw_num_samples, str):
        text = raw_num_samples.strip().lower()
        if text in {'dataset', 'len', 'auto'}:
            return max(1, int(round(int(dataset_size) * multiplier)))
        try:
            raw_num_samples = int(float(text.replace(',', '.')))
        except ValueError:
            return max(1, int(round(int(dataset_size) * multiplier)))
    try:
        return max(1, int(raw_num_samples))
    except (TypeError, ValueError):
        return max(1, int(round(int(dataset_size) * multiplier)))


def _build_policy_sampling_weights(dataset, cfg):
    soft_targets = getattr(dataset, 'soft_targets', None)
    if soft_targets is None:
        return None, "soft_targets_missing"

    source = str(cfg.get('source', 'sample_weight') or 'sample_weight').strip().lower()
    key_map = {
        'sample_weight': 'sample_weight',
        'policy_weight': 'sample_weight',
        'occurrence': 'occurrence_count',
        'occurrence_count': 'occurrence_count',
        'count': 'occurrence_count',
    }
    key = key_map.get(source)
    if key is None or key not in soft_targets:
        return None, f"missing_{source}"

    weights = np.asarray(soft_targets[key], dtype=np.float64)
    if len(weights) != len(dataset):
        return None, f"length_mismatch_{key}"
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

    try:
        min_weight = max(0.0, float(cfg.get('min_weight', 0.0) or 0.0))
    except (TypeError, ValueError):
        min_weight = 0.0
    try:
        max_weight = max(0.0, float(cfg.get('max_weight', 0.0) or 0.0))
    except (TypeError, ValueError):
        max_weight = 0.0
    if min_weight > 0.0:
        weights = np.maximum(weights, min_weight)
    if max_weight > 0.0:
        weights = np.minimum(weights, max_weight)

    try:
        power = float(cfg.get('weight_power', 1.0))
    except (TypeError, ValueError):
        power = 1.0
    power = max(0.05, min(4.0, power))
    if abs(power - 1.0) > 1.0e-8:
        weights = np.power(weights, power)

    if float(weights.sum()) <= 0.0:
        return None, f"zero_{key}"
    return weights, key


def _compute_phase_labels_for_indices(binary_file, position_size, total_positions, indices,
                                      game_length_by_id, cfg):
    if game_length_by_id is None or len(indices) == 0:
        return None
    try:
        opening_max = float(cfg.get('opening_progress_max', 0.25))
    except (TypeError, ValueError):
        opening_max = 0.25
    try:
        endgame_min = float(cfg.get('endgame_progress_min', 0.75))
    except (TypeError, ValueError):
        endgame_min = 0.75
    opening_max = max(0.0, min(1.0, opening_max))
    endgame_min = max(opening_max, min(1.0, endgame_min))
    try:
        chunk_size = int(cfg.get('phase_chunk_size', 1_000_000))
    except (TypeError, ValueError):
        chunk_size = 1_000_000
    chunk_size = max(10_000, chunk_size)

    record_dtype = np.dtype({
        'names': ['game_id', 'move_idx'],
        'formats': ['<u4', '<u2'],
        'offsets': [38, 42],
        'itemsize': int(position_size),
    })
    records = np.memmap(binary_file, dtype=record_dtype, mode='r', shape=(int(total_positions),))
    labels = np.ones((len(indices),), dtype=np.uint8)
    try:
        for start in range(0, len(indices), chunk_size):
            end = min(len(indices), start + chunk_size)
            chunk_indices = np.asarray(indices[start:end], dtype=np.int64)
            game_ids = np.asarray(records['game_id'][chunk_indices], dtype=np.int64)
            move_idx = np.asarray(records['move_idx'][chunk_indices], dtype=np.float32)

            if isinstance(game_length_by_id, np.ndarray):
                total_moves = np.zeros((len(chunk_indices),), dtype=np.float32)
                valid = (game_ids >= 0) & (game_ids < len(game_length_by_id))
                total_moves[valid] = np.asarray(game_length_by_id[game_ids[valid]], dtype=np.float32)
            else:
                total_moves = np.fromiter(
                    (_lookup_game_length(game_length_by_id, int(game_id)) for game_id in game_ids),
                    dtype=np.float32,
                    count=len(game_ids),
                )

            progress = np.ones((len(chunk_indices),), dtype=np.float32)
            valid_total = total_moves > 1.0
            progress[valid_total] = move_idx[valid_total] / total_moves[valid_total]
            progress = np.clip(progress, 0.0, 1.0)
            labels[start:end][progress <= opening_max] = 0
            labels[start:end][progress >= endgame_min] = 2
    finally:
        del records
    return labels


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
        record_dtype = np.dtype({
            'names': ['game_id'],
            'formats': ['<u4'],
            'offsets': [38],
            'itemsize': position_size,
        })
        records = np.memmap(binary_file, dtype=record_dtype, mode='r', shape=(total_positions,))
        game_ids = records['game_id']
        change_points = np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1
        starts = np.concatenate(([0], change_points)).astype(np.int64, copy=False)
        ends = np.concatenate((change_points, [total_positions])).astype(np.int64, copy=False)
        start_game_ids = np.asarray(game_ids[starts], dtype=np.uint32)
        ranges = [
            (int(game_id), int(start), int(end))
            for game_id, start, end in zip(start_game_ids, starts, ends)
        ]
        del records
        return ranges
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
    tail_count = total_positions - start_position
    try:
        record_dtype = np.dtype({
            'names': ['game_id'],
            'formats': ['<u4'],
            'offsets': [38],
            'itemsize': int(position_size),
        })
        records = np.memmap(
            binary_file,
            dtype=record_dtype,
            mode='r',
            offset=start_position * int(position_size),
            shape=(tail_count,),
        )
        game_ids = records['game_id']
        change_points = np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1
        starts_local = np.concatenate(([0], change_points)).astype(np.int64, copy=False)
        ends_local = np.concatenate((change_points, [tail_count])).astype(np.int64, copy=False)
        start_game_ids = np.asarray(game_ids[starts_local], dtype=np.uint32)
        ranges = [
            (int(game_id), int(start_position + start), int(start_position + end))
            for game_id, start, end in zip(start_game_ids, starts_local, ends_local)
        ]
        del records
        return ranges
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
            else:
                print(f"  - Loaded cached game ranges ({len(ranges):,} games)")
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


def _build_game_length_map(binary_file, position_size, total_positions):
    """
    Build {game_id: total_moves} mapping using contiguous ranges.
    """
    ranges = _build_game_ranges(binary_file, position_size, total_positions)
    return {game_id: end - start for game_id, start, end in ranges}


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
    
    print(f"  • Loading/building game ranges for {total_positions:,} binary records...")
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
    print(f"  • Found {len(game_ranges):,} games for split")
    
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
    train_sampling_cfg = dict(data_cfg.get('train_sampling', {}) or {})
    positions_per_game_enabled = bool(positions_per_game_cfg.get('enabled', False))
    train_soft_targets = None
    val_soft_targets = None
    selection_stages = []
    game_length_by_id = None
    need_game_length = True
    index_cache_paths = _build_index_cache_paths(metadata, config)
    history_positions = config['model'].get('history_positions', 0)
    hw_cfg = config.get('hardware', {})
    
    if split_by_game:
        if need_game_length:
            print("\n" + "=" * 70)
            print("Preparing IL dataset indices/cache...")
            print(f"  • Binary positions: {total_positions:,}")
            print(f"  • positions_per_game: {'on' if positions_per_game_enabled else 'off'}")
            print(f"  • sample_dedup: {'on' if sample_dedup_cfg.get('enabled', False) else 'off'}")
            print(f"  • soft_targets: {'on' if soft_targets_cfg.get('enabled', False) else 'off'}")
            print(f"  • cache_dir: {index_cache_paths['cache_dir']}")
            print(f"  • cache keys: idx={index_cache_paths['index_digest']}, soft={index_cache_paths['soft_digest']}")
            manifest_path = Path(index_cache_paths['manifest'])
            print(f"  • prepare manifest: {'found' if manifest_path.exists() else 'will save'} ({manifest_path.name})")
            print("=" * 70)
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
                original_positions_per_game_cfg = dict(positions_per_game_cfg)
                positions_per_game_cfg = _maybe_expand_positions_per_game_for_target(
                    positions_per_game_cfg,
                    train_ranges,
                    val_ranges,
                    data_cfg.get('target_positions', 'max'),
                    sample_dedup_cfg,
                )
                if _positions_per_game_cache_config(positions_per_game_cfg) != _positions_per_game_cache_config(original_positions_per_game_cfg):
                    data_cfg = dict(data_cfg)
                    data_cfg['positions_per_game'] = dict(positions_per_game_cfg)
                    config['data'] = data_cfg
                    index_cache_paths = _build_index_cache_paths(metadata, config)
                    print(
                        f"  • cache keys updated for expanded positions_per_game: "
                        f"idx={index_cache_paths['index_digest']}, soft={index_cache_paths['soft_digest']}"
                    )
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
                print(
                    f"  - Train positions_per_game split: {raw_train_count:,} -> {len(train_indices):,} "
                    f"(from all-cache {len(all_ppg_indices):,})"
                )
                print(
                    f"  - Val positions_per_game split: {raw_val_count:,} -> {len(val_indices):,} "
                    f"(from all-cache {len(all_ppg_indices):,})"
                )
            selection_stages.append(("positions_per_game", len(train_indices), len(val_indices)))
            train_soft_source_indices = train_indices
            val_soft_source_indices = val_indices
            train_indices = _dedupe_indices_by_signature(
                metadata['binary_file'],
                metadata['position_size'],
                train_indices,
                sample_dedup_cfg,
                "Train",
                history_positions,
                cache_path=index_cache_paths['train'],
                legacy_cache_paths=index_cache_paths['legacy_train'],
            )
            val_indices = _dedupe_indices_by_signature(
                metadata['binary_file'],
                metadata['position_size'],
                val_indices,
                sample_dedup_cfg,
                "Val",
                history_positions,
                cache_path=index_cache_paths['val'],
                legacy_cache_paths=index_cache_paths['legacy_val'],
            )
            selection_stages.append(("sample_dedup", len(train_indices), len(val_indices)))
            target_train_count, target_val_count, target_limited = _split_target_counts(
                len(train_indices),
                len(val_indices),
                data_cfg.get('target_positions', 'max'),
                data_cfg.get('train_split', 0.85),
            )
            if target_limited:
                before_train = len(train_indices)
                before_val = len(val_indices)
                train_indices = _evenly_limit_indices(train_indices, target_train_count)
                val_indices = _evenly_limit_indices(val_indices, target_val_count)
                print(
                    f"  • IL target_positions: {before_train + before_val:,} -> "
                    f"{len(train_indices) + len(val_indices):,} "
                    f"(train={len(train_indices):,}, val={len(val_indices):,})"
                )
            else:
                target_requested = _resolve_target_positions(data_cfg.get('target_positions', 'max'))
                available_after_dedup = len(train_indices) + len(val_indices)
                print(
                    f"  • IL target_positions: using all selected positions "
                    f"({available_after_dedup:,})"
                )
                if target_requested is not None and available_after_dedup < int(target_requested):
                    print(
                        f"  ! IL target_positions shortfall: requested {int(target_requested):,}, "
                        f"available {available_after_dedup:,}. Add more PGNs/data, reduce dedup, "
                        "or lower positions_per_game spacing."
                    )
            selection_stages.append(("target_positions", len(train_indices), len(val_indices)))
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
    print("\nIL Dataset Ready")
    print("-" * 70)
    print(f"  Binary positions : {total_positions:,}")
    print(f"  Selected samples : {selected_total:,} (train={len(train_indices):,}, val={len(val_indices):,})")
    print(f"  Encoding         : {position_size}B records, {input_planes} planes (16 x {1 + history_positions})")
    if positions_per_game_cfg.get('enabled', False):
        phase_budgets = _positions_per_game_budgets(positions_per_game_cfg)
        print(
            f"  Per-game pick   : {positions_per_game_cfg.get('selection_mode', 'even')}, "
            f"max={positions_per_game_cfg.get('max_total', 32)}, "
            f"min_distance={positions_per_game_cfg.get('min_distance', 1)}, "
            f"budgets O/M/E/R="
            f"{phase_budgets.get('opening', 0)}/"
            f"{phase_budgets.get('middlegame', 0)}/"
            f"{phase_budgets.get('endgame', 0)}/"
            f"{phase_budgets.get('rare_or_eventful', 0)}"
        )
    else:
        print("  Per-game pick   : off")
    print("  Split           : by game")
    if sample_dedup_cfg.get('enabled', False):
        print(
            f"  Dedup           : {sample_dedup_cfg.get('mode', 'position_plus_move')}, "
            f"max_count={sample_dedup_cfg.get('max_count', 4)}, "
            f"turn/history={sample_dedup_cfg.get('include_turn', True)}/"
            f"{sample_dedup_cfg.get('include_history', True)}"
        )
    else:
        print("  Dedup           : off")
    if soft_targets_cfg.get('enabled', False):
        sw_cfg = soft_targets_cfg.get('sample_weight', {}) or {}
        print(
            f"  Soft targets    : {soft_targets_cfg.get('mode', 'fen')}, "
            f"policy/value={soft_targets_cfg.get('policy', True)}/{soft_targets_cfg.get('value', True)}, "
            f"turn/history={soft_targets_cfg.get('include_turn', True)}/"
            f"{soft_targets_cfg.get('include_history', True)}, "
            f"top_moves={soft_targets_cfg.get('max_policy_moves', 16)}"
        )
        print(
            f"  Sample weights  : enabled={sw_cfg.get('enabled', True)}, "
            f"mode={sw_cfg.get('mode', 'occurrence_count')}, max={sw_cfg.get('max', 0)}"
        )
    else:
        print("  Soft targets    : off")
    if train_sampling_cfg.get('enabled', False):
        print(
            f"  Train sampler   : {train_sampling_cfg.get('mode', 'policy_weighted')}, "
            f"source={train_sampling_cfg.get('source', 'sample_weight')}, "
            f"power={train_sampling_cfg.get('weight_power', 1.0)}, "
            f"opening_floor={train_sampling_cfg.get('opening_floor', 0.0)}"
        )
    else:
        print("  Train sampler   : default shuffle")
    if split_by_game and game_count is not None:
        print(f"  Games           : {game_count:,} (train={train_game_count:,}, val={val_game_count:,})")
    print("-" * 70)
    
    # Create datasets
    use_index_mmap = bool(config.get('hardware', {}).get('dataloader_index_mmap', True))
    dataloader_board_dtype = str(hw_cfg.get('dataloader_board_dtype', 'float32') or 'float32')
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
        filter_stats=train_filter_stats,
    )
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
    
    print(f"  • Train positions (selected): {len(train_dataset):,}")
    print(f"  • Val positions (selected): {len(val_dataset):,}")
    print(f"{'='*70}\n")
    
    # Dataloaders
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
    train_in_order = bool(hw_cfg.get('dataloader_train_in_order', True))
    val_in_order = bool(hw_cfg.get('dataloader_val_in_order', True))
    train_sampler = None
    train_sampling_label = "shuffle"
    train_sampling_mode = str(train_sampling_cfg.get('mode', 'shuffle') or 'shuffle').strip().lower()
    weighted_modes = {'weighted', 'policy_weighted', 'weighted_policy'}
    if bool(train_sampling_cfg.get('enabled', False)) and train_sampling_mode in weighted_modes:
        weights, weight_source = _build_policy_sampling_weights(train_dataset, train_sampling_cfg)
        if weights is None:
            print(f"  ! Train sampling: weighted sampler disabled ({weight_source}); using shuffle.")
        else:
            opening_floor = max(0.0, min(1.0, float(train_sampling_cfg.get('opening_floor', 0.0) or 0.0)))
            phase_labels = None
            if opening_floor > 0.0:
                phase_labels = _compute_phase_labels_for_indices(
                    metadata['binary_file'],
                    position_size,
                    total_positions,
                    train_dataset.indices,
                    game_length_by_id,
                    train_sampling_cfg,
                )
            num_samples = _resolve_weighted_epoch_size(len(train_dataset), train_sampling_cfg)
            train_sampler = NumpyWeightedSampler(
                train_dataset,
                weights,
                seed=config.get('seed', 0),
                num_samples=num_samples,
                phase_labels=phase_labels,
                opening_floor=opening_floor,
            )
            train_sampling_label = f"policy_weighted:{weight_source}"
            weight_min = float(np.min(weights)) if len(weights) else 0.0
            weight_avg = float(np.mean(weights)) if len(weights) else 0.0
            weight_max = float(np.max(weights)) if len(weights) else 0.0
            print(
                f"  • Train sampling mode: {train_sampling_label} "
                f"(epoch_samples={len(train_sampler):,}, weights min/avg/max="
                f"{weight_min:.3f}/{weight_avg:.3f}/{weight_max:.3f})"
            )
            if phase_labels is not None:
                opening_mask = phase_labels == 0
                opening_rows = int(np.count_nonzero(opening_mask))
                opening_row_frac = opening_rows / max(1, len(phase_labels))
                opening_weight_frac = (
                    float(weights[opening_mask].sum()) / max(float(weights.sum()), 1.0e-12)
                    if opening_rows > 0 else 0.0
                )
                effective_opening = max(opening_floor, opening_weight_frac)
                print(
                    f"    - opening rows/weighted/effective: "
                    f"{opening_row_frac:.2%}/{opening_weight_frac:.2%}/{effective_opening:.2%}"
                )
    if train_sampler is None:
        train_sampler = (
            NumpyRandomSampler(train_dataset, seed=config.get('seed', 0))
            if use_numpy_shuffle
            else None
        )
        train_sampling_label = "numpy_shuffle" if train_sampler is not None else "torch_shuffle"
        print(f"  • Train sampling mode: {train_sampling_label}")
    
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
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=val_pin_memory,
        persistent_workers=val_persistent_workers,
        prefetch_factor=val_prefetch_factor if val_num_workers > 0 else None,
        in_order=val_in_order if val_num_workers > 0 else True,
    )
    
    print(
        "DataLoaders created "
        f"(train_batch={train_batch_size}, eval_batch={eval_batch_size}, "
        f"workers={train_num_workers}/{val_num_workers}, "
        f"prefetch={train_prefetch_factor}/{val_prefetch_factor}, "
        f"in_order={'on' if train_in_order else 'off'}/{'on' if val_in_order else 'off'}, "
        f"board_dtype={dataloader_board_dtype}, "
        f"index_mmap={'on' if use_index_mmap else 'off'}, "
        f"train_sampling={train_sampling_label})"
    )
    
    return train_loader, val_loader
