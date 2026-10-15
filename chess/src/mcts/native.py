"""Optional native CPU kernels for the Python MCTS tree.

The extension deliberately uses a small C ABI instead of the PyTorch C++ ABI:
the hot data already lives in NumPy arrays, and a standalone DLL avoids a
PyTorch/MSVC rebuild check in every Windows ``spawn`` worker.  The source is
hashed into the cache path, built once under a process-safe lock, and never
written into the repository.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from filelock import FileLock


_API_VERSION = 7
_INSTANCE = None
_LOAD_ATTEMPTED = False
_INSTANCE_LOCK = threading.Lock()


def _enabled_by_environment() -> bool:
    value = str(os.environ.get("CHESS_NATIVE_MCTS", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _source_path() -> Path:
    return Path(__file__).resolve().parent / "cpp" / "mcts_kernels.cpp"


def _find_compiler() -> str | None:
    override = os.environ.get("CHESS_CXX")
    if override:
        resolved = shutil.which(override)
        if resolved:
            return resolved
    for candidate in ("g++", "clang++"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _cache_root() -> Path:
    override = os.environ.get("CHESS_NATIVE_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # Keep the default next to the chess package. This directory is writable in
    # the normal source checkout, survives Windows worker restarts, and avoids
    # corporate/sandbox policies that deny new directories under LOCALAPPDATA.
    return Path(__file__).resolve().parents[2] / ".cache" / "native_mcts"


def _build_library(source: Path, compiler: str | None) -> Path:
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(
        source_bytes
        + f"abi={_API_VERSION};std=c++17;opt=O3;static-runtime=v1".encode("ascii")
        + sys.platform.encode("ascii")
        + str(ctypes.sizeof(ctypes.c_void_p)).encode("ascii")
    ).hexdigest()[:20]
    build_dir = _cache_root() / digest
    suffix = ".dll" if os.name == "nt" else ".so"
    library = build_dir / f"mcts_kernels{suffix}"
    if library.is_file():
        return library
    if compiler is None:
        raise RuntimeError("native MCTS cache is empty and no g++/clang++ compiler was found")

    build_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(build_dir / "build.lock"), timeout=180)
    with lock:
        if library.is_file():
            return library
        temporary = build_dir / f"mcts_kernels.{os.getpid()}.tmp{suffix}"
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fno-math-errno",
            "-shared",
        ]
        if os.name != "nt":
            command.append("-fPIC")
        elif "g++" in Path(compiler).name.lower():
            # MinGW's libstdc++ pulls libwinpthread even when libgcc/libstdc++
            # are static. A fully static runtime keeps spawned workers from
            # depending on the compiler's bin directory being on DLL search
            # paths (Python 3.8+ intentionally no longer searches PATH here).
            command.extend(["-static", "-static-libgcc", "-static-libstdc++"])
        command.extend([str(source), "-o", str(temporary)])
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0 or not temporary.is_file():
                details = (completed.stderr or completed.stdout or "unknown compiler error").strip()
                raise RuntimeError(f"native MCTS build failed: {details}")
            os.replace(temporary, library)
        finally:
            temporary.unlink(missing_ok=True)
    return library


def _pointer(array: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(array.ctypes.data))


class NativeMCTSKernels:
    """Thin, allocation-free wrappers around the native MCTS kernels."""

    backend_name = "cpp-dll"

    def __init__(self, library_path: Path):
        # PyDLL keeps the GIL held. These kernels are only a few dozen scalar
        # operations; releasing/reacquiring it would cost more than it helps.
        self.library_path = Path(library_path)
        self._library = ctypes.PyDLL(str(self.library_path))
        self._bind()
        version = int(self._library.mcts_api_version())
        if version != _API_VERSION:
            raise RuntimeError(
                f"native MCTS ABI mismatch: Python={_API_VERSION}, DLL={version}"
            )

    def _bind(self) -> None:
        lib = self._library
        lib.mcts_api_version.argtypes = []
        lib.mcts_api_version.restype = ctypes.c_int32

        board_plane_args = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.mcts_encode_board_planes_f32.argtypes = board_plane_args
        lib.mcts_encode_board_planes_f32.restype = ctypes.c_int32
        lib.mcts_encode_board_planes_f16.argtypes = board_plane_args
        lib.mcts_encode_board_planes_f16.restype = ctypes.c_int32
        lib.mcts_encode_move_hashes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.mcts_encode_move_hashes.restype = ctypes.c_int32

        lib.mcts_hash_positions.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
        ]
        lib.mcts_hash_positions.restype = ctypes.c_int32

        lib.mcts_completed_q.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_double, ctypes.c_int32, ctypes.c_double, ctypes.c_void_p,
        ]
        lib.mcts_completed_q.restype = ctypes.c_int32
        lib.mcts_improved_policy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
        ]
        lib.mcts_improved_policy.restype = ctypes.c_int32
        lib.mcts_select_root.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int32,
        ]
        lib.mcts_select_root.restype = ctypes.c_int32
        lib.mcts_final_action.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_double, ctypes.c_double,
        ]
        lib.mcts_final_action.restype = ctypes.c_int32
        lib.mcts_soften_policy.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_double, ctypes.c_void_p,
        ]
        lib.mcts_soften_policy.restype = ctypes.c_int32

        lib.mcts_forest_create.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32,
        ]
        lib.mcts_forest_create.restype = ctypes.c_void_p
        lib.mcts_forest_destroy.argtypes = [ctypes.c_void_p]
        lib.mcts_forest_destroy.restype = None
        lib.mcts_forest_add_node.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_double, ctypes.c_int32, ctypes.c_double, ctypes.c_int32,
        ]
        lib.mcts_forest_add_node.restype = ctypes.c_int32
        lib.mcts_forest_set_edges.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ]
        lib.mcts_forest_set_edges.restype = ctypes.c_int32
        lib.mcts_forest_configure_root.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
        ]
        lib.mcts_forest_configure_root.restype = ctypes.c_int32
        lib.mcts_forest_reroot.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_double, ctypes.c_int32,
            ctypes.c_double, ctypes.c_int32,
        ]
        lib.mcts_forest_reroot.restype = ctypes.c_int32
        lib.mcts_forest_select_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.mcts_forest_select_batch.restype = ctypes.c_int32
        lib.mcts_forest_expand.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_double,
        ]
        lib.mcts_forest_expand.restype = ctypes.c_int32
        lib.mcts_forest_expand_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ]
        lib.mcts_forest_expand_batch.restype = ctypes.c_int32
        lib.mcts_forest_backup_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ]
        lib.mcts_forest_backup_batch.restype = ctypes.c_int32
        lib.mcts_forest_export_node.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ]
        lib.mcts_forest_export_node.restype = ctypes.c_int32
        lib.mcts_forest_node_count.argtypes = [ctypes.c_void_p]
        lib.mcts_forest_node_count.restype = ctypes.c_int32
        lib.mcts_forest_edge_count.argtypes = [ctypes.c_void_p]
        lib.mcts_forest_edge_count.restype = ctypes.c_int32
        lib.mcts_forest_export_all.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ]
        lib.mcts_forest_export_all.restype = ctypes.c_int32
        lib.mcts_forest_compact.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_int32,
        ]
        lib.mcts_forest_compact.restype = ctypes.c_int32

    def hash_positions(
        self,
        boards: np.ndarray,
        legal_indices: np.ndarray,
        legal_counts: np.ndarray,
        output: np.ndarray,
    ) -> None:
        if boards.ndim < 2 or not boards.flags.c_contiguous:
            raise ValueError("boards must be a contiguous batched array")
        if legal_indices.ndim != 2 or not legal_indices.flags.c_contiguous:
            raise ValueError("legal_indices must be a contiguous matrix")
        if legal_counts.dtype != np.int16 or not legal_counts.flags.c_contiguous:
            raise ValueError("legal_counts must be contiguous int16")
        board_bytes = int(np.prod(boards.shape[1:], dtype=np.int64)) * int(boards.dtype.itemsize)
        status = self._library.mcts_hash_positions(
            _pointer(boards), int(boards.strides[0]), board_bytes,
            _pointer(legal_indices), int(legal_indices.shape[1]), _pointer(legal_counts),
            int(boards.shape[0]), _pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"mcts_hash_positions failed with status {status}")

    def encode_board_planes(
        self,
        piece_masks: np.ndarray,
        black_to_move: np.ndarray,
        castling_masks: np.ndarray,
        en_passant_masks: np.ndarray,
        halfmove_values: np.ndarray,
        fullmove_values: np.ndarray,
        output: np.ndarray,
    ) -> None:
        batch_size = int(piece_masks.shape[0])
        if (
            piece_masks.dtype != np.uint64
            or piece_masks.shape != (batch_size, 12)
            or not piece_masks.flags.c_contiguous
        ):
            raise ValueError("piece_masks must be contiguous uint64 with shape (batch, 12)")
        expected_vectors = (
            (black_to_move, np.uint8, "black_to_move"),
            (castling_masks, np.uint64, "castling_masks"),
            (en_passant_masks, np.uint64, "en_passant_masks"),
            (halfmove_values, np.float32, "halfmove_values"),
            (fullmove_values, np.float32, "fullmove_values"),
        )
        for values, dtype, name in expected_vectors:
            if (
                values.dtype != np.dtype(dtype)
                or values.shape != (batch_size,)
                or not values.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be contiguous {np.dtype(dtype)} with shape (batch,)")
        if (
            output.shape != (batch_size, 16, 8, 8)
            or output.strides[1:] != (
                8 * 8 * output.dtype.itemsize,
                8 * output.dtype.itemsize,
                output.dtype.itemsize,
            )
        ):
            raise ValueError(
                "output must have shape (batch, 16, 8, 8) with contiguous planes"
            )
        if output.dtype == np.float16:
            function = self._library.mcts_encode_board_planes_f16
        elif output.dtype == np.float32:
            function = self._library.mcts_encode_board_planes_f32
        else:
            raise ValueError("output must use float16 or float32")
        status = function(
            _pointer(piece_masks),
            _pointer(black_to_move),
            _pointer(castling_masks),
            _pointer(en_passant_masks),
            _pointer(halfmove_values),
            _pointer(fullmove_values),
            batch_size,
            int(output.strides[0] // output.dtype.itemsize),
            _pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"mcts_encode_board_planes failed with status {status}")

    def encode_move_hashes(
        self,
        move_hashes: np.ndarray,
        node_offsets: np.ndarray,
        black_to_move: np.ndarray,
        output: np.ndarray,
    ) -> None:
        move_hashes = np.asarray(move_hashes, dtype=np.int64)
        node_offsets = np.asarray(node_offsets, dtype=np.int32)
        black_to_move = np.asarray(black_to_move, dtype=np.uint8)
        output = np.asarray(output)
        if move_hashes.ndim != 1 or not move_hashes.flags.c_contiguous:
            raise ValueError("move_hashes must be a contiguous int64 vector")
        if node_offsets.ndim != 1 or not node_offsets.flags.c_contiguous:
            raise ValueError("node_offsets must be a contiguous int32 vector")
        if black_to_move.ndim != 1 or not black_to_move.flags.c_contiguous:
            raise ValueError("black_to_move must be a contiguous uint8 vector")
        if output.dtype != np.int32 or output.ndim != 1 or not output.flags.c_contiguous:
            raise ValueError("output must be a contiguous int32 vector")
        node_count = int(black_to_move.shape[0])
        if node_count <= 0 or node_offsets.shape != (node_count + 1,):
            raise ValueError("node_offsets must have one more entry than black_to_move")
        if int(node_offsets[0]) != 0 or int(node_offsets[-1]) != int(move_hashes.size):
            raise ValueError("node_offsets do not cover move_hashes")
        if output.shape != move_hashes.shape:
            raise ValueError("output shape must match move_hashes")
        status = int(
            self._library.mcts_encode_move_hashes(
                _pointer(move_hashes),
                _pointer(node_offsets),
                _pointer(black_to_move),
                node_count,
                _pointer(output),
            )
        )
        if status != 0:
            raise RuntimeError(f"mcts_encode_move_hashes failed with status {status}")

    def completed_q(
        self,
        visits: np.ndarray,
        value_sums: np.ndarray,
        priors: np.ndarray,
        raw_value: float,
        use_mixed_value: bool,
        q_range_floor: float,
        output: np.ndarray,
    ) -> None:
        status = self._library.mcts_completed_q(
            _pointer(visits), _pointer(value_sums), _pointer(priors), visits.size,
            raw_value, int(use_mixed_value), q_range_floor, _pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"mcts_completed_q failed with status {status}")

    def improved_policy(
        self,
        log_priors: np.ndarray,
        priors: np.ndarray,
        completed_q: np.ndarray,
        max_visit: float,
        c_visit: float,
        c_scale: float,
        output: np.ndarray,
    ) -> None:
        status = self._library.mcts_improved_policy(
            _pointer(log_priors), _pointer(priors), _pointer(completed_q),
            log_priors.size, max_visit, c_visit, c_scale, _pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"mcts_improved_policy failed with status {status}")

    def select_root(
        self,
        log_priors: np.ndarray,
        gumbel: np.ndarray,
        completed_q: np.ndarray,
        initial_visits: np.ndarray,
        visits: np.ndarray,
        total_counts: np.ndarray,
        virtual_losses: np.ndarray,
        considered_visit: float,
        c_visit: float,
        c_scale: float,
        apply_virtual_loss: bool,
    ) -> int:
        return int(self._library.mcts_select_root(
            _pointer(log_priors), _pointer(gumbel), _pointer(completed_q),
            _pointer(initial_visits), _pointer(visits), _pointer(total_counts),
            _pointer(virtual_losses), log_priors.size, considered_visit,
            c_visit, c_scale, int(apply_virtual_loss),
        ))

    def final_action(
        self,
        log_priors: np.ndarray,
        gumbel: np.ndarray,
        completed_q: np.ndarray,
        initial_visits: np.ndarray,
        visits: np.ndarray,
        c_visit: float,
        c_scale: float,
    ) -> int:
        return int(self._library.mcts_final_action(
            _pointer(log_priors), _pointer(gumbel), _pointer(completed_q),
            _pointer(initial_visits), _pointer(visits), log_priors.size,
            c_visit, c_scale,
        ))

    def soften_policy(
        self,
        probabilities: np.ndarray,
        temperature: float,
        output: np.ndarray,
    ) -> None:
        status = self._library.mcts_soften_policy(
            _pointer(probabilities), probabilities.size, temperature, _pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"mcts_soften_policy failed with status {status}")

    def create_forest(
        self,
        roots,
        *,
        c_visit: float,
        c_scale: float,
        q_range_floor: float,
        use_mixed_value: bool,
    ):
        return NativeMCTSForest(
            self,
            roots,
            c_visit=c_visit,
            c_scale=c_scale,
            q_range_floor=q_range_floor,
            use_mixed_value=use_mixed_value,
        )


class NativeMCTSForest:
    """Persistent native tree for traversal, virtual visits and backup."""

    def __init__(
        self,
        kernels: NativeMCTSKernels,
        roots,
        *,
        c_visit: float,
        c_scale: float,
        q_range_floor: float,
        use_mixed_value: bool,
    ):
        self.kernels = kernels
        self._lib = kernels._library
        self._context = self._lib.mcts_forest_create(
            float(c_visit),
            float(c_scale),
            float(q_range_floor),
            int(bool(use_mixed_value)),
            4096,
            65536,
        )
        if not self._context:
            raise RuntimeError("mcts_forest_create failed")
        self.nodes = []
        self._node_ids = {}
        self._native_expanded = set()
        self.root_ids = []
        self._active_roots = {}
        self._prepare_calls = 0
        self._released_since_compact = 0
        try:
            self.root_ids = self._import_roots(list(roots))
        except Exception:
            self.close()
            raise

    @property
    def active(self) -> bool:
        return bool(self._context)

    @property
    def node_count(self) -> int:
        return int(self._lib.mcts_forest_node_count(self._context))

    @property
    def edge_count(self) -> int:
        return int(self._lib.mcts_forest_edge_count(self._context))

    def close(self) -> None:
        context = self._context
        if context:
            self._context = None
            self._lib.mcts_forest_destroy(context)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _raw_value(node) -> float:
        value = getattr(node, "raw_value", None)
        return float("nan") if value is None else float(value)

    def _add_python_node(self, node, parent_id: int, parent_edge: int) -> int:
        existing = self._node_ids.get(id(node))
        if existing is not None:
            return int(existing)
        node_id = int(self._lib.mcts_forest_add_node(
            self._context,
            int(parent_id),
            int(parent_edge),
            int(bool(getattr(node, "expanded", False))),
            self._raw_value(node),
            int(getattr(node, "_root_visit_count", 0) or 0),
            float(getattr(node, "_root_value_sum", 0.0) or 0.0),
            int(getattr(node, "_root_virtual_loss", 0) or 0),
        ))
        if node_id < 0:
            raise RuntimeError(f"mcts_forest_add_node failed with status {node_id}")
        if node_id != len(self.nodes):
            raise RuntimeError(
                f"native node id is not dense: expected {len(self.nodes)}, got {node_id}"
            )
        self.nodes.append(node)
        self._node_ids[id(node)] = node_id
        return node_id

    def _import_roots(self, roots):
        pending = []
        queued_ids = set()
        imported_root_ids = []
        for root in roots:
            root_id = self._add_python_node(root, -1, -1)
            imported_root_ids.append(root_id)
            if root_id not in queued_ids:
                pending.append((root_id, root))
                queued_ids.add(root_id)

        cursor = 0
        while cursor < len(pending):
            node_id, node = pending[cursor]
            cursor += 1
            if (
                node_id in self._native_expanded
                or not bool(getattr(node, "expanded", False))
            ):
                continue
            edges = node.edges
            child_ids = np.full(len(edges.moves), -1, dtype=np.int32)
            for edge_index, child in enumerate(edges.nodes):
                if child is None:
                    continue
                child_id = self._add_python_node(child, node_id, edge_index)
                child_ids[edge_index] = child_id
                if child_id not in queued_ids:
                    pending.append((child_id, child))
                    queued_ids.add(child_id)
            status = self._lib.mcts_forest_set_edges(
                self._context,
                node_id,
                _pointer(np.ascontiguousarray(edges.base_priors, dtype=np.float32)),
                _pointer(np.ascontiguousarray(edges.visit_counts, dtype=np.int32)),
                _pointer(np.ascontiguousarray(edges.value_sums, dtype=np.float32)),
                _pointer(np.ascontiguousarray(edges.virtual_losses, dtype=np.int16)),
                _pointer(child_ids),
                len(edges.moves),
            )
            if status != 0:
                raise RuntimeError(f"mcts_forest_set_edges failed with status {status}")
            self._native_expanded.add(node_id)
        return imported_root_ids

    def prepare_roots(self, roots, tree_keys) -> None:
        """Bind current Python roots without rebuilding the persistent forest."""
        if len(roots) != len(tree_keys):
            raise ValueError("native root/key size mismatch")

        prepared_ids = []
        for root, tree_key in zip(roots, tree_keys):
            previous = self._active_roots.get(tree_key)
            node_id = self._node_ids.get(id(root))
            if node_id is None:
                node_id = self._import_roots([root])[0]
            elif previous is None or int(previous[1]) != int(node_id):
                root_visits = int(getattr(root, "visit_count", 0) or 0)
                root_value_sum = float(getattr(root, "value_sum", 0.0) or 0.0)
                root_virtual_loss = int(getattr(root, "virtual_loss", 0) or 0)
                raw_value = self._raw_value(root)
                # Descendant edge statistics already live in the native tree.
                # Pull only this node's edges before making it the new root.
                self.sync_node(root, sync_root_stats=False)
                status = self._lib.mcts_forest_reroot(
                    self._context,
                    int(node_id),
                    raw_value,
                    root_visits,
                    root_value_sum,
                    root_virtual_loss,
                )
                if status != 0:
                    raise RuntimeError(f"mcts_forest_reroot failed with status {status}")
            else:
                # Refresh the Python root view before dynamic-budget and
                # tree-reuse telemetry read its counters.
                self.sync_node(root)

            self._active_roots[tree_key] = (root, int(node_id))
            prepared_ids.append(int(node_id))

        self.root_ids = prepared_ids
        self._prepare_calls += 1
        if (
            self._released_since_compact >= 8
            or self._prepare_calls % 16 == 0
        ):
            self.compact()
            self.root_ids = [
                int(self._active_roots[tree_key][1])
                for tree_key in tree_keys
            ]

    def release_tree(self, tree_key) -> None:
        if self._active_roots.pop(tree_key, None) is not None:
            self._released_since_compact += 1
        if not self._active_roots and self.node_count:
            self.compact()

    def compact(self) -> None:
        """Drop nodes unreachable from registered game roots entirely in C++."""
        old_count = self.node_count
        if old_count <= 0:
            self.root_ids = []
            self._released_since_compact = 0
            return

        active_items = list(self._active_roots.items())
        active_ids = np.ascontiguousarray(
            [entry[1][1] for entry in active_items],
            dtype=np.int32,
        )
        old_to_new = np.empty(old_count, dtype=np.int32)
        status = self._lib.mcts_forest_compact(
            self._context,
            _pointer(active_ids),
            int(active_ids.size),
            _pointer(old_to_new),
            int(old_to_new.size),
        )
        if status < 0:
            raise RuntimeError(f"mcts_forest_compact failed with status {status}")

        new_count = int(status)
        new_nodes = [None] * new_count
        for old_id, new_id in enumerate(old_to_new):
            if int(new_id) >= 0:
                new_nodes[int(new_id)] = self.nodes[old_id]
        if any(node is None for node in new_nodes):
            raise RuntimeError("native compaction returned a sparse node mapping")

        self.nodes = new_nodes
        self._node_ids = {id(node): node_id for node_id, node in enumerate(new_nodes)}
        self._native_expanded = {
            int(old_to_new[old_id])
            for old_id in self._native_expanded
            if int(old_to_new[old_id]) >= 0
        }
        self._active_roots = {
            tree_key: (root, int(old_to_new[old_node_id]))
            for tree_key, (root, old_node_id) in active_items
            if int(old_to_new[old_node_id]) >= 0
        }
        self.root_ids = [
            int(old_to_new[root_id])
            for root_id in self.root_ids
            if int(old_to_new[root_id]) >= 0
        ]
        self._released_since_compact = 0

    def configure_root_states(self, states) -> None:
        for root_id, state in zip(self.root_ids, states):
            if not state:
                continue
            initial = np.ascontiguousarray(state["initial_visits"], dtype=np.int32)
            gumbel = np.ascontiguousarray(state["gumbel"], dtype=np.float64)
            sequence = np.ascontiguousarray(state.get("sequence", ()), dtype=np.int32)
            if initial.size != gumbel.size:
                raise ValueError("native root state size mismatch")
            root = self.nodes[root_id]
            if initial.size != len(root.edges.moves):
                raise ValueError(
                    "native root state does not match the current root edges"
                )
            status = self._lib.mcts_forest_configure_root(
                self._context,
                int(root_id),
                _pointer(initial),
                _pointer(gumbel),
                int(initial.size),
                _pointer(sequence),
                int(sequence.size),
            )
            if status != 0:
                raise RuntimeError(
                    f"mcts_forest_configure_root failed with status {status}"
                )

    def _ensure_python_leaf(
        self,
        leaf_id: int,
        parent_id: int,
        parent_edge: int,
    ):
        if leaf_id < len(self.nodes):
            node = self.nodes[leaf_id]
            if node is None:
                raise RuntimeError(f"native leaf {leaf_id} has an empty Python mapping")
            return node
        if leaf_id != len(self.nodes):
            raise RuntimeError(
                f"native leaf id is not dense: expected {len(self.nodes)}, got {leaf_id}"
            )
        if parent_id < 0 or parent_id >= len(self.nodes):
            raise RuntimeError(f"native leaf {leaf_id} has invalid parent {parent_id}")
        parent = self.nodes[parent_id]
        child = parent.edges.get_or_create_child(parent, int(parent_edge))
        self.nodes.append(child)
        self._node_ids[id(child)] = leaf_id
        return child

    def select_batch(self, game_indices):
        game_indices = np.ascontiguousarray(game_indices, dtype=np.int32)
        root_ids = np.ascontiguousarray(
            [self.root_ids[int(index)] for index in game_indices],
            dtype=np.int32,
        )
        count = int(root_ids.size)
        selection_ids = np.empty(count, dtype=np.int32)
        leaf_ids = np.empty(count, dtype=np.int32)
        parent_ids = np.empty(count, dtype=np.int32)
        parent_edges = np.empty(count, dtype=np.int32)
        depths = np.empty(count, dtype=np.int32)
        status = self._lib.mcts_forest_select_batch(
            self._context,
            _pointer(root_ids),
            count,
            _pointer(selection_ids),
            _pointer(leaf_ids),
            _pointer(parent_ids),
            _pointer(parent_edges),
            _pointer(depths),
        )
        if status != 0:
            raise RuntimeError(f"mcts_forest_select_batch failed with status {status}")
        leaves = [
            self._ensure_python_leaf(int(leaf), int(parent), int(edge))
            for leaf, parent, edge in zip(leaf_ids, parent_ids, parent_edges)
        ]
        return leaves, selection_ids, depths

    def sync_expansions(self, nodes) -> None:
        seen = set()
        pending_nodes = []
        node_ids = []
        for node in nodes:
            node_id = self._node_ids[id(node)]
            if node_id in seen or node_id in self._native_expanded or not node.expanded:
                continue
            seen.add(node_id)
            pending_nodes.append(node)
            node_ids.append(node_id)
        if not pending_nodes:
            return

        node_ids_array = np.ascontiguousarray(node_ids, dtype=np.int32)
        offsets = np.empty(len(pending_nodes) + 1, dtype=np.int32)
        offsets[0] = 0
        prior_arrays = []
        raw_values = np.empty(len(pending_nodes), dtype=np.float64)
        for index, node in enumerate(pending_nodes):
            priors = np.ascontiguousarray(node.edges.base_priors, dtype=np.float32)
            prior_arrays.append(priors)
            offsets[index + 1] = offsets[index] + int(priors.size)
            raw_values[index] = self._raw_value(node)
        flat_priors = (
            np.concatenate(prior_arrays)
            if int(offsets[-1]) > 0
            else np.empty(0, dtype=np.float32)
        )
        status = self._lib.mcts_forest_expand_batch(
            self._context,
            _pointer(node_ids_array),
            _pointer(offsets),
            _pointer(flat_priors),
            _pointer(raw_values),
            int(node_ids_array.size),
        )
        if status < 0:
            raise RuntimeError(f"mcts_forest_expand_batch failed with status {status}")
        self._native_expanded.update(node_ids)

    def backup_batch(self, selection_ids, values) -> None:
        selection_ids = np.ascontiguousarray(selection_ids, dtype=np.int32)
        values = np.ascontiguousarray(values, dtype=np.float64)
        if selection_ids.size != values.size:
            raise ValueError("native backup batch size mismatch")
        status = self._lib.mcts_forest_backup_batch(
            self._context,
            _pointer(selection_ids),
            _pointer(values),
            int(values.size),
        )
        if status != 0:
            raise RuntimeError(f"mcts_forest_backup_batch failed with status {status}")

    def sync_node(self, node, sync_root_stats=True) -> None:
        node_id = self._node_ids[id(node)]
        edges = node.edges
        root_visits = ctypes.c_int32()
        root_value_sum = ctypes.c_double()
        root_virtual_loss = ctypes.c_int32()
        count = int(self._lib.mcts_forest_export_node(
            self._context,
            int(node_id),
            ctypes.byref(root_visits),
            ctypes.byref(root_value_sum),
            ctypes.byref(root_virtual_loss),
            _pointer(edges.visit_counts),
            _pointer(edges.value_sums),
            _pointer(edges.virtual_losses),
            _pointer(edges.total_counts),
            int(len(edges.moves)),
        ))
        if count < 0:
            raise RuntimeError(f"mcts_forest_export_node failed with status {count}")
        if sync_root_stats:
            node._root_visit_count = int(root_visits.value)
            node._root_value_sum = float(root_value_sum.value)
            node._root_virtual_loss = int(root_virtual_loss.value)
        if count:
            edges.total_count_sum = float(edges.total_counts.sum(dtype=np.float64))
            edges.mark_search_stats_changed()

    def sync_roots(self) -> None:
        for root_id in self.root_ids:
            self.sync_node(self.nodes[root_id])

    def sync_all(self) -> None:
        node_count = int(self._lib.mcts_forest_node_count(self._context))
        edge_count = int(self._lib.mcts_forest_edge_count(self._context))
        if node_count != len(self.nodes):
            raise RuntimeError(
                f"native/Python node mapping mismatch: {node_count} != {len(self.nodes)}"
            )
        node_edge_begins = np.empty(node_count, dtype=np.int32)
        node_edge_counts = np.empty(node_count, dtype=np.int32)
        root_visits = np.empty(node_count, dtype=np.int32)
        root_value_sums = np.empty(node_count, dtype=np.float64)
        root_virtual_losses = np.empty(node_count, dtype=np.int32)
        edge_visits = np.empty(edge_count, dtype=np.int32)
        edge_value_sums = np.empty(edge_count, dtype=np.float32)
        edge_virtual_losses = np.empty(edge_count, dtype=np.int16)
        edge_total_counts = np.empty(edge_count, dtype=np.float32)
        status = self._lib.mcts_forest_export_all(
            self._context,
            _pointer(node_edge_begins),
            _pointer(node_edge_counts),
            _pointer(root_visits),
            _pointer(root_value_sums),
            _pointer(root_virtual_losses),
            _pointer(edge_visits),
            _pointer(edge_value_sums),
            _pointer(edge_virtual_losses),
            _pointer(edge_total_counts),
            node_count,
            edge_count,
        )
        if status != 0:
            raise RuntimeError(f"mcts_forest_export_all failed with status {status}")
        for node_id, node in enumerate(self.nodes):
            node._root_visit_count = int(root_visits[node_id])
            node._root_value_sum = float(root_value_sums[node_id])
            node._root_virtual_loss = int(root_virtual_losses[node_id])
            count = int(node_edge_counts[node_id])
            if count <= 0:
                continue
            begin = int(node_edge_begins[node_id])
            end = begin + count
            edges = node.edges
            if count != len(edges.moves):
                raise RuntimeError(
                    f"native edge count mismatch for node {node_id}: "
                    f"{count} != {len(edges.moves)}"
                )
            np.copyto(edges.visit_counts, edge_visits[begin:end])
            np.copyto(edges.value_sums, edge_value_sums[begin:end])
            np.copyto(edges.virtual_losses, edge_virtual_losses[begin:end])
            np.copyto(edges.total_counts, edge_total_counts[begin:end])
            edges.total_count_sum = float(edges.total_counts.sum(dtype=np.float64))
            edges.mark_search_stats_changed()


def get_native_mcts() -> NativeMCTSKernels | None:
    """Return the process-local native backend, or ``None`` on safe fallback."""
    global _INSTANCE, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _INSTANCE
    if not _enabled_by_environment():
        _LOAD_ATTEMPTED = True
        return None
    with _INSTANCE_LOCK:
        if _LOAD_ATTEMPTED:
            return _INSTANCE
        source = _source_path()
        if not source.is_file():
            _LOAD_ATTEMPTED = True
            return None
        try:
            _INSTANCE = NativeMCTSKernels(_build_library(source, _find_compiler()))
        except Exception:
            # Native acceleration must never make an RL run unavailable. Tests
            # can request a hard failure through CHESS_NATIVE_MCTS_REQUIRED=1.
            required = str(os.environ.get("CHESS_NATIVE_MCTS_REQUIRED", "0")).lower()
            if required in {"1", "true", "yes", "on"}:
                raise
            _INSTANCE = None
        _LOAD_ATTEMPTED = True
        return _INSTANCE
