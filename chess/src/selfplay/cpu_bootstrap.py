"""Load the CPU-only torch overlay before importing a spawned worker target.

Keep this module limited to the standard library: multiprocessing imports the
target module while rebuilding its Process object, before the target is called.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def resolve_cpu_worker_torch_mode(runtime_mode):
    """Resolve the setting without treating ``"auto"`` as Python truthiness."""
    if isinstance(runtime_mode, bool):
        return runtime_mode, False
    mode_text = str(runtime_mode or "auto").strip().lower()
    if mode_text in {"auto", "automatic"}:
        return True, True
    if mode_text in {"true", "on", "enabled", "1"}:
        return True, False
    if mode_text in {"false", "off", "disabled", "0", "none"}:
        return False, False
    raise ValueError(
        "central_inference.cpu_worker_torch_enabled must be 'auto', true, or false."
    )


def validate_cpu_worker_torch_runtime(
    runtime_info,
    *,
    expected_torch_version: str,
    expected_base_version: str,
    python_version: str,
    platform: str,
    wheel_tag: str,
) -> None:
    """Fail closed if the isolated wheel marker does not match this runtime."""
    if not isinstance(runtime_info, dict):
        raise RuntimeError("CPU worker torch runtime marker must be a JSON object.")
    expected_fields = {
        "torch_version": str(expected_torch_version),
        "torch_base_version": str(expected_base_version),
        "python_version": str(python_version),
        "platform": str(platform),
        "wheel_tag": str(wheel_tag),
    }
    for key, expected in expected_fields.items():
        found = str(runtime_info.get(key, ""))
        if found != expected:
            raise RuntimeError(
                f"CPU worker torch runtime {key} mismatch: "
                f"expected {expected}, found {found}."
            )
    wheel_base_version = str(expected_torch_version).split("+", 1)[0]
    if wheel_base_version != str(expected_base_version):
        raise RuntimeError(
            "CPU worker torch wheel version and expected base version disagree: "
            f"{wheel_base_version} != {expected_base_version}."
        )


def cpu_worker_torch_supported() -> bool:
    """Whether the isolated wheel matches this project's Windows wheel target."""
    return (
        os.name == "nt"
        and sys.maxsize > 2**32
        and platform.machine().lower() in {"amd64", "x86_64"}
    )


def _read_cpu_torch_wheel(overlay: Path) -> dict[str, str]:
    """Read installed wheel identity and verify required CPU package files."""
    torch_package = overlay / "torch" / "__init__.py"
    cpu_dll = overlay / "torch" / "lib" / "torch_cpu.dll"
    if not torch_package.is_file() or not cpu_dll.is_file():
        raise RuntimeError(
            f"CPU torch overlay is incomplete under {overlay}; expected "
            "torch/__init__.py and torch/lib/torch_cpu.dll."
        )

    dist_info_dirs = sorted(overlay.glob("torch-*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError(
            f"Expected one torch-*.dist-info directory under {overlay}, "
            f"found {len(dist_info_dirs)}."
        )
    dist_info = dist_info_dirs[0]
    metadata_file = dist_info / "METADATA"
    wheel_file = dist_info / "WHEEL"
    if not metadata_file.is_file() or not wheel_file.is_file():
        raise RuntimeError(f"CPU torch wheel metadata is incomplete under {dist_info}.")

    try:
        metadata_lines = metadata_file.read_text(encoding="utf-8").splitlines()
        wheel_lines = wheel_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"CPU torch wheel metadata cannot be read under {dist_info}.") from exc
    version_lines = [
        line.partition(":")[2].strip()
        for line in metadata_lines
        if line.startswith("Version:")
    ]
    wheel_tags = [
        line.partition(":")[2].strip()
        for line in wheel_lines
        if line.startswith("Tag:")
    ]
    if len(version_lines) != 1 or not wheel_tags:
        raise RuntimeError(f"CPU torch wheel identity is missing from {dist_info}.")
    return {
        "torch_version": version_lines[0],
        "torch_base_version": version_lines[0].split("+", 1)[0],
        "wheel_tag": wheel_tags[0],
        "wheel_tags": wheel_tags,
    }


def validate_cpu_worker_torch_overlay(
    overlay_path,
    *,
    expected_torch_version: str,
    expected_base_version: str,
    python_version: str,
    platform_name: str,
    wheel_tag: str,
) -> None:
    """Validate both the marker and the actual installed CPU wheel files."""
    overlay = Path(overlay_path).resolve()
    actual_wheel = _read_cpu_torch_wheel(overlay)
    runtime_file = overlay / "cpu_torch_runtime.json"
    try:
        runtime_info = json.loads(runtime_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"CPU torch overlay runtime marker is missing or invalid: {runtime_file}."
        ) from exc

    validate_cpu_worker_torch_runtime(
        runtime_info,
        expected_torch_version=expected_torch_version,
        expected_base_version=expected_base_version,
        python_version=python_version,
        platform=platform_name,
        wheel_tag=wheel_tag,
    )
    for key in ("torch_version", "torch_base_version"):
        if actual_wheel[key] != str(runtime_info.get(key, "")):
            raise RuntimeError(
                f"CPU torch overlay {key} does not match its marker: "
                f"wheel={actual_wheel[key]}, marker={runtime_info.get(key)}."
            )
    if str(runtime_info.get("wheel_tag", "")) not in actual_wheel["wheel_tags"]:
        raise RuntimeError(
            "CPU torch overlay wheel tag does not match its marker: "
            f"wheel tags={actual_wheel['wheel_tags']}, "
            f"marker={runtime_info.get('wheel_tag')}."
        )


class _CpuTorchInstallLock:
    """Small bounded Windows byte-range lock for concurrent RL startups."""

    def __init__(self, path, timeout_s):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.fd = None
        self.locked = False

    def __enter__(self):
        if not cpu_worker_torch_supported():
            raise RuntimeError("Automatic CPU torch setup requires Windows x64.")
        import msvcrt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        if os.fstat(self.fd).st_size == 0:
            os.write(self.fd, b"\0")
        deadline = time.monotonic() + self.timeout_s
        while True:
            os.lseek(self.fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                self.locked = True
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(self.fd)
                    self.fd = None
                    raise TimeoutError(
                        f"Timed out waiting for CPU worker torch setup lock: {self.path}"
                    )
                time.sleep(0.25)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is None:
            return False
        try:
            if self.locked:
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(self.fd)
            self.fd = None
            self.locked = False
        return False


def ensure_cpu_worker_torch_overlay(
    project_root,
    parent_torch_version: str,
    *,
    overlay_override=None,
    python_executable=None,
    pip_runner=None,
    pip_timeout_s: float = 300.0,
    lock_timeout_s: float = 360.0,
) -> Path:
    """Reuse or install a matching CPU torch wheel in the project cache.

    The CUDA-enabled environment is only the interpreter and dependency source;
    pip writes into a unique staging directory and publishes the validated
    wheel atomically under ``.cache/rl-cpu-workers``.
    """
    if not cpu_worker_torch_supported():
        raise RuntimeError("Automatic CPU worker torch setup requires Windows x64.")

    project_root = Path(project_root).resolve()
    runtime_root = project_root / ".cache" / "rl-cpu-workers"
    base_version = str(parent_torch_version).split("+", 1)[0].strip()
    if not base_version:
        raise RuntimeError(f"Could not resolve the parent PyTorch version: {parent_torch_version!r}.")
    cpu_version = f"{base_version}+cpu"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_tag = f"{python_tag}-{python_tag}-win_amd64"
    versioned_target = (
        runtime_root
        / f"torch-{cpu_version}-{wheel_tag}"
        / "site-packages"
    )
    legacy_target = runtime_root / "site-packages"

    candidates = []
    if overlay_override:
        override = Path(os.path.expandvars(str(overlay_override))).expanduser()
        if not override.is_absolute():
            override = project_root / override
        candidates.append(override.resolve())
    candidates.extend((legacy_target, versioned_target))
    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))

    lock_path = runtime_root / ".install.lock"
    try:
        with _CpuTorchInstallLock(lock_path, lock_timeout_s):
            invalid_candidates = {}
            for candidate in unique_candidates:
                if not candidate.exists():
                    continue
                try:
                    validate_cpu_worker_torch_overlay(
                        candidate,
                        expected_torch_version=cpu_version,
                        expected_base_version=base_version,
                        python_version=python_version,
                        platform_name=sys.platform,
                        wheel_tag=wheel_tag,
                    )
                    return candidate
                except (OSError, RuntimeError, UnicodeError) as exc:
                    invalid_candidates[candidate] = str(exc)

            if versioned_target.exists():
                raise RuntimeError(
                    "The versioned CPU worker torch cache exists but is invalid; "
                    "it was left untouched: "
                    f"{versioned_target}: "
                    f"{invalid_candidates.get(versioned_target, 'validation failed')}."
                )

            runtime_root.mkdir(parents=True, exist_ok=True)
            staging = runtime_root / (
                f"site-packages.staging-{os.getpid()}-{uuid.uuid4().hex}"
            )
            installer = pip_runner or subprocess.run
            command = [
                str(python_executable or sys.executable),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--only-binary=:all:",
                "--retries",
                "1",
                "--timeout",
                "20",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "--target",
                str(staging),
                f"torch=={cpu_version}",
            ]
            try:
                print(
                    "Preparing the isolated CPU-only PyTorch runtime "
                    f"({cpu_version}) for central-inference self-play. "
                    "This first download is limited to 300 seconds; the existing "
                    "CUDA environment will not be modified.",
                    flush=True,
                )
                result = installer(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=float(pip_timeout_s),
                    check=False,
                )
                if int(result.returncode) != 0:
                    detail = (result.stderr or result.stdout or "").strip()
                    detail = "\n".join(detail.splitlines()[-8:])
                    raise RuntimeError(
                        "CPU torch wheel installation failed"
                        + (f":\n{detail}" if detail else ".")
                    )

                actual_wheel = _read_cpu_torch_wheel(staging)
                if (
                    actual_wheel["torch_version"] != cpu_version
                    or actual_wheel["torch_base_version"] != base_version
                    or wheel_tag not in actual_wheel["wheel_tags"]
                ):
                    raise RuntimeError(
                        "Downloaded CPU torch wheel does not match the expected "
                        f"version/ABI: expected {cpu_version} / {wheel_tag}, "
                        f"found {actual_wheel}."
                    )

                marker = {
                    "torch_version": cpu_version,
                    "torch_base_version": base_version,
                    "python_version": python_version,
                    "platform": sys.platform,
                    "wheel_tag": wheel_tag,
                }
                (staging / "cpu_torch_runtime.json").write_text(
                    json.dumps(marker, indent=2) + "\n",
                    encoding="utf-8",
                )
                validate_cpu_worker_torch_overlay(
                    staging,
                    expected_torch_version=cpu_version,
                    expected_base_version=base_version,
                    python_version=python_version,
                    platform_name=sys.platform,
                    wheel_tag=wheel_tag,
                )
                versioned_target.parent.mkdir(parents=True, exist_ok=True)
                if versioned_target.exists():
                    raise RuntimeError(
                        "The versioned CPU worker torch cache appeared during setup; "
                        f"it was left untouched: {versioned_target}."
                    )
                os.rename(staging, versioned_target)
                validate_cpu_worker_torch_overlay(
                    versioned_target,
                    expected_torch_version=cpu_version,
                    expected_base_version=base_version,
                    python_version=python_version,
                    platform_name=sys.platform,
                    wheel_tag=wheel_tag,
                )
                return versioned_target.resolve()
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"CPU torch wheel installation exceeded {pip_timeout_s:g} seconds. "
                    "The CUDA environment was not changed; retry when the CPU wheel index is reachable."
                ) from exc
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "Could not prepare the isolated CPU-only PyTorch runtime for "
                    "central-inference self-play. The CUDA environment was not changed. "
                    f"Details: {exc}"
                ) from exc
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
    except TimeoutError as exc:
        raise RuntimeError(
            "Another RL launch is still preparing the isolated CPU torch runtime; "
            f"wait and retry. Details: {exc}"
        ) from exc


def bootstrap_cpu_torch_worker(
    overlay_path: str,
    expected_torch_version: str,
    target_module: str,
    target_name: str,
    target_args: tuple,
) -> None:
    """Load the pinned CPU torch package, then import and call the worker.

    ``target_args`` must contain only standard multiprocessing primitives and
    serializable configuration. Keeping the callable as two strings prevents
    multiprocessing from importing the target (and CUDA torch) before this
    bootstrap runs.
    """
    if "torch" in sys.modules:
        raise RuntimeError(
            "CPU worker bootstrap ran after torch was imported; check the "
            "spawn entrypoint import guard."
        )

    if os.name != "nt":
        raise RuntimeError("The isolated CPU worker torch overlay is Windows-only.")

    overlay = Path(overlay_path).resolve()
    expected_base_version = str(expected_torch_version).split("+", 1)[0]
    expected_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    expected_python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    expected_wheel_tag = f"{expected_python_tag}-{expected_python_tag}-win_amd64"
    validate_cpu_worker_torch_overlay(
        overlay,
        expected_torch_version=expected_torch_version,
        expected_base_version=expected_base_version,
        python_version=expected_python_version,
        platform_name=sys.platform,
        wheel_tag=expected_wheel_tag,
    )
    if (target_module, target_name) != (
        "src.selfplay.workers",
        "persistent_selfplay_worker",
    ):
        raise ValueError(
            f"Unsupported CPU worker target {target_module}.{target_name}."
        )

    # These workers own MCTS/boards only; the central server owns the GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    sys.path.insert(0, str(overlay))
    torch = importlib.import_module("torch")
    loaded_from = Path(torch.__file__).resolve()
    if not loaded_from.is_relative_to(overlay):
        raise RuntimeError(
            "CPU worker loaded torch outside its overlay: "
            f"{loaded_from} (expected under {overlay})."
        )
    if str(torch.__version__) != str(expected_torch_version):
        raise RuntimeError(
            f"CPU worker imported torch {torch.__version__}; "
            f"expected {expected_torch_version}."
        )
    if getattr(torch.version, "cuda", None) is not None:
        raise RuntimeError(
            f"CPU worker overlay unexpectedly contains CUDA torch {torch.__version__}."
        )

    # Register tensor reducers before the CPU worker consumes the existing
    # torch.multiprocessing queues and returns CPU tensor batches to the parent.
    importlib.import_module("torch.multiprocessing")

    target = getattr(importlib.import_module(target_module), target_name)
    target(*target_args)
