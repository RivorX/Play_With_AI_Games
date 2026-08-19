from __future__ import annotations

import html.parser
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


_USER_AGENT = "ChessAI-SyzygyDownloader/1.0"
_BASE_URLS = {
    "lichess": "https://tablebase.lichess.ovh/tables/standard",
    "sesse": "https://tablebase.sesse.net/syzygy",
}
_PRESET_DIRS_BY_PROVIDER = {
    "lichess": {
        "wdl_345": ("3-4-5-wdl",),
        "wdl_345_6": ("3-4-5-wdl", "6-wdl"),
    },
    "sesse": {
        "wdl_345": ("3-4-5",),
        "wdl_345_6": ("3-4-5", "6-WDL"),
    },
}


class _DirectoryIndexParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                break


def _default_logger(message: str) -> None:
    print(message)


def get_project_chess_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_syzygy_paths(config: dict, chess_dir: Path | None = None) -> list[Path]:
    chess_dir = Path(chess_dir) if chess_dir is not None else get_project_chess_dir()
    rl_cfg = dict(config.get("reinforcement_learning", {}) or {})
    raw_paths = rl_cfg.get("syzygy_paths", []) or []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]

    resolved = []
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        path_str = str(raw_path).strip()
        if not path_str:
            continue
        candidate = Path(path_str)
        if not candidate.is_absolute():
            # Keep relative Syzygy paths local to the chess/ project directory.
            candidate = chess_dir / candidate
        resolved.append(candidate.resolve())
    return resolved


def get_syzygy_primary_path(config: dict, chess_dir: Path | None = None) -> Path:
    resolved = resolve_syzygy_paths(config, chess_dir=chess_dir)
    if resolved:
        return resolved[0]
    chess_dir = Path(chess_dir) if chess_dir is not None else get_project_chess_dir()
    return (chess_dir / "data" / "syzygy").resolve()


def count_syzygy_wdl_files(paths: list[Path] | tuple[Path, ...]) -> int:
    total = 0
    for base_path in paths:
        try:
            if base_path.exists():
                total += sum(1 for _ in base_path.rglob("*.rtbw"))
        except Exception:
            continue
    return int(total)


def syzygy_piece_counts(
    paths: list[Path] | tuple[Path, ...],
    *,
    extension: str = ".rtbw",
) -> frozenset[int]:
    """Return material sizes for which local Syzygy files are available."""
    suffix = str(extension).lower()
    counts: set[int] = set()
    for base_path in paths:
        try:
            files = base_path.rglob(f"*{suffix}") if base_path.exists() else ()
            for table_path in files:
                material = table_path.stem.upper()
                piece_count = sum(material.count(symbol) for symbol in "KQRBNP")
                if piece_count >= 2:
                    counts.add(int(piece_count))
        except Exception:
            continue
    return frozenset(counts)


def _fetch_directory_listing(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content_type = str(resp.headers.get("Content-Type", ""))
        payload = resp.read()
    if "html" not in content_type.lower() and b"<html" not in payload[:512].lower():
        raise RuntimeError(f"Unexpected directory listing response from {url}")

    parser = _DirectoryIndexParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return list(parser.hrefs)


def _is_sesse_ssl_hostname_error(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        message = str(exc).lower()
        return "hostname mismatch" in message or "certificate verify failed" in message
    if isinstance(exc, urllib.error.URLError):
        message = str(exc.reason).lower()
        return "hostname mismatch" in message or "certificate verify failed" in message
    return False


def _build_preset_file_plan(*, provider: str, preset: str) -> list[tuple[str, str]]:
    base_url = _BASE_URLS.get(provider)
    if not base_url:
        raise ValueError(f"Unsupported Syzygy provider: {provider}")
    provider_presets = _PRESET_DIRS_BY_PROVIDER.get(provider, {})
    dir_names = provider_presets.get(preset)
    if not dir_names:
        raise ValueError(f"Unsupported Syzygy preset '{preset}' for provider '{provider}'")

    plan = []
    seen_files = set()
    for dir_name in dir_names:
        directory_url = f"{base_url}/{dir_name}/"
        hrefs = _fetch_directory_listing(directory_url)
        for href in hrefs:
            joined = urllib.parse.urljoin(directory_url, href)
            file_name = Path(urllib.parse.urlparse(joined).path).name
            lower_name = file_name.lower()
            if not lower_name.endswith(".rtbw"):
                continue
            if file_name in seen_files:
                continue
            seen_files.add(file_name)
            plan.append((joined, file_name))
    plan.sort(key=lambda item: item[1])
    return plan


def _download_file(url: str, destination: Path, *, logger=_default_logger) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, open(temp_path, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    os.replace(temp_path, destination)
    logger(f"  Downloaded {destination.name} ({total / 1024 / 1024:.1f} MB)")
    return int(total)


def ensure_syzygy_tables(
    config: dict,
    *,
    chess_dir: Path | None = None,
    logger=_default_logger,
    force: bool = False,
) -> dict:
    chess_dir = Path(chess_dir) if chess_dir is not None else get_project_chess_dir()
    rl_cfg = dict(config.get("reinforcement_learning", {}) or {})
    if not bool(rl_cfg.get("syzygy_enabled", False)):
        return {"enabled": False, "downloaded_files": 0, "downloaded_bytes": 0, "destination": None}
    if not bool(rl_cfg.get("syzygy_auto_download_enabled", False)):
        return {"enabled": True, "downloaded_files": 0, "downloaded_bytes": 0, "destination": None}

    provider = str(rl_cfg.get("syzygy_auto_download_provider", "lichess") or "lichess").strip().lower()
    preset = str(rl_cfg.get("syzygy_auto_download_preset", "wdl_345") or "wdl_345").strip().lower()
    destination = get_syzygy_primary_path(config, chess_dir=chess_dir)
    destination.mkdir(parents=True, exist_ok=True)

    requested_provider = provider
    try:
        plan = _build_preset_file_plan(provider=provider, preset=preset)
    except Exception as exc:
        if provider == "sesse" and _is_sesse_ssl_hostname_error(exc):
            provider = "lichess"
            logger("Syzygy provider 'sesse' SSL error; retrying with provider 'lichess'.")
            plan = _build_preset_file_plan(provider=provider, preset=preset)
        else:
            raise
    if not plan:
        return {
            "enabled": True,
            "downloaded_files": 0,
            "downloaded_bytes": 0,
            "destination": destination,
            "preset": preset,
            "provider": provider,
            "provider_requested": requested_provider,
        }

    downloaded_files = 0
    downloaded_bytes = 0
    for url, file_name in plan:
        target = destination / file_name
        if target.exists() and not force:
            continue
        downloaded_bytes += _download_file(url, target, logger=logger)
        downloaded_files += 1

    return {
        "enabled": True,
        "downloaded_files": int(downloaded_files),
        "downloaded_bytes": int(downloaded_bytes),
        "destination": destination,
        "preset": preset,
        "provider": provider,
        "provider_requested": requested_provider,
        "total_wdl_files_now": count_syzygy_wdl_files([destination]),
    }


def describe_syzygy_status(config: dict, *, chess_dir: Path | None = None) -> dict:
    chess_dir = Path(chess_dir) if chess_dir is not None else get_project_chess_dir()
    paths = resolve_syzygy_paths(config, chess_dir=chess_dir)
    if not paths:
        paths = [get_syzygy_primary_path(config, chess_dir=chess_dir)]
    return {
        "paths": paths,
        "wdl_files": count_syzygy_wdl_files(paths),
    }
