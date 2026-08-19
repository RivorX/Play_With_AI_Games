"""
List and download PGN datasets from https://database.nikonoel.fr.

The script always resolves paths relative to the chess/ project directory, so
it can be called from any working directory.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

script_dir = Path(__file__).resolve().parent
chess_dir = script_dir.parent
sys.path.insert(0, str(chess_dir))


BASE_URL = "https://database.nikonoel.fr/"
USER_AGENT = "ChessAI-NikonoelDownloader/1.0"
ARCHIVE_LINK_RE = re.compile(
    r"""href=["'](?P<href>[^"']+\.(?:zip|7z|tar(?:\.gz)?|tgz))["']""",
    re.IGNORECASE,
)


def _load_config() -> dict:
    from src.config import load_project_config

    return load_project_config()


def _resolve_data_dir(config: dict) -> Path:
    configured = Path(config["paths"].get("data_dir", "data"))
    if configured.is_absolute():
        data_dir = configured
    else:
        data_dir = chess_dir / configured
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _resolve_archive_cache_dir(data_dir: Path) -> Path:
    cache_dir = data_dir / "_archives" / "nikonoel"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _strip_archive_suffix(filename: str) -> str:
    lower = filename.lower()
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip", ".7z"):
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def _fetch_catalog_html() -> str:
    req = urllib.request.Request(BASE_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _dataset_sort_key(dataset: dict) -> tuple[int, str]:
    stem = dataset["stem"]
    match = re.search(r"(\d{4}-\d{2})$", stem)
    if match:
        year, month = match.group(1).split("-")
        return (int(year) * 100 + int(month), stem)
    return (0, stem)


def fetch_dataset_catalog(data_dir: Path) -> list[dict]:
    html = _fetch_catalog_html()
    seen = {}
    for match in ARCHIVE_LINK_RE.finditer(html):
        href = urllib.parse.urljoin(BASE_URL, match.group("href"))
        archive_name = Path(urllib.parse.urlparse(href).path).name
        if not archive_name:
            continue
        stem = _strip_archive_suffix(archive_name)
        pgn_name = f"{stem}.pgn"
        archive_ext = archive_name.lower()
        if archive_ext.endswith(".tar.gz"):
            archive_type = "tar.gz"
        elif archive_ext.endswith(".tgz"):
            archive_type = "tgz"
        else:
            archive_type = Path(archive_name).suffix.lower().lstrip(".")
        local_pgn = data_dir / pgn_name
        dataset_id = stem.removeprefix("lichess_elite_")
        seen[archive_name] = {
            "id": dataset_id,
            "stem": stem,
            "archive_name": archive_name,
            "archive_type": archive_type,
            "url": href,
            "pgn_name": pgn_name,
            "local_pgn": local_pgn,
            "present": local_pgn.exists(),
        }
    datasets = sorted(seen.values(), key=_dataset_sort_key, reverse=True)
    return datasets


def _print_table(rows: list[dict], title: str, show_index: bool = True) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    headers = ("No", "ID", "Archive", "Type", "PGN", "Status") if show_index else ("ID", "Archive", "Type", "PGN", "Status")
    data_rows = []
    for idx, row in enumerate(rows, start=1):
        values = [
            row["id"],
            row["archive_name"],
            row["archive_type"],
            row["pgn_name"],
            "present" if row["present"] else "missing",
        ]
        if show_index:
            values.insert(0, str(idx))
        data_rows.append(tuple(values))
    widths = [len(h) for h in headers]
    for row in data_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in data_rows:
        print(fmt.format(*row))
    print("=" * 100)


def _prompt_text(prompt: str, default: str = "") -> str:
    if default:
        prompt_text = f"{prompt} (default {default}): "
    else:
        prompt_text = f"{prompt}: "
    try:
        raw = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw or default


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    default_str = "y" if default else "n"
    while True:
        try:
            raw = input(f"{prompt} [y/n] (default {default_str}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            return default
        if raw in {"y", "yes", "t", "true", "1"}:
            return True
        if raw in {"n", "no", "f", "false", "0"}:
            return False
        print("Invalid choice. Enter y or n.")


def _prompt_menu(prompt: str, options: list[str], default_idx: int = 0) -> int:
    default_idx = max(0, min(default_idx, len(options) - 1))
    print(f"\n{prompt}")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}) {option}")
    while True:
        try:
            raw = input(f"Choose [1-{len(options)}] (default {default_idx + 1}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default_idx
        if not raw:
            return default_idx
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if 1 <= value <= len(options):
            return value - 1
        print(f"Invalid choice. Enter a number from 1 to {len(options)}.")


def _parse_limit(raw: str) -> int | None:
    text = raw.strip().lower()
    if not text or text in {"all", "max", "full"}:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError("Limit must be a positive integer or 'all'.") from exc
    if value <= 0:
        raise ValueError("Limit must be > 0.")
    return value


def _print_selection_help() -> None:
    print("\nSelection formats:")
    print("  - Dataset IDs: 2025-09,2025-08")
    print("  - Row numbers from column 'No': 1,2,5-7")
    print("  - Special values: missing, all")


def _parse_index_selection(text: str, max_index: int) -> list[int]:
    indices = set()
    for part in [chunk.strip() for chunk in text.split(",") if chunk.strip()]:
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start_idx = int(start_text)
            end_idx = int(end_text)
            lo = min(start_idx, end_idx)
            hi = max(start_idx, end_idx)
            for idx in range(lo, hi + 1):
                if 1 <= idx <= max_index:
                    indices.add(idx - 1)
        else:
            idx = int(part)
            if 1 <= idx <= max_index:
                indices.add(idx - 1)
    return sorted(indices)


def _interactive_dataset_selection(displayed_rows: list[dict], all_datasets: list[dict]) -> list[dict]:
    _print_selection_help()
    while True:
        raw = _prompt_text(
            "Enter dataset IDs / row numbers / 'missing' / 'all'",
            "missing",
        ).strip()
        lowered = raw.lower()
        if lowered == "all":
            return all_datasets
        if lowered == "missing":
            selected = [dataset for dataset in all_datasets if not dataset["present"]]
            if selected:
                return selected
            print("All listed datasets are already present.")
            continue

        selected = []
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if not tokens:
            print("No selection provided.")
            continue

        try:
            if all(re.fullmatch(r"\d+(?:-\d+)?", token) for token in tokens):
                indices = _parse_index_selection(raw, len(displayed_rows))
                selected = [displayed_rows[idx] for idx in indices]
            else:
                selected = _select_datasets(all_datasets, tokens, select_all=False)
        except Exception:
            selected = []

        if selected:
            return selected
        print("No matching datasets found. Try again.")


def _interactive_options(datasets: list[dict], data_dir: Path) -> tuple[list[dict] | None, bool, bool]:
    limit = None
    while True:
        try:
            limit = _parse_limit(_prompt_text("How many rows to show in the table", "12"))
            break
        except ValueError as exc:
            print(exc)

    table_rows = datasets[:limit] if limit else datasets
    _print_table(table_rows, f"Nikonoel datasets ({len(datasets)} found) -> {data_dir}")

    action_idx = _prompt_menu(
        "What do you want to do?",
        [
            "Show list only and exit",
            "Download selected datasets",
            "Download all missing datasets",
            "Download all datasets",
        ],
        default_idx=0,
    )

    if action_idx == 0:
        return None, False, False

    if action_idx == 1:
        selected = _interactive_dataset_selection(table_rows, datasets)
    elif action_idx == 2:
        selected = [dataset for dataset in datasets if not dataset["present"]]
        if not selected:
            print("\nAll datasets are already present.")
            return None, False, False
    else:
        selected = datasets

    overwrite = _prompt_yes_no("Overwrite existing PGN files", False)
    keep_archives = _prompt_yes_no("Keep downloaded archives in data/_archives/nikonoel", False)
    print(f"\nSelected {len(selected)} dataset(s):")
    _print_table(selected, "Download selection")
    if not _prompt_yes_no("Start download now", False):
        print("Download cancelled.")
        return None, False, False
    return selected, overwrite, keep_archives


def _download_archive(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, open(destination, "wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)


def _extract_pgn_from_zip(archive_path: Path, output_path: Path, overwrite: bool) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        members = [info for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith(".pgn")]
        if not members:
            raise FileNotFoundError(f"No .pgn file found in {archive_path.name}")
        member = members[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            return
        with zf.open(member) as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def _extract_pgn_from_tar(archive_path: Path, output_path: Path, overwrite: bool) -> None:
    with tarfile.open(archive_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".pgn")]
        if not members:
            raise FileNotFoundError(f"No .pgn file found in {archive_path.name}")
        member = members[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            return
        with contextlib.closing(tf.extractfile(member)) as src:
            if src is None:
                raise FileNotFoundError(f"Could not extract {member.name} from {archive_path.name}")
            with open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def _extract_pgn_from_7z(archive_path: Path, output_path: Path, overwrite: bool) -> None:
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if seven_zip is None:
        raise RuntimeError(
            "7z archive detected, but no '7z' or '7za' executable is available on PATH."
        )
    if output_path.exists() and not overwrite:
        return
    with tempfile.TemporaryDirectory(prefix="nikonoel_7z_", dir=str(archive_path.parent)) as tmp_dir:
        extract_dir = Path(tmp_dir)
        subprocess.run(
            [seven_zip, "x", "-y", f"-o{extract_dir}", str(archive_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        pgn_files = sorted(extract_dir.rglob("*.pgn"))
        if not pgn_files:
            raise FileNotFoundError(f"No .pgn file found in {archive_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pgn_files[0]), str(output_path))


def extract_archive(archive_path: Path, output_path: Path, overwrite: bool = False) -> None:
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        _extract_pgn_from_zip(archive_path, output_path, overwrite)
    elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        _extract_pgn_from_tar(archive_path, output_path, overwrite)
    elif lower.endswith(".7z"):
        _extract_pgn_from_7z(archive_path, output_path, overwrite)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive_path.name}")


def _select_datasets(all_datasets: list[dict], selectors: list[str], select_all: bool) -> list[dict]:
    if select_all:
        return all_datasets

    if not selectors:
        return []

    normalized = {selector.strip().lower() for selector in selectors}
    selected = []
    for dataset in all_datasets:
        keys = {
            dataset["id"].lower(),
            dataset["stem"].lower(),
            dataset["archive_name"].lower(),
            dataset["pgn_name"].lower(),
        }
        if keys & normalized:
            selected.append(dataset)
    return selected


def download_datasets(
    datasets: list[dict],
    data_dir: Path,
    keep_archives: bool = False,
    overwrite: bool = False,
) -> None:
    cache_dir = _resolve_archive_cache_dir(data_dir)
    for idx, dataset in enumerate(datasets, start=1):
        output_path = dataset["local_pgn"]
        archive_path = cache_dir / dataset["archive_name"]
        print(f"\n[{idx}/{len(datasets)}] {dataset['archive_name']}")
        if output_path.exists() and not overwrite:
            print(f"  • PGN already present: {output_path}")
            continue

        print(f"  • Download: {dataset['url']}")
        print(f"  • Target PGN: {output_path}")
        _download_archive(dataset["url"], archive_path)
        print(f"  • Archive saved: {archive_path}")

        extract_archive(archive_path, output_path, overwrite=overwrite)
        print(f"  • Extracted PGN: {output_path}")

        if not keep_archives and archive_path.exists():
            archive_path.unlink()
            print(f"  • Removed archive cache: {archive_path.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or download PGN datasets from database.nikonoel.fr"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset IDs or names to download, e.g. 2025-11 or lichess_elite_2025-11",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets in a table and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets returned by the site listing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many rows are shown in the table output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PGN files in chess/data.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives under data/_archives/nikonoel after extraction.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive question mode.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = _load_config()
    data_dir = _resolve_data_dir(config)
    datasets = fetch_dataset_catalog(data_dir)
    if not datasets:
        raise RuntimeError("No downloadable datasets found on database.nikonoel.fr")

    cli_mode = bool(args.list or args.datasets or args.all or args.limit is not None or args.overwrite or args.keep_archives)
    interactive_mode = bool(args.interactive or not cli_mode)

    if interactive_mode:
        selected, overwrite, keep_archives = _interactive_options(datasets, data_dir)
        if not selected:
            return 0
    else:
        table_rows = datasets[: args.limit] if args.limit else datasets
        _print_table(table_rows, f"Nikonoel datasets ({len(datasets)} found) -> {data_dir}")

        if args.list or (not args.datasets and not args.all):
            return 0

        selected = _select_datasets(datasets, args.datasets, args.all)
        if not selected:
            raise RuntimeError("No matching datasets selected. Use --list to inspect available IDs.")
        overwrite = args.overwrite
        keep_archives = args.keep_archives

    print(f"\nDownloading {len(selected)} dataset(s) into {data_dir}...")
    download_datasets(
        selected,
        data_dir=data_dir,
        keep_archives=keep_archives,
        overwrite=overwrite,
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
