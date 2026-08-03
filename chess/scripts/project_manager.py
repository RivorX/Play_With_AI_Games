"""Desktop control center for chess models, training, Elo and diagnostics."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


SCRIPT_DIR = Path(__file__).resolve().parent
CHESS_DIR = SCRIPT_DIR.parent
MODELS_DIR = CHESS_DIR / "models"
LOGS_DIR = CHESS_DIR / "logs"
sys.path.insert(0, str(CHESS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from src.models.catalog import format_elo_summary, load_checkpoint_metadata  # noqa: E402


def _open_path(path: Path):
    if os.name == "nt":
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _safe_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ProjectManager(tk.Tk):
    BG = "#0f172a"
    PANEL = "#172033"
    TEXT = "#e5e7eb"
    MUTED = "#94a3b8"
    ACCENT = "#38bdf8"

    def __init__(self):
        super().__init__()
        self.title("Chess AI Project Manager")
        self.geometry("1480x880")
        self.minsize(1120, 700)
        self.configure(bg=self.BG)
        self.entries = []
        self.entry_by_item = {}
        self._configure_style()
        self._build_ui()
        self.after(80, self.refresh_models)

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.TEXT, fieldbackground=self.PANEL)
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 18), foreground="#f8fafc")
        style.configure("TButton", padding=(10, 7), background="#23314d", foreground=self.TEXT)
        style.map("TButton", background=[("active", "#334766")])
        style.configure(
            "Treeview", background=self.PANEL, fieldbackground=self.PANEL,
            foreground=self.TEXT, rowheight=28, borderwidth=0,
        )
        style.configure("Treeview.Heading", background="#23314d", foreground="#f8fafc")
        style.map("Treeview", background=[("selected", "#075985")])
        style.configure("TNotebook", background=self.BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), background="#1e293b", foreground=self.MUTED)
        style.map("TNotebook.Tab", background=[("selected", self.PANEL)], foreground=[("selected", self.TEXT)])

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(15, 10))
        ttk.Label(header, text="Chess AI Control Center", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header, text="models · Elo history · training · benchmarks · artifacts",
            style="Muted.TLabel",
        ).pack(side="left", padx=16, pady=(7, 0))

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=18, pady=(0, 10))
        actions = (
            ("Refresh", self.refresh_models),
            ("Play", lambda: self.launch("play.py")),
            ("Elo selected", self.run_elo_selected),
            ("Train IL", lambda: self.launch("train_il.py")),
            ("Train RL", lambda: self.launch("train_rl.py")),
            ("Download PGNs", lambda: self.launch("download_nikonoel_pgns.py")),
            ("Syzygy", lambda: self.launch("download_syzygy.py")),
            ("Run cleanup", lambda: self.launch("debug/cleanup_training_runs.py")),
            ("Archive model", self.archive_model),
        )
        for label, command in actions:
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=(0, 7))

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        left = ttk.Frame(paned, style="Panel.TFrame")
        right = ttk.Frame(paned, style="Panel.TFrame")
        paned.add(left, weight=3); paned.add(right, weight=2)

        filter_bar = ttk.Frame(left, style="Panel.TFrame")
        filter_bar.pack(fill="x", padx=10, pady=10)
        ttk.Label(filter_bar, text="Filter", background=self.PANEL).pack(side="left")
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_bar, textvariable=self.filter_var)
        filter_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.filter_var.trace_add("write", lambda *_: self._fill_tree())

        columns = ("folder", "version", "step", "elo", "size", "modified")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Checkpoint")
        for column, title, width in (
            ("folder", "Stage", 60), ("version", "Version", 80), ("step", "Epoch/iter", 80),
            ("elo", "Elo", 180), ("size", "MiB", 65), ("modified", "Modified", 125),
        ):
            self.tree.heading(column, text=title); self.tree.column(column, width=width, anchor="center")
        self.tree.column("#0", width=265)
        self.tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.overview = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.elo_tab = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.artifacts_tab = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.benchmark_tab = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(self.overview, text="Overview")
        self.notebook.add(self.elo_tab, text="Elo history")
        self.notebook.add(self.artifacts_tab, text="Training charts")
        self.notebook.add(self.benchmark_tab, text="Benchmarks")

        self.info = tk.Text(
            self.overview, bg=self.PANEL, fg=self.TEXT, insertbackground=self.TEXT,
            relief="flat", font=("Cascadia Mono", 10), padx=14, pady=14, wrap="word",
        )
        self.info.pack(fill="both", expand=True)

        self._build_elo_tab()
        self._build_artifacts_tab()
        self._build_benchmark_tab()

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, style="Muted.TLabel").pack(fill="x", padx=20, pady=(0, 10))

    def _build_elo_tab(self):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            self.elo_figure = Figure(figsize=(5.4, 4.2), dpi=100, facecolor=self.PANEL)
            self.elo_axis = self.elo_figure.add_subplot(111)
            self.elo_canvas = FigureCanvasTkAgg(self.elo_figure, master=self.elo_tab)
            self.elo_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        except Exception as exc:
            self.elo_axis = None
            ttk.Label(self.elo_tab, text=f"Matplotlib unavailable: {exc}").pack(padx=15, pady=15)

    def _build_artifacts_tab(self):
        self.artifact_list = tk.Listbox(
            self.artifacts_tab, bg=self.PANEL, fg=self.TEXT, selectbackground="#075985",
            relief="flat", font=("Segoe UI", 10),
        )
        self.artifact_list.pack(fill="both", expand=True, padx=10, pady=10)
        self.artifact_list.bind("<Double-Button-1>", lambda _event: self.open_artifact())
        ttk.Button(self.artifacts_tab, text="Open selected chart", command=self.open_artifact).pack(pady=(0, 10))
        self.current_artifacts = []

    def _build_benchmark_tab(self):
        controls = ttk.Frame(self.benchmark_tab, style="Panel.TFrame")
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Button(
            controls, text="Native MCTS microbenchmark",
            command=lambda: self.run_captured("debug/benchmark_native_mcts.py"),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            controls, text="Self-play stream A/B",
            command=lambda: self.launch("debug/benchmark_selfplay_stream.py"),
        ).pack(side="left")
        self.benchmark_output = tk.Text(
            self.benchmark_tab, bg="#0b1120", fg="#d1fae5", relief="flat",
            font=("Cascadia Mono", 9), padx=12, pady=12,
        )
        self.benchmark_output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.benchmark_output.insert("end", "Select a benchmark. Results are also saved under logs/csv.\n")

    def refresh_models(self):
        self.status.set("Scanning checkpoints…")
        self.update_idletasks()
        paths = sorted(MODELS_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        paths = [path for path in paths if ".trash" not in path.parts]
        self.entries = [load_checkpoint_metadata(path, MODELS_DIR) for path in paths]
        self._fill_tree()
        self.status.set(f"Loaded {len(self.entries)} checkpoints")

    def _fill_tree(self):
        query = self.filter_var.get().strip().lower() if hasattr(self, "filter_var") else ""
        self.tree.delete(*self.tree.get_children()); self.entry_by_item.clear()
        for entry in self.entries:
            haystack = " ".join(str(entry.get(key) or "") for key in ("model_name", "folder", "version")).lower()
            if query and query not in haystack:
                continue
            item = self.tree.insert(
                "", "end", text=entry["model_name"],
                values=(
                    entry.get("folder") or "-", entry.get("version") or "-",
                    entry.get("epoch") if entry.get("epoch") is not None else "-",
                    format_elo_summary(entry), f"{entry.get('size_mb', 0):.1f}", entry.get("modified") or "-",
                ),
            )
            self.entry_by_item[item] = entry

    def selected_entry(self):
        selection = self.tree.selection()
        return self.entry_by_item.get(selection[0]) if selection else None

    def _on_select(self, _event=None):
        entry = self.selected_entry()
        if not entry:
            return
        summary = self._training_summary(entry)
        lines = [
            entry["model_name"], "=" * min(72, len(entry["model_name"])),
            f"Path:       {entry['path']}",
            f"Stage:      {entry.get('folder')}",
            f"Version:    {entry.get('version') or 'n/a'}",
            f"Epoch/iter: {entry.get('epoch') if entry.get('epoch') is not None else 'n/a'}",
            f"Elo:        {format_elo_summary(entry)}",
            f"Loss:       {entry.get('val_loss') if entry.get('val_loss') is not None else 'n/a'}",
            f"Policy:     top1={entry.get('top1')}, loss={entry.get('policy_loss')}",
            f"Value loss: {entry.get('value_loss')}",
            f"Optimizer:  {'yes' if entry.get('optimizer') else 'no'}",
            f"SWA:        {'yes' if entry.get('swa') else 'no'}",
            "", "Training artifacts:",
            f"  matching runs: {summary['runs']}",
            f"  IL max epoch: {summary['il_steps']}",
            f"  RL max iteration: {summary['rl_steps']}",
            f"  replay positions logged: {summary['positions']:,}",
        ]
        if entry.get("error"):
            lines.extend(("", f"ERROR: {entry['error']}"))
        self.info.delete("1.0", "end"); self.info.insert("1.0", "\n".join(lines))
        self._plot_elo(entry)
        self._load_artifacts(entry)

    def _matching_main_csvs(self, entry):
        version = str(entry.get("version") or "").lower()
        stem = str(entry.get("model_name") or "").lower().removesuffix(".pt")
        paths = []
        for path in (LOGS_DIR / "csv").glob("*.csv"):
            name = path.stem.lower()
            if name.endswith(("_performance", "_data_quality", "_training_profile")):
                continue
            if (version and version in name) or (stem and stem in name):
                paths.append(path)
        return paths

    def _training_summary(self, entry):
        result = {"runs": 0, "il_steps": 0, "rl_steps": 0, "positions": 0}
        for path in self._matching_main_csvs(entry):
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                    rows = list(csv.DictReader(line.replace("\x00", "") for line in handle if not line.startswith("#")))
            except OSError:
                continue
            result["runs"] += 1
            for row in rows:
                if row.get("epoch"):
                    result["il_steps"] = max(result["il_steps"], int(_safe_number(row.get("epoch"))))
                if row.get("iteration"):
                    result["rl_steps"] = max(result["rl_steps"], int(_safe_number(row.get("iteration"))))
                result["positions"] += int(_safe_number(row.get("positions_added")))
        return result

    def _plot_elo(self, entry):
        if self.elo_axis is None:
            return
        axis = self.elo_axis; axis.clear(); axis.set_facecolor(self.PANEL)
        mcts_by_simulations = dict(entry.get("elo_mcts_by_simulations") or {})
        if not mcts_by_simulations and entry.get("elo_mcts") is not None:
            simulations = int(entry.get("elo_mcts_simulations") or 0)
            if simulations > 0:
                mcts_by_simulations[simulations] = {"elo": float(entry["elo_mcts"])}
        mcts_points = [
            (int(simulations), float(info["elo"]))
            for simulations, info in mcts_by_simulations.items()
            if isinstance(info, dict) and info.get("elo") is not None
        ]
        mcts_points.sort()
        palette = ("#f59e0b", "#7c3aed", "#ef4444", "#10b981", "#ec4899")
        if mcts_points:
            xs, ys = zip(*mcts_points)
            axis.plot(xs, ys, lw=1.5, color="#64748b", alpha=.65)
            for index, (simulations, elo) in enumerate(mcts_points):
                axis.scatter(
                    [simulations],
                    [elo],
                    s=42,
                    color=palette[index % len(palette)],
                    label=f"MCTS @{simulations}",
                    zorder=3,
                )
        if entry.get("elo_nn") is not None:
            axis.axhline(
                float(entry["elo_nn"]),
                color="#38bdf8",
                linestyle="--",
                linewidth=1.8,
                label=f"Raw NN {int(round(float(entry['elo_nn'])))}",
            )
        axis.set_title("Checkpoint Elo by search budget", color=self.TEXT, pad=12)
        axis.set_xlabel("MCTS simulations", color=self.MUTED); axis.set_ylabel("Elo", color=self.MUTED)
        axis.tick_params(colors=self.MUTED); axis.grid(alpha=.18, color="#94a3b8")
        for spine in axis.spines.values(): spine.set_color("#475569")
        if entry.get("elo_nn") is not None or mcts_points:
            legend = axis.legend(facecolor=self.PANEL, edgecolor="#475569")
            for text in legend.get_texts(): text.set_color(self.TEXT)
        else:
            axis.text(.5, .5, "No Elo profile in checkpoint", ha="center", va="center", color=self.MUTED, transform=axis.transAxes)
        self.elo_figure.tight_layout(); self.elo_canvas.draw_idle()

    def _load_artifacts(self, entry):
        version = str(entry.get("version") or "").lower()
        stem = str(entry.get("model_name") or "").lower().removesuffix(".pt")
        self.current_artifacts = [
            path for path in LOGS_DIR.rglob("*.png")
            if (version and version in path.name.lower()) or (stem and stem in path.name.lower())
        ]
        self.current_artifacts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self.artifact_list.delete(0, "end")
        for path in self.current_artifacts:
            self.artifact_list.insert("end", str(path.relative_to(LOGS_DIR)))

    def open_artifact(self):
        selection = self.artifact_list.curselection()
        if selection:
            _open_path(self.current_artifacts[selection[0]])

    def launch(self, relative_script: str, extra_args=()):
        command = [sys.executable, str(SCRIPT_DIR / relative_script), *map(str, extra_args)]
        kwargs = {"cwd": str(CHESS_DIR)}
        if os.name == "nt": kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(command, **kwargs)
        self.status.set(f"Launched: {Path(relative_script).name}")

    def run_elo_selected(self):
        entry = self.selected_entry()
        if not entry:
            messagebox.showinfo("Elo", "Select a checkpoint first.")
            return
        self.launch("eval_elo.py", ("--model", entry["path"]))

    def archive_model(self):
        entry = self.selected_entry()
        if not entry:
            messagebox.showinfo("Archive", "Select a checkpoint first.")
            return
        source = Path(entry["path"]).resolve()
        if MODELS_DIR.resolve() not in source.parents:
            messagebox.showerror("Archive", "Selected file is outside the models directory.")
            return
        if not messagebox.askyesno("Archive model", f"Move {source.name} to models/.trash?\nThis is recoverable."):
            return
        relative = source.relative_to(MODELS_DIR.resolve())
        target = MODELS_DIR / ".trash" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S") / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        self.status.set(f"Archived to {target}"); self.refresh_models()

    def run_captured(self, relative_script: str):
        self.benchmark_output.delete("1.0", "end")
        self.benchmark_output.insert("end", f"Running {relative_script}…\n")
        self.status.set("Benchmark running…")

        def worker():
            completed = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / relative_script)], cwd=CHESS_DIR,
                capture_output=True, text=True, errors="replace",
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            self.after(0, lambda: self._show_benchmark_result(relative_script, completed.returncode, output))
        threading.Thread(target=worker, daemon=True).start()

    def _show_benchmark_result(self, name, returncode, output):
        self.benchmark_output.delete("1.0", "end")
        self.benchmark_output.insert("end", output or "(no output)")
        self.benchmark_output.see("end")
        self.status.set(f"{Path(name).name} finished with code {returncode}")


if __name__ == "__main__":
    ProjectManager().mainloop()
