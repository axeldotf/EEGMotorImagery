"""
EDF+ viewer built with Tkinter and pyEDFlib.

- Pick an EDF/EDF+ file via a file dialog
- View the general header (pretty JSON), per-signal metadata, and EDF+ annotations
- Plot selected channels over a chosen interval (if matplotlib is installed)

Run: python edf_viewer.py
Requirements: pyedflib (matplotlib optional for plotting)
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np
import pyedflib

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # plotting is optional
    plt = None


class EDFViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EDF+ Viewer")
        self.geometry("1000x700")

        # --- Top toolbar ---
        toolbar = tk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_open = tk.Button(toolbar, text="Open EDF+", command=self.open_file)
        btn_open.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_plot = tk.Button(toolbar, text="Plot selected channels", command=self.plot_selected)
        self.btn_plot.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_plot.config(state=tk.DISABLED)

        # --- Notebook with tabs ---
        self.nb = ttk.Notebook(self)
        self.nb.pack(expand=True, fill=tk.BOTH)

        # Header tab
        self.txt_header = tk.Text(self.nb, wrap=tk.NONE)
        self.txt_header.configure(font=("Courier", 10))
        self.nb.add(self.txt_header, text="Header")

        # Signals tab (treeview)
        sig_frame = tk.Frame(self.nb)
        self.nb.add(sig_frame, text="Signals")
        cols = (
            "#", "label", "unit", "fs", "n_samples",
            "phys_min", "phys_max", "dig_min", "dig_max", "prefilter"
        )
        self.tree = ttk.Treeview(sig_frame, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90, anchor=tk.CENTER)
        self.tree.column("label", width=130, anchor=tk.W)
        self.tree.column("prefilter", width=200, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(sig_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        # Annotations tab
        ann_frame = tk.Frame(self.nb)
        self.nb.add(ann_frame, text="Annotations")
        ann_cols = ("onset_s", "duration_s", "text")
        self.ann_tree = ttk.Treeview(ann_frame, columns=ann_cols, show="headings")
        for c in ann_cols:
            self.ann_tree.heading(c, text=c)
            self.ann_tree.column(c, width=120 if c != "text" else 600, anchor=tk.W)
        self.ann_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ann_vsb = ttk.Scrollbar(ann_frame, orient="vertical", command=self.ann_tree.yview)
        ann_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ann_tree.configure(yscrollcommand=ann_vsb.set)

        # Internal state
        self.current_path: Path | None = None
        self.reader: pyedflib.EdfReader | None = None
        self.labels: list[str] = []

    # --------------------
    # File handling
    # --------------------
    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an EDF/EDF+ file",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.load_edf(Path(path))
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open file:\n{e}")

    def load_edf(self, path: Path) -> None:
        # Close previous
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None

        reader = pyedflib.EdfReader(str(path))
        self.reader = reader
        self.current_path = path

        # --- Header ---
        header = reader.getHeader()  # dict
        header_pretty = json.dumps(header, indent=2, default=str)
        self.txt_header.delete("1.0", tk.END)
        self.txt_header.insert(tk.END, header_pretty)
        self.txt_header.see("1.0")

        # --- Signals ---
        for row in self.tree.get_children():
            self.tree.delete(row)

        self.labels = reader.getSignalLabels()
        n_signals = reader.signals_in_file
        units = [reader.getPhysicalDimension(i) for i in range(n_signals)]
        fs = [reader.getSampleFrequency(i) for i in range(n_signals)]
        n_samples = reader.getNSamples()
        phys_min = [reader.getPhysicalMinimum(i) for i in range(n_signals)]
        phys_max = [reader.getPhysicalMaximum(i) for i in range(n_signals)]
        dig_min = [reader.getDigitalMinimum(i) for i in range(n_signals)]
        dig_max = [reader.getDigitalMaximum(i) for i in range(n_signals)]
        prefilter = [reader.getPrefilter(i) for i in range(n_signals)]

        for i, lab in enumerate(self.labels):
            self.tree.insert(
                "",
                tk.END,
                values=(
                    i,
                    lab.strip(),
                    (units[i] or "").strip(),
                    fs[i],
                    n_samples[i],
                    phys_min[i],
                    phys_max[i],
                    dig_min[i],
                    dig_max[i],
                    (prefilter[i] or "").strip(),
                ),
            )

        # --- Annotations ---
        for row in self.ann_tree.get_children():
            self.ann_tree.delete(row)
        try:
            onsets, durations, texts = reader.readAnnotations()
        except Exception:
            onsets, durations, texts = [], [], []
        for o, d, t in zip(onsets, durations, texts):
            self.ann_tree.insert("", tk.END, values=(o, d, t))

        # Enable/disable plot button
        self.btn_plot.config(state=tk.NORMAL if plt is not None else tk.DISABLED)

        self.nb.select(0)  # switch to header tab
        self.title(f"EDF+ Viewer - {path.name}")

    # --------------------
    # Plotting
    # --------------------
    def plot_selected(self) -> None:
        if plt is None:
            messagebox.showinfo("Plot not available", "matplotlib not installed")
            return
        if self.reader is None:
            return

        sels = self.tree.selection()
        if not sels:
            messagebox.showinfo("No channel selected", "Select one or more channels in the Signals tab")
            return
        idxs = [int(self.tree.item(s, "values")[0]) for s in sels]

        # Ask how many seconds to plot
        sec = simple_float_dialog(self, "Plot duration (s)", "How many seconds do you want to plot?", default="10")
        if sec is None:
            return
        try:
            sec = float(sec)
        except ValueError:
            messagebox.showerror("Error", "Invalid value for duration")
            return

        start_sec = simple_float_dialog(self, "Start time (s)", "Start second (0 = file start)", default="0")
        if start_sec is None:
            return
        try:
            start_sec = float(start_sec)
        except ValueError:
            messagebox.showerror("Error", "Invalid value for start second")
            return

        r = self.reader
        plt.figure()
        for ch in idxs:
            fs = r.getSampleFrequency(ch)
            start_sample = int(start_sec * fs)
            n_samp = int(sec * fs)
            data = r.readSignal(ch, start=start_sample, n=n_samp)
            t = np.arange(len(data)) / fs + start_sec
            # Safe label retrieval
            label = self.labels[ch] if 0 <= ch < len(self.labels) else f"ch{ch}"
            plt.plot(t, data, label=label)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (physical units)")
        plt.legend()
        plt.title(Path(r.file_name).name if hasattr(r, "file_name") else str(self.current_path))
        plt.tight_layout()
        plt.show()


def simple_float_dialog(master: tk.Tk, title: str, prompt: str, default: str = "") -> str | None:
    """A tiny modal dialog to ask the user for a number. Returns None on cancel."""
    win = tk.Toplevel(master)
    win.title(title)
    win.resizable(False, False)
    win.transient(master)
    win.grab_set()

    tk.Label(win, text=prompt).pack(padx=10, pady=(10, 0))
    var = tk.StringVar(value=default)
    entry = tk.Entry(win, textvariable=var)
    entry.pack(padx=10, pady=5)
    entry.focus_set()

    result: dict[str, str | None] = {"value": None}

    def ok() -> None:
        result["value"] = var.get()
        win.destroy()

    def cancel() -> None:
        win.destroy()

    btns = tk.Frame(win)
    btns.pack(pady=10)
    tk.Button(btns, text="OK", width=8, command=ok).pack(side=tk.LEFT, padx=5)
    tk.Button(btns, text="Cancel", width=8, command=cancel).pack(side=tk.LEFT, padx=5)

    win.bind("<Return>", lambda _e: ok())
    win.bind("<Escape>", lambda _e: cancel())

    master.wait_window(win)
    return result["value"]


if __name__ == "__main__":
    app = EDFViewer()
    app.mainloop()
