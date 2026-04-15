from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk #tkinter is a library(module), that contains many classes
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from python_ui.benchmark import (
    OPERATION_LABELS,
    BenchmarkReport,
    CudaImageLibrary,
    array_to_image,
    benchmark_report,
    image_to_array,
    load_library,
    repo_root,
)

BG = "#0b1220"
PANEL = "#111827"
CARD = "#1f2937"
CARD_ALT = "#243244"
TEXT = "#e5eefb"
MUTED = "#94a3b8"
ACCENT = "#22d3ee"
ACCENT_DARK = "#0891b2"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
ERROR = "#f87171"

PREVIEW_SIZE = (560, 360)
MAX_BENCHMARK_RUNS = 3

LABEL_TO_OPERATION = {label: key for key, label in OPERATION_LABELS.items()}


class ImageBenchmarkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CUDA Image Studio")
        self.geometry("1500x960")
        self.minsize(1280, 860)
        self.configure(bg=BG)
        self.withdraw()

        try:
            self.library: CudaImageLibrary = load_library()
        except FileNotFoundError as exc:
            messagebox.showerror("CUDA library not found", str(exc))
            self.destroy()
            raise

        self.current_image_array: np.ndarray | None = None
        self.current_output_array: np.ndarray | None = None
        self.current_file: Path | None = None
        self._input_photo: ImageTk.PhotoImage | None = None
        self._output_photo: ImageTk.PhotoImage | None = None
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.operation_var = tk.StringVar(value=OPERATION_LABELS["blur"])
        self.strength_var = tk.DoubleVar(value=1.0)
        self.status_var = tk.StringVar(value=f"Loaded CUDA library: {self.library.path.name}")
        self.cpu_time_var = tk.StringVar(value="-")
        self.naive_time_var = tk.StringVar(value="-")
        self.optimized_time_var = tk.StringVar(value="-")
        self.speedup_naive_var = tk.StringVar(value="-")
        self.speedup_optimized_var = tk.StringVar(value="-")
        self.input_info_var = tk.StringVar(value="No image loaded")
        self.output_info_var = tk.StringVar(value="No output yet")

        self._configure_styles()
        self._build_layout()
        self._set_placeholder(self.input_canvas, "Load an image to start")
        self._set_placeholder(self.output_canvas, "Benchmark output will appear here")
        self.deiconify()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.TCombobox", fieldbackground=CARD_ALT, background=CARD_ALT, foreground=TEXT, arrowcolor=TEXT, bordercolor=CARD_ALT)
        style.map("Dark.TCombobox", fieldbackground=[("readonly", CARD_ALT)], foreground=[("readonly", TEXT)])

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(18, 10))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        tk.Label(header, text="CUDA Image Studio", bg=BG, fg=TEXT, font=("Segoe UI", 24, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Simple CPU vs GPU image processing benchmark",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        tk.Label(
            header,
            textvariable=self.status_var,
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI", 11, "bold"),
            wraplength=520,
            justify="right",
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        body = tk.Frame(self, bg=BG)
        body.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=18)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.control_panel = tk.Frame(body, bg=PANEL, highlightbackground=CARD, highlightthickness=1, width=300)
        self.control_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        self.control_panel.grid_propagate(False)
        self.control_panel.grid_columnconfigure(0, weight=1)

        self.content_panel = tk.Frame(body, bg=PANEL, highlightbackground=CARD, highlightthickness=1)
        self.content_panel.grid(row=0, column=1, sticky="nsew")
        self.content_panel.grid_columnconfigure(0, weight=1)
        self.content_panel.grid_rowconfigure(0, weight=0)
        self.content_panel.grid_rowconfigure(1, weight=0)
        self.content_panel.grid_rowconfigure(2, weight=1)

        self._build_controls()
        self._build_previews()
        self._build_results()
        self._build_plot()

        footer = tk.Frame(self, bg=BG)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(10, 14))
        tk.Label(footer, textvariable=self.status_var, bg=BG, fg=MUTED, anchor="w", justify="left").pack(fill="x")

    def _build_controls(self) -> None:
        tk.Label(self.control_panel, text="Controls", bg=PANEL, fg=TEXT, font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 10))

        button_row = tk.Frame(self.control_panel, bg=PANEL)
        button_row.grid(row=1, column=0, sticky="ew", padx=14)
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)

        self.load_button = tk.Button(
            button_row,
            text="Load Image",
            command=self.load_image_dialog,
            bg=ACCENT,
            fg="#042f3a",
            activebackground=ACCENT_DARK,
            activeforeground=TEXT,
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8,
        )
        self.load_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.save_button = tk.Button(
            button_row,
            text="Save Output",
            command=self.save_output_image,
            bg=SUCCESS,
            fg="#052e16",
            activebackground="#10b981",
            activeforeground=TEXT,
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8,
        )
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        operation_card = tk.Frame(self.control_panel, bg=CARD, highlightbackground=CARD_ALT, highlightthickness=1)
        operation_card.grid(row=2, column=0, sticky="ew", padx=14, pady=(14, 0))
        operation_card.grid_columnconfigure(0, weight=1)
        tk.Label(operation_card, text="Operation", bg=CARD, fg=TEXT, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        self.operation_combo = ttk.Combobox(
            operation_card,
            values=list(OPERATION_LABELS.values()),
            textvariable=self.operation_var,
            state="readonly",
            style="Dark.TCombobox",
        )
        self.operation_combo.grid(row=1, column=0, sticky="ew", padx=12)

        strength_frame = tk.Frame(self.control_panel, bg=CARD, highlightbackground=CARD_ALT, highlightthickness=1)
        strength_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(14, 0))
        strength_frame.grid_columnconfigure(0, weight=1)
        tk.Label(strength_frame, text="Strength", bg=CARD, fg=TEXT, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        tk.Label(strength_frame, textvariable=self.strength_var, bg=CARD, fg=ACCENT, font=("Segoe UI", 11, "bold")).grid(row=1, column=0, sticky="w", padx=12)
        self.strength_scale = tk.Scale(
            strength_frame,
            from_=1.0,
            to=5.0,
            orient="horizontal",
            resolution=1.0,
            variable=self.strength_var,
            bg=CARD,
            fg=TEXT,
            troughcolor=CARD_ALT,
            highlightthickness=0,
            activebackground=ACCENT,
            relief="flat",
        )
        self.strength_scale.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        action_frame = tk.Frame(self.control_panel, bg=PANEL)
        action_frame.grid(row=4, column=0, sticky="ew", padx=14, pady=(14, 0))
        action_frame.grid_columnconfigure(0, weight=1)
        self.run_button = tk.Button(
            action_frame,
            text="Run Benchmark",
            command=self.run_benchmark,
            bg=WARNING,
            fg="#3f2b00",
            activebackground="#f59e0b",
            activeforeground=TEXT,
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=10,
        )
        self.run_button.grid(row=0, column=0, sticky="ew")

        note = tk.Label(
            self.control_panel,
            text="Build the CUDA library first, then benchmark CPU, CUDA naive, and CUDA optimized paths.",
            bg=PANEL,
            fg=MUTED,
            wraplength=250,
            justify="left",
        )
        note.grid(row=5, column=0, sticky="w", padx=14, pady=(16, 0))

    def _build_previews(self) -> None:
        preview_row = tk.Frame(self.content_panel, bg=PANEL)
        preview_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        preview_row.grid_columnconfigure(0, weight=1)
        preview_row.grid_columnconfigure(1, weight=1)

        self.input_card, self.input_canvas, self.input_info_label = self._create_preview_card(preview_row, "Input Image", self.input_info_var)
        self.input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.output_card, self.output_canvas, self.output_info_label = self._create_preview_card(preview_row, "Output Image", self.output_info_var)
        self.output_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    def _build_results(self) -> None:
        results_card = tk.Frame(self.content_panel, bg=CARD, highlightbackground=CARD_ALT, highlightthickness=1)
        results_card.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        results_card.grid_columnconfigure((0, 1, 2), weight=1)
        results_card.grid_rowconfigure((0, 1), weight=1)

        self._create_metric_card(results_card, 0, 0, "CPU Time", self.cpu_time_var, ACCENT)
        self._create_metric_card(results_card, 0, 1, "CUDA Naive", self.naive_time_var, WARNING)
        self._create_metric_card(results_card, 0, 2, "CUDA Optimized", self.optimized_time_var, SUCCESS)
        self._create_metric_card(results_card, 1, 0, "Speedup Naive", self.speedup_naive_var, ACCENT)
        self._create_metric_card(results_card, 1, 1, "Speedup Optimized", self.speedup_optimized_var, SUCCESS)

    def _build_plot(self) -> None:
        plot_card = tk.Frame(self.content_panel, bg=CARD, highlightbackground=CARD_ALT, highlightthickness=1)
        plot_card.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        plot_card.grid_columnconfigure(0, weight=1)
        plot_card.grid_rowconfigure(1, weight=1)

        tk.Label(plot_card, text="Visualization", bg=CARD, fg=TEXT, font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))
        self.figure = Figure(figsize=(9.6, 4.8), dpi=100, facecolor=CARD)
        self.ax_size = self.figure.add_subplot(1, 2, 1)
        self.ax_compare = self.figure.add_subplot(1, 2, 2)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_card)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=CARD, highlightthickness=0)
        self.canvas_widget.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._clear_plot()

    def _create_preview_card(self, parent: tk.Widget, title: str, info_var: tk.StringVar) -> tuple[tk.Frame, tk.Canvas, tk.Label]:
        card = tk.Frame(parent, bg=CARD, highlightbackground=CARD_ALT, highlightthickness=1)
        card.grid_columnconfigure(0, weight=1)
        tk.Label(card, text=title, bg=CARD, fg=TEXT, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))
        canvas = tk.Canvas(card, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], bg=BG, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew", padx=12)
        info = tk.Label(card, textvariable=info_var, bg=CARD, fg=MUTED, anchor="w")
        info.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))
        return card, canvas, info

    def _create_metric_card(self, parent: tk.Widget, row: int, column: int, title: str, value_var: tk.StringVar, accent_color: str) -> None:
        card = tk.Frame(parent, bg=CARD_ALT, highlightbackground=CARD, highlightthickness=1)
        card.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)
        tk.Label(card, text=title, bg=CARD_ALT, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(card, textvariable=value_var, bg=CARD_ALT, fg=accent_color, font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(4, 12))

    def _style_axes(self) -> None:
        for axis in (self.ax_size, self.ax_compare):
            axis.set_facecolor(PANEL)
            axis.tick_params(colors=TEXT)
            axis.spines["top"].set_color(MUTED)
            axis.spines["bottom"].set_color(MUTED)
            axis.spines["left"].set_color(MUTED)
            axis.spines["right"].set_color(MUTED)
            axis.title.set_color(TEXT)
            axis.xaxis.label.set_color(TEXT)
            axis.yaxis.label.set_color(TEXT)

    def _clear_plot(self) -> None:
        self.ax_size.clear()
        self.ax_compare.clear()
        self._style_axes()
        self.ax_size.set_title("Image Size vs Time")
        self.ax_size.set_xlabel("Image size (px)")
        self.ax_size.set_ylabel("Milliseconds")
        self.ax_size.text(0.5, 0.5, "Run a benchmark to see the graph", transform=self.ax_size.transAxes, ha="center", va="center", color=MUTED)
        self.ax_compare.set_title("CPU vs GPU Comparison")
        self.ax_compare.set_ylabel("Milliseconds")
        self.ax_compare.text(0.5, 0.5, "Run a benchmark to compare modes", transform=self.ax_compare.transAxes, ha="center", va="center", color=MUTED)
        self.canvas.draw_idle()

    def _set_placeholder(self, canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        canvas.create_rectangle(0, 0, PREVIEW_SIZE[0], PREVIEW_SIZE[1], fill=BG, outline=BG)
        canvas.create_text(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, text=text, fill=MUTED, font=("Segoe UI", 13), justify="center")

    def _show_array_on_canvas(self, canvas: tk.Canvas, array: np.ndarray, storage: str, placeholder: str) -> None:
        if array is None:
            self._set_placeholder(canvas, placeholder)
            return

        image = array_to_image(array)
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, PREVIEW_SIZE[0], PREVIEW_SIZE[1], fill=BG, outline=BG)
        canvas.create_image(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2, image=photo, anchor="center")
        setattr(self, storage, photo)

    def load_image_dialog(self) -> None:
        start_dir = repo_root() / "images"
        file_path = filedialog.askopenfilename(
            title="Load image",
            initialdir=str(start_dir),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self.load_image(Path(file_path))

    def load_image(self, path: Path) -> None:
        with Image.open(path) as image:
            self.current_image_array = image_to_array(image)

        self.current_file = path
        self.current_output_array = None
        self.input_info_var.set(f"{path.name} • {self.current_image_array.shape[1]} x {self.current_image_array.shape[0]}")
        self.output_info_var.set("No output yet")
        self._show_array_on_canvas(self.input_canvas, self.current_image_array, "_input_photo", "No image loaded")
        self._set_placeholder(self.output_canvas, "Run the benchmark to generate an output")
        self.cpu_time_var.set("-")
        self.naive_time_var.set("-")
        self.optimized_time_var.set("-")
        self.speedup_naive_var.set("-")
        self.speedup_optimized_var.set("-")
        self._clear_plot()
        self.status_var.set(f"Loaded {path.name}")

    def save_output_image(self) -> None:
        if self.current_output_array is None:
            messagebox.showinfo("Nothing to save", "Run a benchmark first so an output image is available.")
            return

        default_name = "output.png"
        if self.current_file is not None:
            default_name = f"{self.current_file.stem}_{self.operation_var.get().replace(' ', '_').lower()}.png"

        file_path = filedialog.asksaveasfilename(
            title="Save output image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("WebP image", "*.webp"),
            ],
        )
        if not file_path:
            return

        array_to_image(self.current_output_array).save(file_path)
        self.status_var.set(f"Saved output to {Path(file_path).name}")

    def run_benchmark(self) -> None:
        if self.current_image_array is None:
            messagebox.showinfo("Load an image", "Choose an image file before running a benchmark.")
            return

        operation_key = LABEL_TO_OPERATION[self.operation_var.get()]
        strength = float(self.strength_var.get())
        image_copy = self.current_image_array.copy()
        self._set_busy(True)
        self.status_var.set("Running benchmark...")

        future = self._executor.submit(self._run_benchmark_worker, image_copy, operation_key, strength)
        future.add_done_callback(lambda completed: self.after(0, lambda: self._handle_benchmark_result(completed)))

    def _run_benchmark_worker(self, image_array: np.ndarray, operation_key: str, strength: float) -> BenchmarkReport:
        return benchmark_report(self.library, image_array, operation_key, strength, runs=MAX_BENCHMARK_RUNS)

    def _handle_benchmark_result(self, future) -> None:
        try:
            report = future.result()
        except Exception as exc:
            self._set_busy(False)
            messagebox.showerror("Benchmark failed", str(exc))
            self.status_var.set("Benchmark failed")
            return

        self.current_output_array = report.optimized_output
        self.cpu_time_var.set(f"{report.timing.cpu_ms:.2f} ms")
        self.naive_time_var.set(f"{report.timing.cuda_naive_ms:.2f} ms")
        self.optimized_time_var.set(f"{report.timing.cuda_optimized_ms:.2f} ms")
        self.speedup_naive_var.set(f"{report.timing.speedup_naive:.2f}x")
        self.speedup_optimized_var.set(f"{report.timing.speedup_optimized:.2f}x")
        if self.current_file is not None:
            self.output_info_var.set(f"Optimized CUDA output • {self.current_file.name}")
        else:
            self.output_info_var.set("Optimized CUDA output")
        self._show_array_on_canvas(self.output_canvas, report.optimized_output, "_output_photo", "No output yet")
        self._update_plot(report)
        self.status_var.set("Benchmark complete")
        self._set_busy(False)

    def _update_plot(self, report: BenchmarkReport) -> None:
        self.ax_size.clear()
        self.ax_compare.clear()
        self._style_axes()

        sizes = [point.size for point in report.size_points]
        cpu_times = [point.cpu_ms for point in report.size_points]
        naive_times = [point.cuda_naive_ms for point in report.size_points]
        optimized_times = [point.cuda_optimized_ms for point in report.size_points]

        self.ax_size.plot(sizes, cpu_times, marker="o", color=ACCENT, label="CPU")
        self.ax_size.plot(sizes, naive_times, marker="o", color=WARNING, label="CUDA Naive")
        self.ax_size.plot(sizes, optimized_times, marker="o", color=SUCCESS, label="CUDA Optimized")
        self.ax_size.set_title("Image Size vs Time")
        self.ax_size.set_xlabel("Image size (square px)")
        self.ax_size.set_ylabel("Milliseconds")
        self.ax_size.legend(facecolor=CARD, edgecolor=CARD_ALT, labelcolor=TEXT)
        self.ax_size.grid(True, alpha=0.18)

        compare_labels = ["CPU", "Naive", "Optimized"]
        compare_values = [report.timing.cpu_ms, report.timing.cuda_naive_ms, report.timing.cuda_optimized_ms]
        bar_colors = [ACCENT, WARNING, SUCCESS]
        bars = self.ax_compare.bar(compare_labels, compare_values, color=bar_colors)
        self.ax_compare.set_title("CPU vs GPU Comparison")
        self.ax_compare.set_ylabel("Milliseconds")
        self.ax_compare.grid(axis="y", alpha=0.18)
        for bar, value in zip(bars, compare_values, strict=False):
            self.ax_compare.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=9,
            )

        self.canvas.draw_idle()

    def _set_busy(self, busy: bool) -> None:
        button_state = tk.DISABLED if busy else tk.NORMAL
        self.load_button.configure(state=button_state)
        self.save_button.configure(state=button_state)
        self.run_button.configure(state=button_state)
        self.operation_combo.configure(state="disabled" if busy else "readonly")
        self.strength_scale.configure(state=button_state)

    def _on_close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


def main() -> None:
    app = ImageBenchmarkApp()
    app.mainloop()

    ''' mainloop() starts an infinite loop that keeps running until you close the window
    - keeps your window open
    - waits for user actions like click
    - responds to events
    '''




    '''
   Your Python code
      ↓
tkinter (Python)
      ↓
_tkinter (C bridge)
      ↓
Tcl/Tk (C library)
      ↓
OS-specific GUI system
    '''
