from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
import ctypes
import os

import numpy as np
from PIL import Image

OPERATION_PREFIXES: dict[str, str] = {
    "blur": "gaussian_blur",
    "sharpen": "sharpening",
    "sobel": "sobel_edge",
    "dilation": "dilation",
    "erosion": "erosion",
}

OPERATION_LABELS: dict[str, str] = {
    "blur": "Gaussian Blur",
    "sharpen": "Sharpening",
    "sobel": "Sobel Edge",
    "dilation": "Dilation",
    "erosion": "Erosion",
}

MODE_SUFFIXES: tuple[str, ...] = ("cpu", "cuda_naive", "cuda_optimized")
DEFAULT_BENCHMARK_SIZES: tuple[int, ...] = (128, 256, 384, 512, 768, 1024)


def _dll_search_handles(library_path: Path) -> list[object]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    handles: list[object] = []
    candidate_directories = [library_path.parent]
    cuda_root = os.environ.get("CUDA_PATH")
    if cuda_root:
        candidate_directories.append(Path(cuda_root) / "bin")
    cuda_root_12_8 = os.environ.get("CUDA_PATH_V12_8")
    if cuda_root_12_8:
        candidate_directories.append(Path(cuda_root_12_8) / "bin")

    for candidate_directory in candidate_directories:
        if candidate_directory.exists():
            handles.append(os.add_dll_directory(str(candidate_directory)))

    return handles


@dataclass(frozen=True)
class TimingResult:
    cpu_ms: float
    cuda_naive_ms: float
    cuda_optimized_ms: float
    speedup_naive: float
    speedup_optimized: float


@dataclass(frozen=True)
class SizeBenchmarkPoint:
    size: int
    cpu_ms: float
    cuda_naive_ms: float
    cuda_optimized_ms: float


@dataclass(frozen=True)
class BenchmarkReport:
    operation_key: str
    timing: TimingResult
    optimized_output: np.ndarray
    size_points: list[SizeBenchmarkPoint]


class CudaImageLibrary:
    def __init__(self, library_path: Path) -> None:
        self.path = library_path
        self._dll_search_handles = _dll_search_handles(library_path)
        self._library = ctypes.CDLL(str(library_path))
        self._bind_functions()

    def _bind_functions(self) -> None:
        pixel_buffer = ctypes.POINTER(ctypes.c_ubyte)
        for operation_prefix in OPERATION_PREFIXES.values():
            for suffix in MODE_SUFFIXES:
                function_name = f"{operation_prefix}_{suffix}"
                function = getattr(self._library, function_name)
                function.argtypes = [pixel_buffer, pixel_buffer, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                function.restype = ctypes.c_int

    def process(self, operation_key: str, mode: str, image_array: np.ndarray, strength: float) -> np.ndarray:
        if operation_key not in OPERATION_PREFIXES:
            raise KeyError(f"Unknown operation: {operation_key}")
        if mode not in MODE_SUFFIXES:
            raise KeyError(f"Unknown processing mode: {mode}")

        array = np.ascontiguousarray(image_array, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError("Expected an RGBA image array with shape (height, width, 4).")

        output = np.empty_like(array)
        function_name = f"{OPERATION_PREFIXES[operation_key]}_{mode}"
        function = getattr(self._library, function_name)
        status = function(
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.c_int(array.shape[1]),
            ctypes.c_int(array.shape[0]),
            ctypes.c_float(float(strength)),
        )
        if status != 0:
            raise RuntimeError(f"CUDA call failed: {function_name} returned {status}")
        return output


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_library_path() -> Path:
    env_path = os.environ.get("CUDA_IMAGE_PROCESSING_LIB")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    search_roots = [
        repo_root() / "cuda" / "build",
        repo_root() / "cuda" / "build" / "Release",
        repo_root() / "cuda" / "build" / "Debug",
    ]
    patterns = ["cuda_image_processing.dll", "libcuda_image_processing.so", "libcuda_image_processing.dylib"]

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for pattern in patterns:
            exact = search_root / pattern
            if exact.exists():
                return exact
            for candidate in search_root.rglob(pattern):
                return candidate

    raise FileNotFoundError(
        "Could not find the CUDA shared library. Build the project with CMake first or set CUDA_IMAGE_PROCESSING_LIB."
    )


def load_library() -> CudaImageLibrary:
    return CudaImageLibrary(find_library_path())


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8), mode="RGBA")


def resize_square(array: np.ndarray, size: int) -> np.ndarray:
    resized = array_to_image(array).resize((size, size), Image.Resampling.LANCZOS)
    return image_to_array(resized)


def benchmark_mode(
    library: CudaImageLibrary,
    image_array: np.ndarray,
    operation_key: str,
    mode: str,
    strength: float,
    runs: int = 3,
) -> tuple[float, np.ndarray]:
    timings: list[float] = []
    last_output: np.ndarray | None = None
    run_count = max(1, int(runs))

    for _ in range(run_count):
        start = perf_counter()
        last_output = library.process(operation_key, mode, image_array, strength)
        timings.append((perf_counter() - start) * 1000.0)

    assert last_output is not None
    return mean(timings), last_output


def benchmark_sizes(min_dimension: int) -> list[int]:
    sizes = [size for size in DEFAULT_BENCHMARK_SIZES if size <= min_dimension]
    if not sizes:
        sizes = [min_dimension]
    elif sizes[-1] != min_dimension and min_dimension <= DEFAULT_BENCHMARK_SIZES[-1]:
        sizes.append(min_dimension)
    return sorted(set(sizes))


def benchmark_report(
    library: CudaImageLibrary,
    image_array: np.ndarray,
    operation_key: str,
    strength: float,
    runs: int = 3,
) -> BenchmarkReport:
    cpu_ms, _ = benchmark_mode(library, image_array, operation_key, "cpu", strength, runs=runs)
    naive_ms, _ = benchmark_mode(library, image_array, operation_key, "cuda_naive", strength, runs=runs)
    optimized_ms, optimized_output = benchmark_mode(library, image_array, operation_key, "cuda_optimized", strength, runs=runs)

    timing = TimingResult(
        cpu_ms=cpu_ms,
        cuda_naive_ms=naive_ms,
        cuda_optimized_ms=optimized_ms,
        speedup_naive=(cpu_ms / naive_ms) if naive_ms > 0 else 0.0,
        speedup_optimized=(cpu_ms / optimized_ms) if optimized_ms > 0 else 0.0,
    )

    size_points: list[SizeBenchmarkPoint] = []
    height, width = image_array.shape[:2]
    limit = min(height, width)
    for size in benchmark_sizes(limit):
        sized_image = resize_square(image_array, size)
        size_cpu_ms, _ = benchmark_mode(library, sized_image, operation_key, "cpu", strength, runs=1)
        size_naive_ms, _ = benchmark_mode(library, sized_image, operation_key, "cuda_naive", strength, runs=1)
        size_optimized_ms, _ = benchmark_mode(library, sized_image, operation_key, "cuda_optimized", strength, runs=1)
        size_points.append(
            SizeBenchmarkPoint(
                size=size,
                cpu_ms=size_cpu_ms,
                cuda_naive_ms=size_naive_ms,
                cuda_optimized_ms=size_optimized_ms,
            )
        )

    return BenchmarkReport(
        operation_key=operation_key,
        timing=timing,
        optimized_output=optimized_output,
        size_points=size_points,
    )
