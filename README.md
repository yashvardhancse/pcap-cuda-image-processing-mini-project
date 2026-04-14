# CUDA Image Studio

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-00A3E0?logo=windows-terminal&logoColor=white&style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Toolkit-76B900?logo=nvidia&logoColor=white&style=for-the-badge)
![NVCC](https://img.shields.io/badge/Compiler-NVCC-76B900?logo=nvidia&logoColor=white&style=for-the-badge)
![CMake](https://img.shields.io/badge/Build-CMake-064F8C?logo=cmake&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)
![Pillow](https://img.shields.io/badge/Pillow-0A1A2F?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![ctypes](https://img.shields.io/badge/Bridge-ctypes-5C6BC0?style=for-the-badge)

Colorful Python + CUDA desktop app for image processing, built around a Tkinter UI, a native CUDA shared library, and CPU/GPU benchmark comparison.

![CUDA Image Studio preview](app%20ui%20screenshots%20python%20tkinter/homepage.png)

## What This Project Does

- Opens an image in a simple Tkinter interface.
- Runs image filters on CPU and on an NVIDIA GPU.
- Measures CPU vs CUDA time and shows speedups.
- Displays the input, output, and benchmark charts in the UI.
- Uses separate CUDA source files for each effect so the project stays easy to read.

## Project Flow

```text
Image file
  -> Pillow / NumPy
  -> Tkinter UI
  -> ctypes bridge
  -> cuda_image_processing.dll
  -> CUDA kernels on GPU
  -> result copied back
  -> Tkinter preview + benchmark chart
```

## 1. What You Need First

### CMake

CMake is a build system generator. It reads the handwritten CMakeLists.txt file and generates the build files for your platform.

- Windows: Visual Studio solution and project files
- Linux: Makefiles or Ninja files
- macOS: Xcode or Ninja files

CMake is usually not preinstalled on all systems. On Windows, you can install it separately or use the copy bundled with Visual Studio.

### nvcc

nvcc is the NVIDIA CUDA compiler. It compiles the .cu files into object files and then into the final shared library.

Verify it with:

```powershell
nvcc -V
nvidia-smi
```

If Windows cannot find nvcc, add this folder to PATH:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

### Python and Tkinter

Tkinter is the standard Python GUI toolkit.

- Windows: Tkinter is included with the official python.org installer.
- Linux: install the Tk package from your package manager.
- macOS: the python.org installer includes Tk support.

Verify Tkinter with:

```bash
python -m tkinter
```

Install the Python packages used by this project:

```bash
pip install numpy pillow matplotlib
```

`ctypes` is part of the Python standard library, so no separate install is needed.

## 2. Build the CUDA Library

The native library is built from the files in the cuda folder.

The build folder inside `cuda/` is not handwritten. CMake creates it automatically when you configure the project. It stores generated Visual Studio files, CMake cache data, intermediate object files, and the final DLL.

Build on Windows with:

```powershell
cmake -S cuda -B cuda/build
cmake --build cuda/build --config Release
```

After a successful build, the main output is:

```text
cuda/build/Release/cuda_image_processing.dll
```

### RTX GPU Targets

The current CMake setup targets common NVIDIA architectures:

| CUDA Architecture | Typical GPU Family |
| --- | --- |
| 75 | RTX 20 series |
| 80 | Ampere data center / workstation class |
| 86 | RTX 30 series |
| 89 | RTX 40 series |

That means the project is already aimed at modern RTX-class cards.

## 3. Run the Tkinter UI

Launch the app from the repository root:

```bash
python python_ui/main.py
```

The UI loads the native DLL with `ctypes`, prepares image data as contiguous RGBA NumPy arrays, and runs benchmarks in a background thread so the window stays responsive.

If you want to point the app to a custom DLL path, set:

```text
CUDA_IMAGE_PROCESSING_LIB
```

## 4. How Python Connects to CUDA and Tkinter

The connection is simple:

1. Tkinter handles buttons, dropdowns, sliders, image previews, and charts.
2. Python converts the selected image into a NumPy RGBA buffer.
3. `ctypes` calls exported native functions from the CUDA shared library.
4. The CUDA code copies pixels to the GPU, runs the kernel, and copies results back.
5. Tkinter displays the processed image and benchmark numbers.

This project uses `extern "C"` exports in the native library so Python can call the functions through a C-compatible ABI.

## 5. How Benchmark Time Is Measured

The benchmark uses Python's `time.perf_counter()` around each native call.

That means the measured time is end-to-end wall clock time:

- host to device copy
- kernel launch and execution
- device to host copy
- return to Python

Each mode is run multiple times and the average is reported. The same input image is used for each mode so the comparison is fair.

Speedup is calculated as:

```text
speedup = CPU_time / GPU_time
```

If the result is greater than 1, the GPU path is faster.

## 6. CPU, CUDA Naive, and CUDA Optimized

| Mode | What it Means | Why It Exists |
| --- | --- | --- |
| CPU | Host-side reference implementation | Baseline for comparison |
| CUDA Naive | One thread per pixel, direct global-memory reads | Easy to understand and debug |
| CUDA Optimized | One thread per pixel, shared-memory tiles | Reduces repeated memory reads and is usually much faster |

Why naive can be slower:

- Every thread reads its neighbors directly from global memory.
- Global memory is much slower than shared memory.
- For small images, launch and transfer overhead can dominate.

Why optimized is faster:

- A block loads a tile of pixels once.
- Threads reuse nearby pixels from shared memory.
- Less repeated work means lower memory traffic.

### Example Benchmark Output

Numbers vary by image size, GPU, driver, and selected filter. A typical result from this project looks like this:

| Mode | Example Time | Speedup vs CPU |
| --- | --- | --- |
| CPU | 39.30 ms | 1.00x |
| CUDA Naive | 30.74 ms | 1.28x |
| CUDA Optimized | 2.73 ms | 14.41x |

On larger images, the optimized CUDA path can deliver much bigger gains. The screenshots in this repo include an example where the optimized path is far ahead of CPU.

## 7. How the Image Techniques Work

### Convolution

Convolution is the core idea behind blur, sharpen, and edge detection. A small kernel slides over the image, and the pixel is replaced by a weighted combination of its neighbors.

```text
3x3 neighborhood        3x3 kernel

[ a  b  c ]            [ 1  2  1 ]
[ d  e  f ]    x       [ 2  4  2 ]   -> weighted sum
[ g  h  i ]            [ 1  2  1 ]
```

### Gaussian Blur

Gaussian blur smooths an image by giving the center pixel more weight than the outer pixels.

The kernel used here is:

```text
1  2  1
2  4  2   / 16
1  2  1
```

Use it when you want to soften noise, reduce detail, or prepare the image for sharpening and edge detection.

### Sharpening

Sharpening increases edge contrast. In this project it is built from a blur pass plus a detail boost:

```text
sharpened = original + amount * (original - blurred)
```

That means flat areas stay similar, while edges become more visible.

### Sobel Edge Detection

Sobel detects edges by measuring how quickly brightness changes in the X and Y directions.

```text
Gx = [ -1  0  1 ]
     [ -2  0  2 ]
     [ -1  0  1 ]

Gy = [ -1 -2 -1 ]
     [  0  0  0 ]
     [  1  2  1 ]
```

The final edge strength is based on the gradient magnitude:

```text
edge = sqrt(Gx^2 + Gy^2)
```

Bright pixels in the output mean stronger edges.

### Dilation

Dilation expands bright regions in an image.

- Look at a 3x3 neighborhood.
- Find the maximum intensity.
- Replace the center pixel with that value.

It is useful for making bright shapes thicker and connecting nearby bright areas.

### Erosion

Erosion is the opposite of dilation.

- Look at a 3x3 neighborhood.
- Find the minimum intensity.
- Replace the center pixel with that value.

It shrinks bright regions and is useful for removing small bright noise.

### Strength / Iterations

In this project, the strength slider controls the effect intensity or the number of passes, depending on the selected operation.

| Strength / Iterations | Result |
| --- | --- |
| 1 | Small change |
| 3 | Bigger change |
| 5 | Very strong change |

For blur, dilation, and erosion, more passes make the result stronger. For sharpening and Sobel, the strength value increases the emphasis.

## 8. One Thread = One Pixel

CUDA splits work into many small threads.

- Each thread processes one pixel.
- Threads are grouped into blocks.
- Blocks are arranged in a grid that covers the full image.

In this project the block size is 16 x 16 threads. That means each block can cover a 16 x 16 tile of the image, and the optimized kernels load the surrounding pixels into shared memory so nearby threads can reuse them.

This is the reason GPU processing can be extremely fast when the image is large enough and the kernel is written well.

## 9. Project Structure

```text
.
├── cuda/
│   ├── CMakeLists.txt
│   ├── common.cuh
│   ├── gaussian_blur.cu
│   ├── sharpening.cu
│   ├── sobel_edge.cu
│   ├── dilation.cu
│   └── erosion.cu
├── python_ui/
│   ├── main.py
│   ├── ui.py
│   └── benchmark.py
├── images/
└── app ui screenshots python tkinter/
```

### What common.cuh Does

`common.cuh` is the shared helper header used by all CUDA files.

It contains:

- shared constants such as channel count and tile size
- pixel indexing helpers
- clamp helpers
- grayscale conversion helpers
- GPU memory allocation and copy helpers
- the reusable `run_cuda_operation` wrapper

### What the build folder does

The `cuda/build/` directory is generated automatically by CMake and Visual Studio. It stores:

- generated solution and project files
- configuration cache files
- object files
- link outputs
- the final DLL or intermediate artifacts

You do not create this folder by hand, and you can safely delete and regenerate it if needed.

### What .cu Files Compile Into

The CUDA source files are compiled by nvcc into object files first and then linked into a shared library:

```text
.cu -> .obj -> cuda_image_processing.dll
```

On Linux the final file would normally be a `.so` shared object.

## 10. No NVIDIA GPU?

This project is designed for NVIDIA CUDA hardware.

- The UI is written in Python and Tkinter.
- The native library is built with CUDA.
- The GPU benchmark paths need NVIDIA hardware and the CUDA runtime.

Without an NVIDIA GPU, the CUDA acceleration path is not available. You can still read the code and study the UI structure, but the full benchmark experience is meant for a CUDA-capable machine.

## 11. Screenshot Gallery

These screenshots live in the `app ui screenshots python tkinter/` folder.

| Homepage | Operation Menu |
| --- | --- |
| ![Homepage](app%20ui%20screenshots%20python%20tkinter/homepage.png) | ![Operation dropdown](app%20ui%20screenshots%20python%20tkinter/operations%20drop%20down%20menu.png) |

| Sharpening Result | Sobel Edge Result |
| --- | --- |
| ![Sharpening result](app%20ui%20screenshots%20python%20tkinter/sharpening%20results%20homepage.png) | ![Sobel result](app%20ui%20screenshots%20python%20tkinter/us%20dollar%20currency%20sobel%20edge%20results.png) |

## 12. How to Run on an RTX Card

If you are using an RTX GPU, the current build already targets modern architectures. The practical checklist is:

1. Install the NVIDIA driver.
2. Install the CUDA Toolkit.
3. Confirm that `nvidia-smi` works.
4. Confirm that `nvcc -V` works.
5. Build the CUDA DLL with CMake.
6. Start the Tkinter app from `python_ui/main.py`.

If you add a new GPU later and want to retarget the build, the architecture list is controlled by `CMAKE_CUDA_ARCHITECTURES` inside `cuda/CMakeLists.txt`.

## 13. Tutorial Links

Tkinter learning resources:

- [Tkinter playlist](https://www.youtube.com/playlist?list=PLu0W_9lII9ajLcqRcj4PoEihkukF_OTzA)
- [Python UI tutorial](https://www.youtube.com/watch?v=ibf5cx221hk&pp=ygUOcHl0aG9uIHVpIHRraW4%3D)

CUDA convolution tutorial:

- [CUDA convolution video](https://www.youtube.com/watch?v=5gO2PwGS2kk)

## 14. Quick FAQ

- What is CMake? A build system generator that reads `CMakeLists.txt` and creates platform-specific build files.
- Is CMake preinstalled? Usually no.
- What is `cuda/build/`? Auto-generated build output from CMake and Visual Studio.
- What format are CUDA files compiled into? Object files first, then a shared library such as `.dll`.
- How does Python talk to CUDA? Through `ctypes` and exported `extern "C"` functions.
- How is the benchmark measured? With `time.perf_counter()` around the full native call.
- What is `common.cuh`? A shared CUDA helper header for all kernels.

## 15. Notes

- The Python UI uses Tkinter, not customtkinter.
- The native library is loaded from `cuda/build/Release` by default.
- You can override the native library location with `CUDA_IMAGE_PROCESSING_LIB`.
- The project includes CPU implementations so timing comparisons are meaningful.
- The optimized CUDA path is the version you want for real RTX GPU acceleration.
