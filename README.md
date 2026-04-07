# CUDA-Accelerated Image Processing with Kernel Optimization and Performance Benchmarking

> PCAP Mini-Project | MIT Manipal | CSE-B | IA4 Evaluation

---

## Team

| Name | Roll No. | Reg. No. |
|------|----------|----------|
| Yashvardhan Gupta | 55 | 230905442 |
| Shubhendu Arya | 56 | 230905458 |
| Abhyuday Verma | 52 | 230905418 |

**Guide:** Dr. Jyothi Upadhya K

---

## Overview

This project implements and benchmarks image processing algorithms using **NVIDIA CUDA**, comparing GPU-accelerated performance against traditional CPU implementations. The goal is to demonstrate the real-world speedup achievable through parallel kernel design and memory optimization.

---

## Algorithms Implemented

| Algorithm | Category |
|-----------|----------|
| Gaussian Blur | Convolution |
| Sharpening Filter | Convolution |
| Sobel Edge Detection | Edge Detection |
| Dilation | Morphological |
| Erosion | Morphological |

---

## CUDA Kernel Versions

Each algorithm is implemented in multiple versions to study optimization impact:

1. **Naive** — global memory only
2. **Shared Memory** — tiled access via `__shared__`
3. **Memory Coalesced** — optimized memory access patterns
4. *(Optional)* **Tiling** — further latency hiding

---

## Project Structure

```
cuda-image-processing/
├── src/
│   ├── cpu/                  # CPU baseline implementations
│   │   ├── gaussian_blur.cpp
│   │   ├── sobel.cpp
│   │   └── morphological.cpp
│   ├── cuda/                 # CUDA kernel implementations
│   │   ├── naive/
│   │   ├── shared_mem/
│   │   └── coalesced/
│   └── benchmark/            # Benchmarking framework
│       └── timer.cuh
├── include/                  # Header files
├── images/                   # Test images (various sizes)
├── results/                  # Benchmark output CSVs and graphs
├── ui/                       # (Optional) Web/CLI UI
├── report/                   # Final report and synopsis
├── Makefile
└── README.md
```

---

## Requirements

- NVIDIA GPU with CUDA Compute Capability ≥ 3.0
- CUDA Toolkit ≥ 11.0
- OpenCV (for image I/O)
- Python 3.x + matplotlib (for benchmark graphs)
- GCC / G++ ≥ 7.0

---

## Build & Run

```bash
# Clone the repo
git clone <repo-url>
cd cuda-image-processing

# Build all targets
make all

# Run a specific operation
./bin/process --image images/sample.jpg --op gaussian --mode cuda_shared

# Run full benchmark suite
./bin/benchmark --image images/sample.jpg --sizes 256 512 1024 2048
```

---

## Benchmarking

The benchmarking framework measures:
- **Execution time** (CPU vs GPU) across image sizes
- **Speedup factor** = `T_cpu / T_gpu`
- **Scalability** with increasing resolution

Results are exported to `results/` as CSV and plotted using Python.

---

## Contributing

1. Pick a task from the [Issues](#) tab or coordinate with the team
2. Create a branch: `git checkout -b feature/<your-feature>`
3. Commit with clear messages: `git commit -m "add: sobel cuda shared mem kernel"`
4. Push and open a Pull Request against `main`

### Commit Prefix Convention
| Prefix | Use for |
|--------|---------|
| `add:` | New feature or file |
| `fix:` | Bug fix |
| `opt:` | Performance optimization |
| `docs:` | Documentation changes |
| `bench:` | Benchmarking code or results |

---

## Timeline

| Milestone | Date |
|-----------|------|
| Synopsis submission | 21 March 2026 ✅ |
| Report + presentation upload | 13 April 2026 |
| Evaluation & presentations | 13–20 April 2026 |
| Final report on LMS | 21 April 2026 |

---

## References

1. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, Pearson, 2002.
2. D. B. Kirk and W. W. Hwu, *Programming Massively Parallel Processors*, Elsevier, 2013.
3. [NVIDIA CUDA Documentation](https://developer.nvidia.com/cuda-zone)
4. [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
5. M. J. Quinn, *Parallel Programming in C with MPI and OpenMP*, McGraw Hill.
6. K. Hwang & Briggs, *Computer Architecture & Parallel Processing*, McGraw Hill.
