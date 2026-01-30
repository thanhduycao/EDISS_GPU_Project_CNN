# Chest X-Ray CNN Benchmark: GPU Optimization

This project implements a Convolutional Neural Network (CNN) forward pass from scratch using **CUDA C++** to benchmark various GPU optimization techniques. The goal is to accelerate feature extraction on high-resolution medical images (Chest X-Rays) compared to a CPU baseline.

The project explores four distinct CUDA optimization strategies: **Constant Memory**, **Shared Memory Tiling**, **Loop Unrolling**, and **Thread Coarsening**.

## Project Structure

The repository is organized by optimization technique:

| Directory | Description | Key File |
| --- | --- | --- |
| `cpu/` | **Baseline Implementation.** Serial C++ implementation for performance comparison. | `cnn_cpu.cpp` |
| `constant/` | **Optimization 1:** Uses **Constant Memory** (64KB cache) to store convolution kernels for faster broadcast access. | `xray_cnn.cu` |
| `shared/` | **Optimization 2:** Uses **Shared Memory** to tile input data, reducing global memory bandwidth usage. | `shared_cnn.cu` |
| `Unrolling/` | **Optimization 3:** Uses `#pragma unroll` and C++ templates to reduce loop overhead and increase instruction throughput. | `xray_unrolled.cu` |
| `thread-coarsening/` | **Optimization 4:** Assigns multiple output pixels to a single thread to hide latency and improve instruction-level parallelism. | `main.cu` |
| `data/` | Directory structure for the dataset (see "Dataset" below). | *(Images)* |

## Benchmark Configurations

The system runs three distinct configurations to test performance across different workload intensities:

1. **Small:** 32x32 Input, 3x3 Kernels (Low intensity)
2. **Medium:** 256x256 Input, 3x3 Kernels (Medium intensity)
3. **Large:** 1024x1024 Input, 7x7 Kernels (High intensity)

Each configuration performs:

* **Convolution Layer** (Feature Extraction)
* **ReLU Activation** (Non-linearity)
* **Max Pooling** (Downsampling)

## Prerequisites

* **CUDA Toolkit** (nvcc compiler)
* **C++ Compiler** (g++ or clang)
* **Python 3** (for visualization script)
* **Libraries:** `stb_image.h` (included in source)
* **Python Deps:** `numpy`, `matplotlib`

## How to Run

### 1. Dataset Setup

Ensure your data is structured as follows in the root directory:

```text
data/
  └── chest_xray/
      └── train/
          ├── NORMAL/
          └── PNEUMONIA/

```

*(Note: The code loads `.jpeg`, `.jpg`, or `.png` images).*

### 2. Compile and Run

Each directory contains a `Makefile`. You can build and run specific versions individually.

**Running the CPU Baseline:**

```bash
cd cpu
g++ -O3 -o cpu cnn_cpu.cpp -lm
./cpu

```

**Running a CUDA Version (e.g., Shared Memory):**

```bash
cd shared
make
./shared_cnn

```

**Running the Unrolled Version:**

```bash
cd Unrolling
nvcc -O3 -arch=sm_80 -o xray_unrolled xray_unrolled.cu
./xray_unrolled

```

### 3. Visualizing Feature Maps

The programs dump the output feature maps to `.txt` files (e.g., `pooled_feature_maps_Medium.txt`). Use the provided Python script to visualize them as heatmaps.

```bash
# Configure the file paths in plot_feature_maps.py if necessary
python3 plot_feature_maps.py

```

This will generate images in the `featuremap_plots/` folder showing what features the network is extracting.

## Performance Analysis

* **CPU Baseline:** Serves as the reference point (1x speed).
* **Constant Memory:** Effective for smaller kernels where weights fit in the constant cache.
* **Shared Memory:** Significantly reduces global memory access; most effective for larger inputs (1024x1024).
* **Loop Unrolling:** Reduces control flow overhead; highly effective for fixed kernel sizes (3x3, 7x7).
* **Thread Coarsening:** Increases work per thread to hide memory latency.

## Authors

* **Adiel Luna Medina**
* **Manuel Padilla**
* **Rajshree Rai**
* **Srishti Karanth**
* **Thanh Duy Cao**

## License

This project is for academic and educational purposes.
