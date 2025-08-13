# MoveNet Performance Benchmark on Jetson Nano

This repository contains the resources and instructions to benchmark the performance of the MoveNet Single-Pose Lightning model on an NVIDIA Jetson Nano 2GB Developer Kit.

The goal is to measure the raw inference speed (latency and FPS) of the model after optimization with NVIDIA TensorRT.

---

## Prerequisites

* **Hardware:** NVIDIA Jetson Nano 2GB Developer Kit
* **Software:** JetPack 4.6.x (which includes TensorRT 8.2.1 and CUDA 10.2)

---

## Benchmarking Procedure

Follow these steps directly on your Jetson Nano terminal to replicate the benchmark.

### Step 1: Download the Model

Download the pre-converted ONNX model from this repository.

1.  Navigate to the `movenet_lightning.onnx` file in the GitHub repository.
2.  Click the "Download raw file" button.
3.  Copy the URL from your browser's address bar.
4.  On your Nano, run the `wget` command with the copied URL:

```bash
# Example URL - replace with the actual raw file link from your repository
wget [https://raw.githubusercontent.com/YourUsername/movenet-jetson-benchmark/main/movenet_lightning.onnx](https://raw.githubusercontent.com/YourUsername/movenet-jetson-benchmark/main/movenet_lightning.onnx)
```

### Step 2: Maximize Jetson Nano Performance

To get stable and reliable benchmark results, you must first lock the Jetson Nano into its maximum performance state.

1.  **Set 10W Power Mode:**
    ```bash
    sudo nvpmodel -m 0
    ```

2.  **Lock GPU and CPU Clocks:**
    ```bash
    sudo jetson_clocks
    ```
    Your fan will likely spin up to maximum speed. This is normal.

### Step 3: Run the TensorRT Benchmarks

We will use the `trtexec` command-line tool to convert the ONNX model into optimized TensorRT engines and measure their performance. This tool is included with JetPack.

*If you get a `trtexec: command not found` error, run this command first to add it to your PATH:*
`echo 'export PATH="$PATH:/usr/src/tensorrt/bin"' >> ~/.bashrc && source ~/.bashrc`

1.  **Benchmark FP32 Precision (Baseline):**
    This command builds and benchmarks the standard 32-bit floating-point engine.
    ```bash
    trtexec --onnx=movenet_lightning.onnx
    ```

2.  **Benchmark FP16 Precision (Optimized):**
    This command builds and benchmarks the 16-bit floating-point engine, which leverages the Nano's hardware for a significant speed-up.
    ```bash
    trtexec --onnx=movenet_lightning.onnx --fp16
    ```

### Step 4: Interpret the Results

At the end of each `trtexec` run, look for the `=== Performance summary ===` section in the output. The most important number for stable performance is the **`median`** latency under **`GPU Compute Time`**.

```
[I] === Performance summary ===
[I] Throughput: 75.583 qps
[I] Latency: min = 13.1934 ms, max = 23.805 ms, mean = 13.2303 ms, median = 13.2354 ms ...
...
[I] GPU Compute Time: min = 13.1445 ms, max = 23.588 ms, mean = 13.1879 ms, median = 13.1925 ms ...
```

* **Latency:** The `median` GPU Compute Time is your raw inference speed in milliseconds (e.g., **13.19 ms**).
* **FPS:** To calculate the Frames Per Second, use the formula: `FPS = 1000 / median_latency`. (e.g., `1000 / 13.19 = ~75.8 FPS`).

Compare the results from the FP32 and FP16 runs to quantify the performance gain from quantization.

---

## Repository Structure

```
.
├── movenet_lightning.onnx
└── README.md
```
