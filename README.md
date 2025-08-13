# MoveNet Performance Benchmark on Jetson Nano

This repository contains the resources and instructions to simulate the performance of the MoveNet Single Pose Lightning model on an NVIDIA Jetson Nano 2GB Developer Kit using the `trtexec` command-line tool of TensorRT.

The goal is to measure the raw inference speed (latency and FPS) of the model after optimization with NVIDIA TensorRT.

<br>


<br>

## Prerequisites

* **Hardware:** NVIDIA Jetson Nano 2GB Developer Kit
* **Software:** JetPack 4.6.x (which includes TensorRT 8.2.1 and CUDA 10.2)

<br>
<br>


## Benchmarking Procedure

### Step 1: Download the Model

Download the pre-converted ONNX model from this repository into the terminal.

```bash
wget -O movenet_singlepose_lightning.onnx [https://raw.githubusercontent.com/emirabayer/movenet-jetson-benchmark/main/movenet_singlepose_lightning.onnx](https://raw.githubusercontent.com/emirabayer/movenet-jetson-benchmark/main/movenet_singlepose_lightning.onnx)
```

<br>

<br>

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


<br>

<br>

### Step 3: Run the TensorRT Benchmarks

We will use the `trtexec` command-line tool to convert the ONNX model into optimized TensorRT engines and measure their performance. This tool is included with JetPack.
*If you get a `trtexec: command not found` error, run this command first to add it to your PATH:*
`echo 'export PATH="$PATH:/usr/src/tensorrt/bin"' >> ~/.bashrc && source ~/.bashrc`

<br>
<br>

1. **Benchmark FP32 Precision (Baseline):**
    ```bash
    trtexec --onnx=movenet_singlepose_lightning.onnx
    ```

2.  **Benchmark FP16 Precision (Optimized):**
    ```bash
    trtexec --onnx=movenet_singlepose_lightning.onnx --fp16
    ```

<br>

<br>

### Step 4: Interpret the Results

At the end of each `trtexec` run, look for the `=== Performance summary ===` section in the output. The most important number for stable performance is the **`median`** latency under **`GPU Compute Time`**.

```
[I] === Performance summary ===
[I] Throughput: 75.583 qps
[I] Latency: min = 13.1934 ms, max = 23.805 ms, mean = 13.2303 ms, median = 13.2354 ms ...
...
[I] GPU Compute Time: min = 13.1445 ms, max = 23.588 ms, mean = 13.1879 ms, median = 13.1925 ms ...
```

<br>
<br>

* **Latency:** The `median` GPU Compute Time is your raw inference speed in milliseconds (e.g., **13.19 ms**).
* **FPS:** To calculate the Frames Per Second, use the formula: `FPS = 1000 / median_latency`. (e.g., `1000 / 13.19 = ~75.8 FPS`).

Compare the results from the FP32 and FP16 runs to quantify the performance gain from quantization.


<br>
<br>

## Decreasing The Size of COCO Validation Set

The `trtexec` tool benchmarks performance using random data. The included `filter_coco.py` script is a utility to create a smaller, focused dataset from the full COCO 2017 validation set.
Using it on the COCO dataset will create a new folder named `val2017_pose_only` containing only the images with person keypoint annotations, which is ideal for realistic benchmarking.

