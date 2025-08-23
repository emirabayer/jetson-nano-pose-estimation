# Jetson Nano Pose Estimation

This repository provides a comprehensive toolkit for benchmarking and analyzing the performance of deep learning based pose estimation models on the NVIDIA Jetson Nano Developer Kit. Developed as part of research at the METU Informatics Institute's AIRLAB, this suite is designed to be a practical resource for edge AI developers and researchers.

The primary focus is on evaluating models optimized with NVIDIA TensorRT, comparing architectures like MoveNet (Single Pose) and YOLOv8n-Pose for speed and accuracy on a standardized dataset.

<br>

<br>


## Features

* Ready to run Python scripts for evaluating _.engine_ files for MoveNet (Single Pose) and Yolov8n-Pose.
* Includes easy to use commands to monitor the Jetson Nano's real time hardware status (temperature, power, clock speeds).
* A detailed guide on how to generate the exact subset of the COCO 2017 dataset used for these benchmarks.
* The structure is designed to easily accommodate new models and benchmark tests.
* DFrom flashing the Jetson Nano to running your first benchmark, all steps are covered.

<br>
<br>


## Prerequisites

* **Hardware:** NVIDIA Jetson Nano 2GB Developer Kit
* **Software:** JetPack 4.6.x (which includes TensorRT 8.2.1 and CUDA 10.2)

<br>
<br>


## Setup and Installation

### Step 0: Initial Jetson Nano Setup

Before cloning this repository, ensure your Jetson Nano 2GB Developer Kit is properly set up. This includes flashing the OS, installing necessary system libraries (like `pip`), and connecting to the internet.

For a fantastic, step-by-step guide from a fellow researcher at AIRLAB, please follow the instructions here:
**➡️ [Jetson Nano Setup Guide by Ali Fırat](http://alifirat.xyz/jetson)**

### Step 1: Clone This Repository

Open a terminal on your Jetson Nano and clone this repository.

```bash
git clone [https://github.com/emirabayer/jetson-nano-pose-estimation.git](https://github.com/emirabayer/jetson-nano-pose-estimation.git)
cd jetson-nano-pose-estimation
```

### Step 2: Install Dependencies

The benchmark scripts rely on several Python libraries. Ensure you have `pip` installed, then install the required packages. It's recommended to use a virtual environment.

```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenjp2-7-dev libtiff-dev

# Install the dependencies
pip3 install -r requirements.txt
```

## Step 3: Prepare the Models

The scripts use TensorRT `.engine` files for optimized inference. You need to generate these from `.onnx` files first.

1.  Obtain the `.onnx` models for MoveNet and YOLOv8-Pose.
2.  Use the `trtexec` command-line tool (included with TensorRT on your Jetson) to convert them. For example:

```bash
trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose_fp32.engine --fp16
```
    * Use `--fp16` for FP16 precision or `--int8` for INT8 precision if you have a calibration dataset.


<br>
<br>


## Benchmarking Procedure

### - Step 1: Download the Model

Download the pre-converted ONNX model from this repository into the terminal.

```bash
wget -O movenet_singlepose_lightning.onnx [https://raw.githubusercontent.com/emirabayer/movenet-jetson-benchmark/main/movenet_singlepose_lightning.onnx](https://raw.githubusercontent.com/emirabayer/movenet-jetson-benchmark/main/movenet_singlepose_lightning.onnx)
```

<br>

<br>

### - Step 2: Maximize Jetson Nano Performance

To get stable and reliable benchmark results, you must first lock the Jetson Nano into its maximum performance state.

1.  **Set 10W Power Mode:**
    ```bash
    sudo nvpmodel -m 0
    ```

2.  **Lock GPU and CPU Clocks:**
    ```bash
    sudo jetson_clocks
    ```


<br>

<br>

### - Step 3: Run the TensorRT Benchmarks

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

### - Step 4: Interpret the Results

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

