# Jetson Nano Pose Estimation

This repository provides a comprehensive toolkit for benchmarking and analyzing the performance of deep learning based pose estimation models on the NVIDIA Jetson Nano Developer Kit. Developed as part of research at the METU Informatics Institute's AIRLAB, this suite is designed to be a practical resource for edge AI developers and researchers.

The primary focus is on evaluating models optimized with NVIDIA TensorRT, comparing architectures like MoveNet (Single Pose) and YOLOv8n-Pose for speed and accuracy on a standardized dataset.

<br>

<br>

![result_000000000785](https://github.com/user-attachments/assets/cfa74320-eb4b-47fc-b414-72d6bfbe694a)

<img width="999" height="104" alt="image" src="https://github.com/user-attachments/assets/8913b5d7-1817-4a1e-902e-97506408edb9" />



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


## 🛠️ Setup and Installation

### Step 0: Initial Jetson Nano Setup

Before cloning this repository, ensure your Jetson Nano 2GB Developer Kit is properly set up. This includes flashing the OS, installing necessary system libraries (like `pip`), and connecting to the internet.

For a fantastic, step by step guide from a fellow researcher at AIRLAB, please follow the instructions here:
**➡[Jetson Nano Setup Guide by Ali Fırat](http://alifirat.xyz/jetson)**


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
```
The second command below adds an environment variable to the terminal's configuration that fixes the low level bug `Illegal instruction (core dumped)` in this version of NumPy on ARM processors.
```bash
# Install the dependencies
python3 -m pip install numpy==1.19.5 opencv-python-headless==4.5.5.64 pycuda
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc && source ~/.bashrc
```


## Step 3: Prepare the Models

The scripts use TensorRT `.engine` files for optimized inference. You need to generate these from `.onnx` files first.

1.  Obtain the `.onnx` models for MoveNet and YOLOv8-Pose.
2.  Use the `trtexec` command line tool (included with TensorRT on your Jetson) to convert them.

```bash
trtexec --onnx=movenet.onnx --saveEngine=movenet_fp16.engine --fp16
trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose_fp32.engine --fp16
```
Use `--fp16` for FP16 precision or `--int8` for INT8 precision if you have a calibration dataset.


<br>

<br>


## 📦 Dataset Preparation

The benchmarks run on a specific subset of the **COCO 2017 Validation** dataset to ensure consistent results. Due to its size, the dataset is not included in this repository.

To generate the exact dataset used:
1.  **Download COCO 2017:**
    * Download the [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) (1GB).
    * Download the [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (241MB).
2.  **Filter the Dataset:**
    * Unzip both files.
    * We need to create a subset containing only images with single person keypoint annotations. You can use a script for this.
    * The `person_keypoints_val2017.json` annotation file will be used to identify the relevant images.
3.  **Organize Files:**
    * Create a directory named `nano_benchmark_set` inside the `dataset/` folder.
    * Copy the filtered images into this new directory.
    * Ensure the `person_keypoints_val2017.json` file is placed in the root directory of the repository or update the path in the scripts.

*(You should place your detailed script/instructions on how you created the dataset inside the `dataset/README.md` file).*

<br>

<br>


## 📊 Running the Benchmarks

Once the setup is complete, running the benchmarks is straightforward.

### 1. Run the MoveNet Benchmark

```bash
python3 benchmarks/movenet_benchmark.py
```

### 2. Run the YOLOv8-Pose Benchmark

```bash
python3 benchmarks/yolov8_pose_benchmark.py
```

The scripts will print the average inference time and FPS to the console. Visualization images, comparing model predictions (red) against ground truth (green), will be saved in the respective `visualizations_*` directories.

<br>

<br>


## 🔬 Monitoring Jetson Nano Performance

To understand the hardware load during benchmarking, you can monitor the device's status. Open a new terminal window and use these commands while a script is running.

#### Check Temperature
The GPU and CPU share a thermal zone. Values are in millidegrees Celsius (divide by 1000).
```bash
cat /sys/devices/virtual/thermal/thermal_zone0/temp
```

#### Check Power Mode
The Jetson Nano has a 5W mode (Mode 1) and a 10W mode (Mode 0, MAXN). For maximum performance, use 10W mode and max clock speed.
```bash
# Check current mode
sudo nvpmodel -q

# Set to 10W mode
sudo nvpmodel -m 0

# Set max clock speed
sudo jetson_clocks
```

#### Check CPU Clock Frequency
See the current frequency of each CPU core.
```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

#### Check GPU Clock Frequency
See the current frequency of the GPU.
```bash
sudo cat /sys/kernel/debug/clk/gpcclk/clk_rate
```

<br>
<br>


## Appendix: Decreasing The Size of COCO Validation Set

To shrink the COCO dataset down to only images with person keypoint annotations, run the `filter_coco.py` script. It will create a new folder named `val2017_pose_only` containing just the filtered images.

