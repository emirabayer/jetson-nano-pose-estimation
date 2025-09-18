# Smoothing on Jetson Nano

Several techniques were used for smoothing the jittery output of the MoveNet single pose estimation model.

## Key Features

This repository provides three distinct inference pipelines for comparison:
* **`MoveNet`:** The baseline, unfiltered output from the TensorRT optimized MoveNet model.
* **`MoveNet + One Euro Filter (OEF)`:** A classic signal processing filter applied to the keypoint data. It's fast and effective but requires manual tuning.
* **`MoveNet + LSTM`:** A custom trained Long Short-Term Memory (LSTM) network that learns the patterns of human motion from data to act as a smart, stateful filter.

<br>
<br>

## Demo & Dataset

A Google Drive folder containing the following can be found here: **[https://drive.google.com/drive/folders/1dSN_O2x4_8f7S8ZCVpMtBMJ9cQy8IGqF?usp=sharing]**

* **`vids` folder:** The video dataset used to train the LSTM model.
* **`outputs` folder:** Sample outputs from the three different pipelines for a direct visual comparison.

---

## How to Replicate This Project

Follow these steps to set up the project and run the different inference pipelines on your own Jetson Nano.

### 1. Prerequisites

* **Hardware:** NVIDIA Jetson Nano (2GB or 4GB model)
* **Software:** NVIDIA JetPack 4.6 or later (includes TensorRT, CUDA, cuDNN)
* **Environment:** A Conda environment with Python 3.10 is recommended for compatibility.

<br>
<br>

### 2. Setup & Installation

**A. Clone the Repository:**
```bash
git clone [https://github.com/emirabayer/jetson-nano-pose-estimation.git](https://github.com/emirabayer/jetson-nano-pose-estimation.git)
cd jetson-nano-pose-estimation
```

**B. Create Conda Environment (Recommended):**
```bash
# Create an environment with Python 3.10
conda create --name pose_env python=3.10 -y

# Activate the environment
conda activate pose_env
```

**C. Install Dependencies:**
```bash
pip install numpy opencv-python
# PyCUDA needs to be installed carefully on Jetson
pip install pycuda
```

<br>
<br>

### 3. Build TensorRT Engines
This project requires two TensorRT engine files (.engine).

1. The engine previously created for the MoveNet Single-Pose Lightning model.
2. LSTM Smoother Engine
   You can either use a pre-trained `.onnx` model or train your own by following the steps below.

   Once you have the `lstm_smoother.onnx`, run `trtexec` to build the engine:
```bash
pip install numpy opencv-python
# PyCUDA needs to be installed carefully on Jetson
pip install pycuda
```

<br>
<br>

### 4. Training Your Own LSTM (Optional)

**A. Generate Training Data:**
1. Create a folder named `vids` and fill it with videos of people moving.
2. Run the data generation script on your Jetson Nano. This will process the videos and create `movenet_raw_keypoints.npy`.
```bash
python generate_data_movenet_output.py
```

**B. Train the LSTM:**
1. On your PC (with the (`pose_env`) conda environment), place the `.npy` file.

2. Run the training script. This will create the `lstm_smoother.onnx` file.
```bash
python train_lstm.py
```

3. Copy the resulting `.onnx` file to your Jetson to build the engine as described above.

<br>
<br>

### 5. Running Inference With LSTM Smoother
Run any of the following scripts on your Jetson Nano to see the results. Make sure your `.engine` files are in the same directory.

```bash
python video_inference_movenet_lstm.py
```
