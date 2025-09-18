import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
import glob

# --- 1. CONFIGURATION ---
MODEL_PATH = 'movenet_fp16.engine'
INPUT_VIDEO_FOLDER = 'vids' 
OUTPUT_NUMPY_FILE = 'movenet_raw_keypoints.npy'

INPUT_HEIGHT = 192
INPUT_WIDTH = 192

# --- 2. TENSORRT INFERENCE CLASS (From your working script) ---
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    def infer(self, input_image):
        np.copyto(self.inputs[0]['host'], input_image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        output_shape = self.engine.get_binding_shape(1)
        return self.outputs[0]['host'].reshape(output_shape)

# --- 3. HELPER FUNCTIONS ---
def preprocess_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    scale = min(INPUT_HEIGHT / h, INPUT_WIDTH / w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    top_pad = (INPUT_HEIGHT - resized_h) // 2
    left_pad = (INPUT_WIDTH - resized_w) // 2
    padded_img = cv2.copyMakeBorder(
        resized_img, top_pad, INPUT_HEIGHT - resized_h - top_pad,
        left_pad, INPUT_WIDTH - resized_w - left_pad,
        cv2.BORDER_CONSTANT, value=0)
    input_tensor = np.expand_dims(padded_img, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.int32)
    return input_tensor

# --- 4. MAIN DATA GENERATION SCRIPT ---
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        exit(f"Error: Model not found at '{MODEL_PATH}'")
    if not os.path.isdir(INPUT_VIDEO_FOLDER):
        exit(f"Error: Input folder not found at '{INPUT_VIDEO_FOLDER}'")

    video_paths = glob.glob(os.path.join(INPUT_VIDEO_FOLDER, '*.mp4'))
    video_paths += glob.glob(os.path.join(INPUT_VIDEO_FOLDER, '*.mov'))
    
    if not video_paths:
        exit(f"No video files found in '{INPUT_VIDEO_FOLDER}'")

    print(f"🚀 Loading Model: {MODEL_PATH}")
    trt_model = TRTInference(MODEL_PATH)
    all_keypoints = []

    for video_path in video_paths:
        print(f"\n📹 Processing Video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            input_tensor = preprocess_frame(frame)
            outputs = trt_model.infer(input_tensor)
            
            # Get the (y,x) coordinates
            raw_yx_coords = np.squeeze(outputs)[:, :2]
            
            all_keypoints.append(raw_yx_coords.copy())
            # --------------------------------
            
            frame_count += 1
            print(f"  > Extracted frame {frame_count}...", end='\r')
        cap.release()
    
    # Save the final, correct data
    if all_keypoints:
        training_data = np.array(all_keypoints)
        print(f"\n\n✅ Success! Saving {training_data.shape[0]} frames of correct keypoint data.")
        np.save(OUTPUT_NUMPY_FILE, training_data)
        print(f"Output saved to: {OUTPUT_NUMPY_FILE}")
    else:
        print("\nNo keypoints were extracted.")