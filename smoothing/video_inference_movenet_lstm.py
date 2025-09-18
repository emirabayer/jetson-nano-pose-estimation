import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
import json
from collections import deque

# --- 1. CONFIGURATION ---
MOVENET_ENGINE_PATH = 'movenet_fp16.engine'
LSTM_ENGINE_PATH = 'lstm_smoother.engine'
LSTM_CONFIG_PATH = 'lstm_config.json'
INPUT_VIDEO_PATH = 'demo_video.mp4'
OUTPUT_VIDEO_PATH = 'output_lstm_only_corrected.mp4' # New output name

INPUT_HEIGHT = 192
INPUT_WIDTH = 192

# --- 2. TENSORRT INFERENCE CLASS (No changes here) ---
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
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

# --- 3. HELPER FUNCTIONS (No changes here) ---
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
    return input_tensor, top_pad, left_pad, scale

def draw_keypoints_on_frame(frame, keypoints, color=(0, 255, 0), radius=5):
    for point in keypoints:
        if point[0] > 0 and point[1] > 0:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), radius, color, -1)
    return frame

# --- 4. MAIN VIDEO PROCESSING SCRIPT ---
if __name__ == '__main__':
    for path in [INPUT_VIDEO_PATH, MOVENET_ENGINE_PATH, LSTM_ENGINE_PATH, LSTM_CONFIG_PATH]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'")
            exit()

    print(f"🚀 Loading MoveNet Model: {MOVENET_ENGINE_PATH}")
    movenet_model = TRTInference(MOVENET_ENGINE_PATH)

    print(f"🚀 Loading LSTM Model: {LSTM_ENGINE_PATH}")
    with open(LSTM_CONFIG_PATH, 'r') as f:
        lstm_config = json.load(f)
    SEQUENCE_LENGTH = lstm_config['sequence_length']
    FEATURES_PER_FRAME = lstm_config['features_per_frame']
    
    lstm_smoother = TRTInference(LSTM_ENGINE_PATH)
    history_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    print(f"📹 Processing Video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_width, frame_height))
    
    total_postprocess_time = 0
    total_inference_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor, top_pad, left_pad, scale = preprocess_frame(frame)

        start_time = time.perf_counter()
        movenet_output = movenet_model.infer(input_tensor)
        inference_time = (time.perf_counter() - start_time)
        total_inference_time += inference_time

        post_start_time = time.perf_counter()
        
        keypoints_with_scores = np.squeeze(movenet_output)
        raw_normalized_yx = keypoints_with_scores[:, :2]

        # --- FIX: ALWAYS fill the buffer with RAW data from MoveNet ---
        history_buffer.append(raw_normalized_yx.flatten())
        # -----------------------------------------------------------

        if len(history_buffer) < SEQUENCE_LENGTH:
            # For the first few frames, use the raw MoveNet output
            keypoints_to_draw_normalized_yx = raw_normalized_yx
        else:
            # Buffer is full, run the LSTM on the history of RAW points
            lstm_input = np.array(history_buffer, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
            lstm_result = lstm_smoother.infer(lstm_input)
            
            # The LSTM output is the final, smoothed point for DRAWING ONLY
            keypoints_to_draw_normalized_yx = lstm_result.reshape((keypoints_with_scores.shape[0], 2))

        # --- Denormalize the final keypoints for drawing ---
        output_xy = keypoints_to_draw_normalized_yx[:, ::-1] * [INPUT_WIDTH, INPUT_HEIGHT]
        final_keypoints_xy = (output_xy - [left_pad, top_pad]) / scale
        
        postprocess_time = (time.perf_counter() - post_start_time)
        total_postprocess_time += postprocess_time
        
        frame_with_keypoints = draw_keypoints_on_frame(frame.copy(), final_keypoints_xy)
        out.write(frame_with_keypoints)
        
        frame_count += 1
        print(f"Processed frame {frame_count}... Inf: {inference_time*1000:.1f}ms, Post: {postprocess_time*1000:.1f}ms", end='\r')

    cap.release()
    out.release()

    print(f"\n✅ Processing complete. Output saved to: {OUTPUT_VIDEO_PATH}")

    if frame_count > 0:
        avg_inference_time_ms = (total_inference_time / frame_count) * 1000
        avg_postprocess_time_ms = (total_postprocess_time / frame_count) * 1000
        total_time_ms = avg_inference_time_ms + avg_postprocess_time_ms
        avg_fps = 1000 / total_time_ms if total_time_ms > 0 else 0

        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Total Frames: {frame_count}")
        print(f"Average Inference Time (GPU - MoveNet): {avg_inference_time_ms:.2f} ms")
        print(f"Average Post-processing Time (GPU - LSTM): {avg_postprocess_time_ms:.2f} ms")
        print("---------------------------------")
        print(f"Total Time Per Frame: {total_time_ms:.2f} ms")
        print(f"Resulting FPS: {avg_fps:.2f}")
    else:
        print("No frames were processed.")