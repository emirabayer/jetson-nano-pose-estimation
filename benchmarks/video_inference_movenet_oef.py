import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
from one_euro_filter_np import OneEuroFilter

# --- 1. CONFIGURATION ---
MODEL_PATH = 'movenet_fp16.engine'
INPUT_VIDEO_PATH = 'demo_video.mp4'
OUTPUT_VIDEO_PATH = 'output_movenet_vectorized.mp4'

# MoveNet's specific input size
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
    return input_tensor, top_pad, left_pad, scale

# --- CHANGE IS HERE ---
def draw_keypoints_on_frame(frame, keypoints, color=(0, 255, 0), radius=5):
    # This function now correctly interprets the (y, x) format
    for point in keypoints:
        # point[0] is the y-coordinate, point[1] is the x-coordinate
        y, x = int(point[0]), int(point[1])
        
        # We pass them to cv2.circle in the correct (x, y) order
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame
# --------------------

# --- 4. MAIN VIDEO PROCESSING SCRIPT ---
if __name__ == '__main__':
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at '{INPUT_VIDEO_PATH}'")
        exit()
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
        exit()

    print(f"🚀 Loading Model: {MODEL_PATH}")
    trt_model = TRTInference(MODEL_PATH)
    
    print(f"📹 Processing Video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_width, frame_height))

    # --- Initialize One Euro Filter ---
    filter_config = {
        'freq': video_fps,
        'min_cutoff': 0.4,
        'beta': 0.05,
        'd_cutoff': 1.0
    }
    keypoint_filter = OneEuroFilter(**filter_config)
    # ----------------------------------------
    
    total_postprocess_time = 0
    total_inference_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor, top_pad, left_pad, scale = preprocess_frame(frame)

        start_time = time.perf_counter()
        outputs = trt_model.infer(input_tensor)
        inference_time = (time.perf_counter() - start_time)
        total_inference_time += inference_time

        # --- Post-processing starts here ---
        post_start_time = time.perf_counter()
        
        keypoints = np.squeeze(outputs)

        # Vectorized scaling of all keypoints at once. Result is in (y, x) order.
        raw_xy = keypoints[:, :2] * [INPUT_HEIGHT, INPUT_WIDTH]
        orig_xy = (raw_xy - [top_pad, left_pad]) / scale

        # Apply filter. Result is still in (y, x) order.
        current_time = frame_count / video_fps
        smoothed_keypoints = keypoint_filter(orig_xy, timestamp=current_time)

        postprocess_time = (time.perf_counter() - post_start_time)
        total_postprocess_time += postprocess_time
        # --- Post-processing ends here ---
        
        # --- CHANGE IS HERE ---
        # Draw and write frame. No need to swap columns anymore.
        frame_with_keypoints = draw_keypoints_on_frame(frame.copy(), smoothed_keypoints)
        out.write(frame_with_keypoints)
        # --------------------
        
        frame_count += 1
        print(f"Processed frame {frame_count}... Inf: {inference_time*1000:.1f}ms, Post: {postprocess_time*1000:.1f}ms", end='\r')

    cap.release()
    out.release()

    print(f"\n✅ Processing complete. Output saved to: {OUTPUT_VIDEO_PATH}")

    # --- 5. RESULTS ---
    if frame_count > 0:
        avg_inference_time_ms = (total_inference_time / frame_count) * 1000
        avg_postprocess_time_ms = (total_postprocess_time / frame_count) * 1000
        total_time_ms = avg_inference_time_ms + avg_postprocess_time_ms
        avg_fps = 1000 / total_time_ms

        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Total Frames: {frame_count}")
        print(f"Average Inference Time (GPU): {avg_inference_time_ms:.2f} ms")
        print(f"Average Post-processing Time (CPU): {avg_postprocess_time_ms:.2f} ms")
        print("---------------------------------")
        print(f"Total Time Per Frame: {total_time_ms:.2f} ms")
        print(f"Resulting FPS: {avg_fps:.2f}")
    else:
        print("No frames were processed.")