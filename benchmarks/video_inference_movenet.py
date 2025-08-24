import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = 'movenet_fp16.engine'
INPUT_VIDEO_PATH = 'demo_video.mp4'
OUTPUT_VIDEO_PATH = 'output_movenet.mp4'

# MoveNet's specific input size
INPUT_HEIGHT = 192
INPUT_WIDTH = 192

# --- 2. TENSORRT INFERENCE CLASS ---
class TRTInference:
    """A class for performing inference with a TensorRT engine."""
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
        """Runs inference on a single image."""
        np.copyto(self.inputs[0]['host'], input_image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        output_shape = self.engine.get_binding_shape(1)
        return self.outputs[0]['host'].reshape(output_shape)

# --- 3. HELPER FUNCTIONS ---
def preprocess_frame(frame):
    """Preprocesses a single video frame for MoveNet."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # Calculate scaling and padding
    scale = min(INPUT_HEIGHT / h, INPUT_WIDTH / w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    
    top_pad = (INPUT_HEIGHT - resized_h) // 2
    left_pad = (INPUT_WIDTH - resized_w) // 2
    
    padded_img = cv2.copyMakeBorder(
        resized_img, top_pad, INPUT_HEIGHT - resized_h - top_pad, 
        left_pad, INPUT_WIDTH - resized_w - left_pad, 
        cv2.BORDER_CONSTANT, value=0
    )
    
    # Finalize tensor
    input_tensor = np.expand_dims(padded_img, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.int32)
    
    return input_tensor, top_pad, left_pad, scale

def draw_keypoints_on_frame(frame, keypoints, top_pad, left_pad, scale, color=(0, 0, 255), radius=5):
    """Draws the detected keypoints onto the original video frame."""
    num_keypoints = keypoints.shape[1]
    
    for i in range(num_keypoints):
        # The model's output is normalized to the input size (192x192)
        y = keypoints[0, i, 0] * INPUT_HEIGHT
        x = keypoints[0, i, 1] * INPUT_WIDTH
        
        # We need to scale the coordinates back to the original frame size
        orig_x = int((x - left_pad) / scale)
        orig_y = int((y - top_pad) / scale)
        
        cv2.circle(frame, (orig_x, orig_y), radius, color, -1)
    
    return frame

# --- 4. MAIN VIDEO PROCESSING SCRIPT ---
if __name__ == '__main__':
    # Check if input video and model exist
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
    
    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_width, frame_height))

    total_inference_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for the model
        input_tensor, top_pad, left_pad, scale = preprocess_frame(frame)

        # Run inference
        start_time = time.perf_counter()
        outputs = trt_model.infer(input_tensor)
        end_time = time.perf_counter()
        
        total_inference_time += (end_time - start_time)
        frame_count += 1
        
        # Squeeze output if necessary (some models have an extra dimension)
        if outputs.ndim == 4 and outputs.shape[1] == 1:
            outputs = np.squeeze(outputs, axis=1)

        # Draw predictions on the original frame
        frame_with_keypoints = draw_keypoints_on_frame(frame.copy(), outputs, top_pad, left_pad, scale)
        
        # Write the processed frame to the output video
        out.write(frame_with_keypoints)
        
        print(f"Processed frame {frame_count}...", end='\r')

    # Release resources
    cap.release()
    out.release()

    print(f"\n✅ Processing complete. Output saved to: {OUTPUT_VIDEO_PATH}")

    # --- 5. RESULTS ---
    if frame_count > 0:
        avg_inference_time_ms = (total_inference_time / frame_count) * 1000
        avg_fps = frame_count / total_inference_time

        print("\n--- INFERENCE SPEED ---")
        print(f"Total Frames: {frame_count}")
        print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
    else:
        print("No frames were processed.")

