import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import json
import time
import os
from math import sqrt

# --- 1. CONFIGURATION ---
IMAGE_DIR = './nano_benchmark_set'
ANNOTATION_FILE = 'person_keypoints_val2017.json'
MODEL_PATH = 'movenet_fp32.engine'
VISUALIZATION_DIR = './visualizations'
IMAGES_TO_VISUALIZE = 5

INPUT_HEIGHT = 192
INPUT_WIDTH = 192

# --- 2. TENSORRT INFERENCE CLASS ---
class TRTInference:
    
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
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
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None, 0, 0, 0, 0, 0, 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    scale = min(INPUT_HEIGHT / h, INPUT_WIDTH / w) if h > 0 and w > 0 else 0
    if scale == 0: return None, None, 0, 0, 0, 0, 0, 0
    resized_h, resized_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    top_pad = (INPUT_HEIGHT - resized_h) // 2
    bottom_pad = INPUT_HEIGHT - resized_h - top_pad
    left_pad = (INPUT_WIDTH - resized_w) // 2
    right_pad = INPUT_WIDTH - resized_w - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    input_tensor = np.expand_dims(padded_img, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.int32)
    return input_tensor, img, h, w, top_pad, left_pad, scale

def draw_keypoints(image, keypoints, color):
    """Verilen koordinatları resmin üzerine çizer."""
    for i in range(17):
        x, y = keypoints[i]
        if x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
    return image

# --- 4. MAIN BENCHMARKING SCRIPT ---
if __name__ == '__main__':
    if not os.path.exists(VISUALIZATION_DIR):
        os.makedirs(VISUALIZATION_DIR)
        print(f"Created visualization directory: {VISUALIZATION_DIR}")

    print(f"--- Benchmarking {os.path.basename(MODEL_PATH)} ---")
    
    trt_model = TRTInference(MODEL_PATH)
    
    with open(ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
    annotations_map = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if ann.get('num_keypoints', 0) > 0 and not ann.get('iscrowd', False):
            if image_id not in annotations_map:
                annotations_map[image_id] = []
            annotations_map[image_id].append(ann)

    total_inference_time = 0
    total_error = 0
    image_count_speed = 0
    image_count_error = 0
    visualized_count = 0
    
    test_image_files = sorted(os.listdir(IMAGE_DIR))
    print(f"Found {len(test_image_files)} images to test.")

    # Warmup run
    if test_image_files:
        warmup_tensor, _, _, _, _, _, _ = preprocess_image(os.path.join(IMAGE_DIR, test_image_files[0]))
        if warmup_tensor is not None:
            _ = trt_model.infer(warmup_tensor)

    for filename in test_image_files:
        image_id = int(os.path.splitext(filename)[0])
        if image_id not in annotations_map:
            continue
            
        image_path = os.path.join(IMAGE_DIR, filename)
        input_tensor, original_image, h, w, top_pad, left_pad, scale = preprocess_image(image_path)
        
        if input_tensor is None:
            continue
        
        start_time = time.perf_counter()
        outputs = trt_model.infer(input_tensor)
        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)
        image_count_speed += 1
        
        try:
            if outputs.ndim == 4 and outputs.shape[1] == 1:
                outputs = np.squeeze(outputs, axis=1)
            predicted_keypoints = outputs
            person_annotations = sorted(annotations_map[image_id], key=lambda x: x['area'], reverse=True)
            if person_annotations:
                main_person = person_annotations[0]
                gt_keypoints_raw = main_person['keypoints']
                bbox = main_person['bbox']
                pred_y = predicted_keypoints[0, :, 0]
                pred_x = predicted_keypoints[0, :, 1]
                pred_x_pad = pred_x * INPUT_WIDTH
                pred_y_pad = pred_y * INPUT_HEIGHT
                orig_pred_x = (pred_x_pad - left_pad) / scale
                orig_pred_y = (pred_y_pad - top_pad) / scale
                total_normalized_dist = 0
                visible_keypoints_count = 0
                
                gt_coords_for_drawing = []
                pred_coords_for_drawing = []

                for i in range(17):
                    gt_x, gt_y, gt_v = float(gt_keypoints_raw[i*3]), float(gt_keypoints_raw[i*3+1]), gt_keypoints_raw[i*3+2]
                    pred_x_scalar = float(np.array(orig_pred_x[i]).item())
                    pred_y_scalar = float(np.array(orig_pred_y[i]).item())
                    
                    pred_coords_for_drawing.append((pred_x_scalar, pred_y_scalar))
                    
                    if gt_v > 0:
                        gt_coords_for_drawing.append((gt_x, gt_y))
                        distance = sqrt((gt_x - pred_x_scalar)**2 + (gt_y - pred_y_scalar)**2)
                        bbox_scale = sqrt(bbox[2]**2 + bbox[3]**2)
                        if bbox_scale > 0:
                            total_normalized_dist += (distance / bbox_scale)
                        visible_keypoints_count += 1
                    else:
                        gt_coords_for_drawing.append((0,0))
                
                if visible_keypoints_count > 0:
                    total_error += (total_normalized_dist / visible_keypoints_count)
                    image_count_error += 1
                    
                if visualized_count < IMAGES_TO_VISUALIZE:
                    vis_image = draw_keypoints(original_image.copy(), gt_coords_for_drawing, (0, 255, 0))
                    vis_image = draw_keypoints(vis_image, pred_coords_for_drawing, (0, 0, 255))
                    
                    save_path = os.path.join(VISUALIZATION_DIR, f"result_{filename}")
                    cv2.imwrite(save_path, vis_image)
                    visualized_count += 1
                    print(f"Saved visualization to {save_path}")

        except Exception:
            pass

    # --- 5. RESULTS ---
    if image_count_speed > 0:
        avg_inference_time_ms = (total_inference_time / image_count_speed) * 1000
        avg_fps = 1000 / avg_inference_time_ms
        avg_error = total_error / image_count_error if image_count_error > 0 else 0

        print("\n--- BENCHMARK COMPLETE ---")
        print(f"Model: {os.path.basename(MODEL_PATH)}")
        print(f"Images processed for speed: {image_count_speed}")
        print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
        print("---")
        print(f"Images successfully processed for error: {image_count_error}")
        print(f"Average Normalized Error (on successful images): {avg_error:.4f}")
    else:
        print("No images were processed. Check image paths.")

