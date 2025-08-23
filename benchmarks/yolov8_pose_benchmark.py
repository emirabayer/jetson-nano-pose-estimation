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
MODEL_PATH = 'yolov8n-pose_fp32.engine'
VISUALIZATION_DIR = './visualizations_yolo'
IMAGES_TO_VISUALIZE = 5

INPUT_HEIGHT = 640
INPUT_WIDTH = 640

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
def preprocess_yolo(img_path):
    """YOLOv8 için bir resmi yükler ve ön işler."""
    original_img = cv2.imread(img_path)
    if original_img is None: return None, None, 0, 0, 0, 0, 0
    
    orig_h, orig_w, _ = original_img.shape
    
    scale = min(INPUT_WIDTH / orig_w, INPUT_HEIGHT / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized_img = cv2.resize(original_img, (new_w, new_h))
    
    top_pad = (INPUT_HEIGHT - new_h) // 2
    left_pad = (INPUT_WIDTH - new_w) // 2
    
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, (INPUT_HEIGHT - new_h - top_pad), left_pad, (INPUT_WIDTH - new_w - left_pad), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    input_tensor = padded_img.transpose(2, 0, 1)
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
    input_tensor /= 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, original_img, orig_h, orig_w, top_pad, left_pad, scale

def postprocess_yolo(output, top_pad, left_pad, scale):
    """YOLOv8'in çıktısını işler ve keypoint'leri çıkarır."""
    output = output.transpose(0, 2, 1)
    
    if output.shape[1] == 0: return None
    
    best_detection_idx = np.argmax(output[0, :, 4])
    best_detection = output[0, best_detection_idx, :]
    
    confidence = best_detection[4]
    if confidence < 0.25: return None

    keypoints = best_detection[5:].reshape(17, 3)
    
    pred_x = (keypoints[:, 0] - left_pad) / scale
    pred_y = (keypoints[:, 1] - top_pad) / scale
    pred_v = keypoints[:, 2]
    
    return np.stack([pred_x, pred_y, pred_v], axis=-1)

def draw_keypoints(image, keypoints, color):
    for i in range(17):
        x, y = keypoints[i, 0], keypoints[i, 1]
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
        warmup_tensor, _, _, _, _, _, _ = preprocess_yolo(os.path.join(IMAGE_DIR, test_image_files[0]))
        if warmup_tensor is not None:
            _ = trt_model.infer(warmup_tensor)

    for filename in test_image_files:
        image_id = int(os.path.splitext(filename)[0])
        if image_id not in annotations_map:
            continue
            
        image_path = os.path.join(IMAGE_DIR, filename)
        input_tensor, original_image, orig_h, orig_w, top_pad, left_pad, scale = preprocess_yolo(image_path)
        
        if input_tensor is None: continue
        
        start_time = time.perf_counter()
        outputs = trt_model.infer(input_tensor)
        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)
        image_count_speed += 1
        
        predicted_keypoints = postprocess_yolo(outputs, top_pad, left_pad, scale)
        
        if predicted_keypoints is not None:
            person_annotations = sorted(annotations_map[image_id], key=lambda x: x['area'], reverse=True)
            if person_annotations:
                main_person = person_annotations[0]
                gt_keypoints_raw = main_person['keypoints']
                bbox = main_person['bbox']
                
                gt_keypoints_for_drawing = np.array(gt_keypoints_raw).reshape(17, 3)
                
                total_normalized_dist = 0
                visible_keypoints_count = 0

                for i in range(17):
                    gt_x, gt_y, gt_v = gt_keypoints_for_drawing[i]
                    if gt_v > 0:
                        pred_x, pred_y, pred_v = predicted_keypoints[i]
                        distance = sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                        bbox_scale = sqrt(bbox[2]**2 + bbox[3]**2)
                        if bbox_scale > 0:
                            total_normalized_dist += (distance / bbox_scale)
                        visible_keypoints_count += 1
                
                if visible_keypoints_count > 0:
                    total_error += (total_normalized_dist / visible_keypoints_count)
                    image_count_error += 1

                if visualized_count < IMAGES_TO_VISUALIZE:
                    vis_image = draw_keypoints(original_image.copy(), gt_keypoints_for_drawing, (0, 255, 0))
                    vis_image = draw_keypoints(vis_image, predicted_keypoints, (0, 0, 255))
                    
                    save_path = os.path.join(VISUALIZATION_DIR, f"yolo_result_{filename}")
                    cv2.imwrite(save_path, vis_image)
                    visualized_count += 1
                    print(f"Saved visualization to {save_path}")

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
        print("No images were processed.")

