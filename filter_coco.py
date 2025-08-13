import json
import os
import shutil
from collections import defaultdict

# main COCO val2017 image directory
IMAGE_DIR = './val2017' 
# JSON annotation file
ANNOTATION_FILE = './annotations/person_keypoints_val2017.json'
# new directory to save the filtered images
OUTPUT_DIR = './val2017_pose_only'

# --- Script ---
def filter_images_with_pose_annotations():
    """
    Reads a COCO annotation file and copies images that have
    person keypoint annotations to a new directory.
    """
    print(f"Loading annotations from: {ANNOTATION_FILE}")
    try:
        with open(ANNOTATION_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Annotation file not found at {ANNOTATION_FILE}")
        return

    # 1. Get a set of all image_ids that have person annotations.
    # The 'annotations' list contains an entry for every person instance.
    annotated_image_ids = {ann['image_id'] for ann in data['annotations']}
    
    print(f"Found {len(data['annotations'])} total person instances in {len(annotated_image_ids)} unique images.")

    # 2. Create the output directory if it doesn't exist.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 3. Iterate through all images and copy the relevant ones.
    copied_count = 0
    for image_info in data['images']:
        if image_info['id'] in annotated_image_ids:
            source_path = os.path.join(IMAGE_DIR, image_info['file_name'])
            destination_path = os.path.join(OUTPUT_DIR, image_info['file_name'])
            
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                copied_count += 1

    print(f"\nProcess complete. Copied {copied_count} relevant images to '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    filter_images_with_pose_annotations()