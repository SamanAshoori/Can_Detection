import os
import shutil
import random

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset')

def setup_dirs():
    # Create YOLO structure
    for category in ['images', 'labels']:
        for split in ['train', 'val']:
            dir_path = os.path.join(DATASET_DIR, category, split)
            os.makedirs(dir_path, exist_ok=True)
            # clear existing files to avoid duplicates if re-running
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))

def move_files(files, split):
    for filename in files:
        base_name = os.path.splitext(filename)[0]
        
        src_img = os.path.join(RAW_DATA_DIR, filename)
        src_label = os.path.join(RAW_DATA_DIR, base_name + '.txt')
        
        dst_img = os.path.join(DATASET_DIR, 'images', split, filename)
        dst_label = os.path.join(DATASET_DIR, 'labels', split, base_name + '.txt')
        
        shutil.copy(src_img, dst_img)
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

def main():
    setup_dirs()
    images = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.jpg', '.jpeg'))]
    random.shuffle(images)
    
    # 80/20 Split
    split_idx = int(len(images) * 0.8)
    train_files = images[:split_idx]
    val_files = images[split_idx:]
    
    print(f"Processing {len(images)} images...")
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    print(f"Done! {len(train_files)} training, {len(val_files)} validation.")

if __name__ == "__main__":
    main()