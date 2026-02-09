import os
import shutil
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset')

def reset_dataset_dirs():
    """
    Wipes the existing dataset directory to prevent class mixing.
    Then recreates the folder structure.
    """
    if os.path.exists(DATASET_DIR):
        print(f"Cleaning old data from {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)
    
    # Create YOLO structure
    for category in ['images', 'labels']:
        for split in ['train', 'val']:
            dir_path = os.path.join(DATASET_DIR, category, split)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")

def move_files(files, split):
    """
    Moves files from raw to the specific split (train or val).
    """
    for filename in files:
        base_name = os.path.splitext(filename)[0]
        
        # Source paths
        src_img = os.path.join(RAW_DATA_DIR, filename)
        src_label = os.path.join(RAW_DATA_DIR, base_name + '.txt')
        
        # Destination paths
        dst_img = os.path.join(DATASET_DIR, 'images', split, filename)
        dst_label = os.path.join(DATASET_DIR, 'labels', split, base_name + '.txt')
        
        # Copy Image
        shutil.copy(src_img, dst_img)
        
        # Copy Label (Crucial: Only copy if it exists)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: No label found for {filename} (Assuming background)")

def main():
    print("--- Starting Dataset Split ---")
    reset_dataset_dirs()
    
    # Get all valid image files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(valid_extensions)]
    
    # Shuffle to ensure random distribution
    random.shuffle(images)
    
    # 80/20 Split
    split_index = int(len(images) * 0.8)
    train_files = images[:split_index]
    val_files = images[split_index:]
    
    print(f"\nFound {len(images)} total images.")
    print(f"Training Set:   {len(train_files)} images")
    print(f"Validation Set: {len(val_files)} images")
    
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    
    print("\n Success! Data split and moved.")

if __name__ == "__main__":
    main()