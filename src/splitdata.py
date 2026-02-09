import os
import shutil
import random

#dirrectories
RAW_DATA_DIR = os.path.join("data","raw")
SPLIT_DATA_DIR = os.path.join("data","datasets")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def setup_directories():
    for category in ['images', 'labels']:
        for split in ['train', 'val']:
            dir_path = os.path.join(project_root, SPLIT_DATA_DIR, category, split)
            os.makedirs(dir_path, exist_ok=True)

def split_data(files, split):
    for file in files:
        base_name = os.path.basename(file)[0]
        category = "heineken_can" if "Heineken_Can" in file else "purdeys_can"

        src_image_path = os.path.join(RAW_DATA_DIR, file)
        src_label_path = os.path.join(RAW_DATA_DIR, base_name + ".txt")

        dst_img = os.path.join(SPLIT_DATA_DIR,'images', split, file)
        dst_label = os.path.join(SPLIT_DATA_DIR,'labels', split, base_name + ".txt")

        shutil.copy(src_image_path, dst_img)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label)



def main():
    setup_directories()

    images = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.jpg')]
    random.shuffle(images)


    split_index = int(len(images) * 0.8)
    train_files = images[:split_index]
    val_files = images[split_index:]

    split_data(train_files, 'train')
    split_data(val_files, 'val')

if __name__ == "__main__":
    main()

