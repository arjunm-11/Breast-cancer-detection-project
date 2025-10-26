import os, shutil, random
import pandas as pd

# ----- CONFIG -----
JPEG_ROOT = "data/raw/archive/jpeg"              # Where all deep subfolders with images are
CSV_FILES = [
    "data/raw/archive/csv/mass_case_description_train_set.csv",
    "data/raw/archive/csv/mass_case_description_test_set.csv",
    "data/raw/archive/csv/calc_case_description_train_set.csv",
    "data/raw/archive/csv/calc_case_description_test_set.csv"
]
OUTPUT_FLAT = "data/preprocessed_all"
OUTPUT_SORTED = "data/preprocessed"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
TRAIN_PCT = 0.7
VAL_PCT = 0.15

# ----- 1. FLATTEN ALL IMAGES -----
os.makedirs(OUTPUT_FLAT, exist_ok=True)
print("Flattening all images in:", JPEG_ROOT)
for root, dirs, files in os.walk(JPEG_ROOT):
    for f in files:
        if f.endswith('.jpg') or f.endswith('.png'):
            src = os.path.join(root, f)
            dst = os.path.join(OUTPUT_FLAT, f)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
print("Done flattening.")

# ----- 2. SORT IMAGES INTO BENIGN AND MALIGNANT -----
os.makedirs(f"{OUTPUT_SORTED}/benign", exist_ok=True)
os.makedirs(f"{OUTPUT_SORTED}/malignant", exist_ok=True)
# Create a mapping: filename -> label
filename_to_label = {}
for csv_file in CSV_FILES:
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        # The filename may be the last item of 'image file path', e.g. "1.3.6.1.4...jpg"
        if 'image file path' in row:
            filename = row['image file path'].split('/')[-1].replace('.dcm', '.jpg') # change to .png if needed
        else:
            continue
        label = row.get('pathology', row.get('abnormality type', None))
        if label is None:
            continue
        label = label.strip().lower()
        if label not in ['benign', 'malignant']:
            continue
        filename_to_label[filename] = label

# Move images into correct folders
print("Sorting images into benign/malignant...")
for fname, label in filename_to_label.items():
    src = os.path.join(OUTPUT_FLAT, fname)
    if os.path.exists(src):
        dst = os.path.join(OUTPUT_SORTED, label, fname)
        shutil.copy(src, dst)
print("Done sorting.")

# ----- 3. SPLIT INTO TRAIN/VAL/TEST -----
def split_class_images(src_dir, train_dir, val_dir, test_dir, class_name, train_pct, val_pct):
    images = [f for f in os.listdir(src_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(images)
    n = len(images)
    n_train, n_val = int(train_pct*n), int(val_pct*n)
    splits = {
        train_dir: images[:n_train],
        val_dir: images[n_train:n_train+n_val],
        test_dir: images[n_train+n_val:]
    }
    for split_dir, img_list in splits.items():
        full_dir = os.path.join(split_dir, class_name)
        os.makedirs(full_dir, exist_ok=True)
        for img in img_list:
            shutil.copy(os.path.join(src_dir, img), os.path.join(full_dir, img))

split_class_images(f"{OUTPUT_SORTED}/benign", TRAIN_DIR, VAL_DIR, TEST_DIR, "benign", TRAIN_PCT, VAL_PCT)
split_class_images(f"{OUTPUT_SORTED}/malignant", TRAIN_DIR, VAL_DIR, TEST_DIR, "malignant", TRAIN_PCT, VAL_PCT)
print("Finished splitting into train/val/test with class folders.")

print("Dataset is now ready for model training (baseline). To train, run your train.py or main.py.")

