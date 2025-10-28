import os, shutil, random

def split_dataset(img_dir, train_dir, val_dir, test_dir, train_pct=0.7, val_pct=0.15):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.shuffle(img_files)
    n_total = len(img_files)
    n_train = int(train_pct * n_total)
    n_val = int(val_pct * n_total)
    print(n_total, n_train, n_val)
    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train+n_val]
    test_files = img_files[n_train+n_val:]

    for d, files in zip([train_dir, val_dir, test_dir], [train_files, val_files, test_files]):
        os.makedirs(d, exist_ok=True)
        for f in files:
            src_dir = os.path.join(img_dir, f)
            des_dir = os.path.join(d, f)
            # print(src_dir, "->", des_dir)
            shutil.copy(src_dir, des_dir)

if __name__ == "__main__":
    split_dataset("data/preprocessed_all/", "data/train/", "data/val/", "data/test/")
