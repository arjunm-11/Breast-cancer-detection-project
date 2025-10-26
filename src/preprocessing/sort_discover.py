import os, re, shutil, json, math
import pandas as pd

RAW_JPEG_ROOT = "data/raw/archive/jpeg"
OUT_BENIGN = "data/preprocessed/benign"
OUT_MALIGN = "data/preprocessed/malignant"
os.makedirs(OUT_BENIGN, exist_ok=True)
os.makedirs(OUT_MALIGN, exist_ok=True)

UID_RE = re.compile(r"(?:1\.3\.6(?:\.\d+){5,})")  # matches 1.3.6.1.4.1.9590... style UID

def is_empty(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and not x.strip())

def extract_last_uid(path_like: str):
    if is_empty(path_like):
        return None
    s = str(path_like).strip().replace("\\", "/").lower()
    uids = UID_RE.findall(s)
    if not uids:
        return None
    return uids[-1]  # last UID in the path

def get_label(row):
    for k in ["pathology","Pathology","label","Label"]:
        if k in row and isinstance(row[k], str) and row[k].strip():
            v = row[k].strip().lower()
            if v in ("benign","malignant"):
                return v
    return None

def find_jpeg_in_uid_folder(uid: str):
    """Look for images directly in <RAW_JPEG_ROOT>/<UID> or one level below."""
    uid_dir = os.path.join(RAW_JPEG_ROOT, uid)
    if os.path.isdir(uid_dir):
        # Direct files
        files = [f for f in os.listdir(uid_dir) if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp"))]
        if files:
            # Prefer 000000.* or 000001.* if present
            preferred = [f for f in files if os.path.splitext(f)[0] in ("000000","000001")]
            pick = preferred[0] if preferred else files[0]
            return os.path.join(uid_dir, pick)
        # Look exactly one level deeper
        for sub in os.listdir(uid_dir):
            subdir = os.path.join(uid_dir, sub)
            if os.path.isdir(subdir):
                files = [f for f in os.listdir(subdir) if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp"))]
                if files:
                    return os.path.join(subdir, files[0])
    return None

CSV_FILES = [
    "data/raw/archive/csv/mass_case_description_train_set.csv",
    "data/raw/archive/csv/mass_case_description_test_set.csv",
    "data/raw/archive/csv/calc_case_description_train_set.csv",
    "data/raw/archive/csv/calc_case_description_test_set.csv",
]

copied = {"benign":0,"malignant":0}
unmatched = []

print("Mapping CSV entries to UID folders in jpeg/...")
for csv_path in CSV_FILES:
    if not os.path.exists(csv_path):
        print("Missing CSV:", csv_path); 
        continue
    df = pd.read_csv(csv_path)
    # pick path-like columns
    path_cols = [c for c in df.columns if "path" in c.lower()]
    if not path_cols:
        print("No path-like columns in", csv_path, "->", list(df.columns))
        continue

    for _, row in df.iterrows():
        label = get_label(row)
        if label not in ("benign","malignant"):
            continue

        uid = None
        for c in path_cols:
            uid = extract_last_uid(row[c])
            if uid:
                break
        if not uid:
            unmatched.append({"csv": os.path.basename(csv_path), "reason": "no_uid"})
            continue

        img = find_jpeg_in_uid_folder(uid)
        if not img or not os.path.isfile(img):
            unmatched.append({"csv": os.path.basename(csv_path), "uid": uid, "reason": "no_image_in_uid_folder"})
            continue

        dst_dir = OUT_BENIGN if label == "benign" else OUT_MALIGN
        dst = os.path.join(dst_dir, f"{uid}_" + os.path.basename(img))
        if not os.path.exists(dst):
            shutil.copy(img, dst)
        copied[label] += 1

print(f"Copied benign: {copied['benign']} | malignant: {copied['malignant']}")
print("Unmatched rows:", len(unmatched))
if unmatched:
    with open("unmatched_uid_folders.json","w") as f:
        json.dump(unmatched[:200], f, indent=2)
    print("Wrote unmatched_uid_folders.json (first 200)")

# Split to train/val/test
def split_class(src_dir, cls, train_pct=0.7, val_pct=0.15):
    import random
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp"))]
    if not files:
        return
    random.shuffle(files)
    n = len(files)
    n_train, n_val = int(n*train_pct), int(n*val_pct)
    splits = {"train": files[:n_train], "val": files[n_train:n_train+n_val], "test": files[n_train+n_val:]}
    for sp, flist in splits.items():
        out = os.path.join("data", sp, cls)
        os.makedirs(out, exist_ok=True)
        for f in flist:
            src = os.path.join(src_dir, f)
            dst = os.path.join(out, f)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

if any(os.scandir(OUT_BENIGN)):
    split_class(OUT_BENIGN, "benign")
if any(os.scandir(OUT_MALIGN)):
    split_class(OUT_MALIGN, "malignant")

print("Done. Verify:")
print("  find data/preprocessed/benign -type f | wc -l")
print("  find data/preprocessed/malignant -type f | wc -l")
print("  find data/train -type f | wc -l")
