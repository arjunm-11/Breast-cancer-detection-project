# src/preprocessing/sort_by_uid.py
import os, re, shutil
import pandas as pd
from collections import defaultdict

# Adjust to your layout
RAW_ROOTS = [
    "data/raw/archive/jpeg"
]
CSV_FILES = [
    "data/raw/archive/csv/mass_case_description_train_set.csv",
    "data/raw/archive/csv/mass_case_description_test_set.csv",
    "data/raw/archive/csv/calc_case_description_train_set.csv",
    "data/raw/archive/csv/calc_case_description_test_set.csv",
]

OUT_BENIGN = "data/preprocessed_all/benign"
OUT_MALIGN = "data/preprocessed_all/malignant"
os.makedirs(OUT_BENIGN, exist_ok=True)
os.makedirs(OUT_MALIGN, exist_ok=True)

# 1) Extract DICOM UIDs from a path string
UID_RE = re.compile(r"(?:1\.3\.6(?:\.\d+){5,})")  # matches 1.3.6.1.4.1.9590... style

def extract_uids(s: str):
    s = str(s).strip().replace("\\", "/")
    return UID_RE.findall(s)

# 2) Build an index: UID -> list of image paths containing that UID in the filename
print("Indexing images by UID...")
uid_to_paths = defaultdict(list)

def index_root(root):
    if not os.path.isdir(root):
        return
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")):
                full = os.path.join(r, f)
                name = f.lower()
                # collect all UID-like substrings present in the filename
                for uid in UID_RE.findall(name):
                    uid_to_paths[uid].append(full)

for rr in RAW_ROOTS:
    index_root(rr)

print(f"Indexed UIDs: {len(uid_to_paths)}")

def get_label(row):
    for k in ["pathology","Pathology","label","Label"]:
        if k in row and isinstance(row[k], str) and row[k].strip():
            v = row[k].strip().lower()
            if v in ("benign","malignant"):
                return v
    return None

copied = {"benign":0, "malignant":0}
missing_rows = []

def copy_first(paths, dst_dir):
    if not paths:
        return False
    src = paths[0]  # take first match
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    return True

print("Matching CSV rows to images by UID...")
for csv_path in CSV_FILES:
    if not os.path.exists(csv_path):
        print("Missing CSV:", csv_path); continue
    df = pd.read_csv(csv_path)
    # Use the two path columns in mass/calc CSVs
    path_cols = [c for c in df.columns if "file path" in c.lower() or c.lower().endswith("path")]
    if not path_cols:
        print("No path columns found in", csv_path, "Columns:", list(df.columns))
        continue

    for _, row in df.iterrows():
        label = get_label(row)
        if label not in ("benign","malignant"):
            continue

        matched = False
        # Check all path-like columns for UIDs
        for c in path_cols:
            uids = extract_uids(row[c])
            for uid in uids:
                # direct UID match
                paths = uid_to_paths.get(uid, [])
                if paths:
                    matched = copy_first(paths, OUT_BENIGN if label=="benign" else OUT_MALIGN)
                    if matched:
                        copied[label] += 1
                        break
            if matched:
                break

        if not matched:
            # Record the row id or image path for inspection
            missing_rows.append({ "csv": os.path.basename(csv_path), "row": dict(row) })

print(f"Copied benign: {copied['benign']} | malignant: {copied['malignant']}")
print(f"Unmatched rows: {len(missing_rows)}")

if missing_rows:
    import json
    with open("unmatched_rows.json","w") as f:
        json.dump(missing_rows[:200], f, indent=2)  # cap for brevity
    print("Wrote unmatched_rows.json (first 200). Inspect sample paths/UIDs.")

# Optional: split into train/val/test after successful copies
def split_class(src_dir, cls, train_pct=0.7, val_pct=0.15):
    import random
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff"))]
    random.shuffle(files)
    n = len(files)
    n_train, n_val = int(train_pct*n), int(val_pct*n)
    splits = {"train": files[:n_train], "val": files[n_train:n_train+n_val], "test": files[n_train+n_val:]}
    for sp, flist in splits.items():
        out = os.path.join("data", sp, cls)
        os.makedirs(out, exist_ok=True)
        for f in flist:
            src = os.path.join(src_dir, f)
            dst = os.path.join(out, f)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

# Only split if we actually copied some files
if any(os.scandir(OUT_BENIGN)):
    split_class(OUT_BENIGN, "benign")
if any(os.scandir(OUT_MALIGN)):
    split_class(OUT_MALIGN, "malignant")

print("Done. Verify with:\n  find data/preprocessed/benign -type f | wc -l\n  find data/preprocessed/malignant -type f | wc -l\n  find data/train -type f | wc -l")
