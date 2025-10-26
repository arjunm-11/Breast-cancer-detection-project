# split_patientwise.py
import os, shutil, random, re, pandas as pd

PRE_BEN = "data/preprocessed/benign"
PRE_MAL = "data/preprocessed/malignant"
CSV_FILES = [
    "data/raw/archive/csv/mass_case_description_train_set.csv",
    "data/raw/archive/csv/mass_case_description_test_set.csv",
    "data/raw/archive/csv/calc_case_description_train_set.csv",
    "data/raw/archive/csv/calc_case_description_test_set.csv",
]

def load_patient_labels():
    pid_to_label = {}
    pat_re = re.compile(r"(P_\d{5})", re.IGNORECASE)
    for csv in CSV_FILES:
        if not os.path.exists(csv): 
            continue
        df = pd.read_csv(csv)
        # pick any path-like column to extract patient id
        path_cols = [c for c in df.columns if "path" in c.lower()]
        if not path_cols:
            continue
        for _, row in df.iterrows():
            label = str(row.get("pathology","")).strip().lower()
            if label not in ("benign","malignant"):
                continue
            s = " ".join([str(row[c]) for c in path_cols if not pd.isna(row[c])])
            m = pat_re.search(s)
            if m:
                pid = m.group(1).upper()
                pid_to_label[pid] = label
    return pid_to_label

def build_pid_index():
    # Map from patient id to list of image paths in preprocessed folders
    pid_index = {"benign": {}, "malignant": {}}
    pat_re = re.compile(r"(P_\d{5})", re.IGNORECASE)
    for label, root in [("benign", PRE_BEN), ("malignant", PRE_MAL)]:
        for f in os.listdir(root):
            if not f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp")):
                continue
            m = pat_re.search(f)
            pid = (m.group(1).upper() if m else f)  # fallback to filename
            pid_index[label].setdefault(pid, []).append(os.path.join(root, f))
    return pid_index

def split_by_patient(pid_index, train_pct=0.7, val_pct=0.15):
    for d in ["data/train/benign","data/train/malignant",
              "data/val/benign","data/val/malignant",
              "data/test/benign","data/test/malignant"]:
        os.makedirs(d, exist_ok=True)
    for label in ["benign","malignant"]:
        pids = list(pid_index[label].keys())
        random.shuffle(pids)
        n = len(pids)
        nt, nv = int(train_pct*n), int(val_pct*n)
        splits = {"train": pids[:nt], "val": pids[nt:nt+nv], "test": pids[nt+nv:]}
        for sp, pid_list in splits.items():
            for pid in pid_list:
                for src in pid_index[label][pid]:
                    dst = os.path.join("data", sp, label, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)

if __name__ == "__main__":
    random.seed(42)
    pid_labels = load_patient_labels()   # not strictly required but can be used for checks
    pid_index = build_pid_index()
    # Optional sanity: print class patient counts
    print("Benign patients:", len(pid_index["benign"]), "Malignant patients:", len(pid_index["malignant"]))
    split_by_patient(pid_index)
    for sp in ["train","val","test"]:
        for cls in ["benign","malignant"]:
            import subprocess
            cnt = int(subprocess.check_output(f"find data/{sp}/{cls} -type f | wc -l", shell=True).decode().strip() or 0)
            print(f"{sp}/{cls}: {cnt}")
