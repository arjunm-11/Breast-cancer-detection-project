import os, cv2, argparse
from src.preprocessing.pipelines import PIPELINES

def process_list(list_file, out_root, fn):
    with open(list_file) as f:
        paths = [l.strip() for l in f if l.strip()]
    for src in paths:
        # src expected to start with data/preprocessed/...
        rel = os.path.relpath(src, start="data/preprocessed")
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        out = fn(img)
        cv2.imwrite(dst, out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--lists_dir", default="experiments/exp0_baseline/config")
    ap.add_argument("--out_root", default="data/enhanced")
    args = ap.parse_args()

    fn = PIPELINES[args.pipeline]
    out_root = os.path.join(args.out_root, args.pipeline)

    names = ["train_benign.txt","train_malignant.txt","val_benign.txt","val_malignant.txt","test_benign.txt","test_malignant.txt"]
    for name in names:
        process_list(os.path.join(args.lists_dir, name), out_root, fn)
    print("Wrote enhanced images to:", out_root)
