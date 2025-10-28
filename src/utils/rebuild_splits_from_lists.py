import os, shutil, argparse

def copy_from_list(list_file, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with open(list_file) as f:
        for src in [l.strip() for l in f if l.strip()]:
            dst = os.path.join(dest_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy(src, dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists_dir", required=True)
    ap.add_argument("--source_root", required=True)   # data/preprocessed or data/enhanced/<pipeline>
    ap.add_argument("--out_root", required=True)      # data/train, data/val, data/test targets
    args = ap.parse_args()

    # wipe current splits
    for split in ["train","val","test"]:
        d = os.path.join(args.out_root if args.out_root.endswith(split) else "", split)
        if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)

    # map each list to target
    mapping = [
        ("train_benign.txt", "data/train/benign"),
        ("train_malignant.txt", "data/train/malignant"),
        ("val_benign.txt", "data/val/benign"),
        ("val_malignant.txt", "data/val/malignant"),
        ("test_benign.txt", "data/test/benign"),
        ("test_malignant.txt", "data/test/malignant"),
    ]
    for lst, dst in mapping:
        os.makedirs(dst, exist_ok=True)

    # copy using lists but replacing root prefix
    for lst, dst in mapping:
        with open(os.path.join(args.lists_dir, lst)) as f:
            for src in [l.strip() for l in f if l.strip()]:
                rel = os.path.relpath(src, start="data/preprocessed")
                new_src = os.path.join(args.source_root, rel)
                os.makedirs(dst, exist_ok=True)
                if os.path.isfile(new_src):
                    tgt = os.path.join(dst, os.path.basename(new_src))
                    if not os.path.exists(tgt):
                        shutil.copy(new_src, tgt)
    print("Rebuilt splits from", args.source_root)
