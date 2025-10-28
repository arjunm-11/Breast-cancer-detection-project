import os, argparse, json, subprocess, time

def run(cmd):
    print(">>", cmd); ret = subprocess.call(cmd, shell=True); 
    if ret != 0: raise SystemExit(ret)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)              # e.g., exp1_bilateral
    ap.add_argument("--pipeline", required=True)          # e.g., bilateral / clahe / hist_eq / ...
    ap.add_argument("--img_size", type=int, nargs=2, default=[128,128])
    ap.add_argument("--gray", action="store_true")
    args = ap.parse_args()

    exp_dir = os.path.join("experiments", args.name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)

    # 1) Generate enhanced set deterministically from baseline file lists
    run(f"python src/utils/apply_pipeline.py --pipeline {args.pipeline} --lists_dir experiments/exp0_baseline/config --out_root data/enhanced")

    # 2) Rebuild train/val/test using exactly the same files (but enhanced)
    src_root = os.path.join("data/enhanced", args.pipeline)
    run(f"python src/utils/rebuild_splits_from_lists.py --lists_dir experiments/exp0_baseline/config --source_root {src_root} --out_root data")

    # 3) Train
    model_path = os.path.join(exp_dir, "models", f"{args.name}.h5")
    run(f"python src/models/train.py")  # ensure train.py saves to models/baseline_cancernet.h5 or accept a --save_path
    # move model to exp folder if saved in default location
    if os.path.exists("models/baseline_cancernet.h5"):
        os.rename("models/baseline_cancernet.h5", model_path)

    # 4) Evaluate
    run(f"TF_ENABLE_ONEDNN_OPTS=0 python src/models/evaluate.py --model {model_path} --test_dir data/test --img_size {args.img_size[0]} {args.img_size[1]} {'--gray' if args.gray else ''}")

    # 5) Archive results
    for f in ["classification_report.txt","confusion_matrix.csv","confusion_matrix.png","roc_curve.png"]:
        if os.path.exists(os.path.join("results", f)):
            os.rename(os.path.join("results", f), os.path.join(exp_dir, "results", f))

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({"name": args.name, "pipeline": args.pipeline, "img_size": args.img_size, "gray": args.gray, "timestamp": time.ctime()}, f, indent=2)

    print("Finished", args.name, "->", exp_dir)
