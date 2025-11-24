import os, sys, cv2, numpy as np

# Ensure repo root on path when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.pipelines import bilateral, clahe, hist_eq, median, unsharp, nlm, gaussian, wiener

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def gamma_corr(img, gamma=0.8):
    g = to_gray(img).astype(np.float32) / 255.0
    out = np.power(g, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def histeq_median(img, k=3):
    g = to_gray(img)
    eq = cv2.equalizeHist(g)
    k = int(max(3, (k // 2) * 2 + 1))  # force odd kernel
    return cv2.medianBlur(eq, k)

def put_label(im, text):
    im3 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()
    h, w = im3.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(w, h) / 512.0)
    thickness = 1 if w < 512 else 2
    margin = int(8 * scale)
    cv2.putText(im3, text, (margin, h - margin), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(im3, text, (margin, h - margin), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return im3

def main():
    src_path = "figures/source/example.jpg"
    out_dir = "figures/panels"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read {src_path}")

    # Exactly 9 variants for a 3x3 grid
    variants = [
        ("Original", img),
        ("Bilateral", bilateral(img)),
        ("wiener", wiener(img, 5)),
        ("HistEq", hist_eq(img)),
        ("HistEq+Median", histeq_median(img, 3)),   # new composite
        ("Gamma 0.8", gamma_corr(img, 0.8)),
        ("Gamma 1.2", gamma_corr(img, 1.2)),
        ("Unsharp", unsharp(img, 1.0)),
        ("Gaussian", gaussian(img, 5, 0))  # new addition,
    ]

    # Normalize size and label; fall back to original if any output is None
    H, W = 384, 384
    tiles = []
    for name, im in variants:
        if im is None:
            im = img
        t = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        t = put_label(t, name)
        tiles.append(t)

    # Compose 3x3 with 4-px gutters
    g = 4
    def hstack3(a, b, c):
        return np.hstack([a, np.full((H, g, 3), 0, np.uint8), b, np.full((H, g, 3), 0, np.uint8), c])
    row1 = hstack3(tiles[0], tiles[1], tiles[2])
    row2 = hstack3(tiles[3], tiles[4], tiles[5])
    row3 = hstack3(tiles[6], tiles[7], tiles[8])
    gutter = np.full((g, row1.shape[1], 3), 0, np.uint8)
    panel = np.vstack([row1, gutter, row2, gutter, row3])

    out_path = os.path.join(out_dir, "panel_labeled_3x3.png")
    cv2.imwrite(out_path, panel)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
