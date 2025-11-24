import cv2
import numpy as np

def _to_gray_u8(img):
    # Ensures single-channel uint8 without using .astype on None
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        # Safeguard conversion; handles float and other dtypes
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def clip_01(x):
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def bilateral(img, d=7, sigmaColor=50, sigmaSpace=50):
    g = to_gray(img)
    out = cv2.bilateralFilter(g, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return clip_01(out)

def clahe(img, clip=2.0, tile=(8,8)):
    g = to_gray(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    out = clahe_obj.apply(g)
    return clip_01(out)

def hist_eq(img):
    g = to_gray(img)
    out = cv2.equalizeHist(g)
    return clip_01(out)

def unsharp(img, k=1.0):
    g = to_gray(img)
    blur = cv2.GaussianBlur(g, (0,0), 2.0)
    out = cv2.addWeighted(g, 1+k, blur, -k, 0)
    return clip_01(out)

def median(img, k=3):
    g = to_gray(img)
    out = cv2.medianBlur(g, k)
    return clip_01(out)

def nlm(img, h=10):
    g = to_gray(img)
    out = cv2.fastNlMeansDenoising(g, None, h, 7, 21)
    return clip_01(out)

_GAMMA08_LUT = np.array([((i / 255.0) ** 0.8) * 255.0 for i in range(256)], dtype=np.uint8)
_GAMMA12_LUT = np.array([((i / 255.0) ** 1.2) * 255.0 for i in range(256)], dtype=np.uint8)

def gamma_08(img):
    """
    Gamma correction with gamma=0.8 using a precomputed LUT.
    Brightens image. Robust to dtype and channels.
    """
    g = _to_gray_u8(img)
    if g is None:
        return None
    return cv2.LUT(g, _GAMMA08_LUT)

def gamma_12(img):
    """
    Gamma correction with gamma=1.2 using a precomputed LUT.
    Darkens image.
    """
    g = _to_gray_u8(img)
    if g is None:
        return None
    return cv2.LUT(g, _GAMMA12_LUT)

def _ensure_u8_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# -------- Wiener filter --------

def wiener(img, ksize=7, K=0.01):
    """
    Frequency-domain Wiener filter with centered box PSF.
    img: uint8 or any; returns uint8; grayscale.
    ksize: odd int > 1
    K: noise-to-signal ratio (>= 1e-6 recommended)
    """
    g = _ensure_u8_gray(img)
    if g is None or g.size == 0:
        return None
    if g.ndim != 2:
        # enforce grayscale
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    # hard cap supported size to avoid accidentally processing huge full-size images
    if max(g.shape) > 1024:
        # optionally downscale or just return original to avoid hour-long FFTs
        # g = cv2.resize(g, (1024, int(g.shape[0]*1024/g.shape[1])))  # optional
        return g

    g_f = g.astype(np.float32) / 255.0
    h, w = g_f.shape

    # ensure odd, min 3
    ksize = int(max(3, ksize // 2 * 2 + 1))

    # centered PSF (box) then circularly shift to top-left for FFT
    psf = np.ones((ksize, ksize), np.float32)
    psf /= psf.sum()
    pad = np.zeros_like(g_f, dtype=np.float32)
    cy, cx = h // 2, w // 2
    ky, kx = psf.shape
    sy, sx = cy - ky // 2, cx - kx // 2
    pad[sy:sy+ky, sx:sx+kx] = psf
    pad = np.roll(pad, shift=(-cy, -cx), axis=(0, 1))  # center → origin

    G = np.fft.fft2(g_f)
    H = np.fft.fft2(pad)

    H_abs2 = np.abs(H)**2
    K = max(float(K), 1e-6)
    W = np.conj(H) / (H_abs2 + K)

    F_hat = W * G
    f_hat = np.fft.ifft2(F_hat).real

    f_hat = np.clip(f_hat, 0.0, 1.0)
    out = (f_hat * 255.0).astype(np.uint8)
    return out

# -------- Pipeline dict --------

def histeq_median(img):
    g = hist_eq(img)
    out = median(g, k=3)
    return out

def gaussian(img, ksize=5, sigma=0):
    """
    Gaussian blur denoiser.
    - ksize: odd kernel size (e.g., 3,5,7). Must be positive and odd.
    - sigma: Gaussian std in pixels. If 0, OpenCV infers from ksize.
    Returns uint8 grayscale.
    """
    g = _ensure_u8_gray(img)
    if g is None:
        return None
    k = int(max(3, (ksize // 2) * 2 + 1))  # force odd, >=3
    out = cv2.GaussianBlur(g, (k, k), sigmaX=float(sigma), sigmaY=0)
    return out



PIPELINES = {
    "bilateral": bilateral,
    "clahe": clahe,
    "hist_eq": hist_eq,
    "unsharp": unsharp,
    "median": median,
    "nlm": nlm,
    "gamma_08": gamma_08,
    "gamma_12": gamma_12,
    "wiener": wiener,
    "histeq_median": histeq_median,
    "gaussian": gaussian,
}