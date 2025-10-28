import cv2
import numpy as np

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
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    out = clahe.apply(g)
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

PIPELINES = {
    "bilateral": bilateral,
    "clahe": clahe,
    "hist_eq": hist_eq,
    "unsharp": unsharp,
    "median": median,
    "nlm": nlm
}
