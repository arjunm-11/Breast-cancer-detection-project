import cv2, os, numpy as np

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(img, table)

def median_filtering(img, k=5):
    return cv2.medianBlur(img, k)

def bilateral_filtering(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def enhance_and_save(input_dir, output_dir, func, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, fname), 0)
        img_enh = func(img, **kwargs) if kwargs else func(img)
        cv2.imwrite(os.path.join(output_dir, fname), img_enh)

if __name__ == "__main__":
    # Change which enhancement you want
    enhance_and_save("data/preprocessed/", "data/enhanced/HE/", histogram_equalization)
    # enhance_and_save("data/preprocessed/", "data/enhanced/gamma/", gamma_correction, gamma=1.5)
    # enhance_and_save("data/preprocessed/", "data/enhanced/median/", median_filtering, k=5)
    # enhance_and_save("data/preprocessed/", "data/enhanced/bilateral/", bilateral_filtering)
