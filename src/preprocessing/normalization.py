import cv2, os

def normalize_img(img_path, save_path):
    img = cv2.imread(img_path, 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(save_path, img)
