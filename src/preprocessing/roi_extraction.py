import cv2, os

def crop_roi(img_path, roi_coords, save_path):
    img = cv2.imread(img_path, 0)
    x, y, w, h = roi_coords
    roi = img[y:y+h, x:x+w]
    cv2.imwrite(save_path, roi)
