import os
import pydicom
import cv2

def dicom_to_png(dicom_path, png_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(png_path, norm_img.astype('uint8'))

def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.dcm'):
            dicom_to_png(os.path.join(input_dir, fname), os.path.join(output_dir, fname.replace('.dcm','.png')))

if __name__ == "__main__":
    batch_convert("data/raw/", "data/preprocessed/")
