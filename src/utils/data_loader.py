import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def get_dataframe(path):
    filenames = []
    labels = []
    for label in os.listdir(path):
        for fname in os.listdir(os.path.join(path, label)):
            filenames.append(os.path.join(path, label, fname))
            labels.append(label)
    return pd.DataFrame({'filename': filenames, 'class': labels})

def make_generator(df, batch_size=32, img_size=(128,128)):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_dataframe(df, x_col='filename', y_col='class', target_size=img_size, color_mode='grayscale', class_mode='categorical', batch_size=batch_size)
