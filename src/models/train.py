import os
import sys
from cancer_net import build_cancer_net
from keras.preprocessing.image import ImageDataGenerator  
from keras.callbacks import EarlyStopping, ModelCheckpoint  

def train_model(train_dir, val_dir, save_path):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        train_dir, 
        target_size=(128,128), 
        color_mode='grayscale', 
        batch_size=32,
        class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        val_dir, 
        target_size=(128,128), 
        color_mode='grayscale', 
        batch_size=32,
        class_mode='categorical'
    )
    model = build_cancer_net()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True), 
        ModelCheckpoint(save_path, save_best_only=True)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)
    return model

if __name__ == "__main__":
    train_model("data/train/", "data/val/", "models/baseline_cancernet.h5")

