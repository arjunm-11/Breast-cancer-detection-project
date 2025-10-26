from keras.preprocessing.image import ImageDataGenerator

def get_augmented_generator(path, batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    return datagen.flow_from_directory(path, target_size=(128,128), color_mode='grayscale', class_mode='categorical', batch_size=batch_size)
