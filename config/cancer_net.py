from keras import models, layers

def build_cancer_net(input_shape=(128,128,1), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu')(inputs)  # groups defaults to 1
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

