"""
This model was inspired from Muhammad Irfan's 2020 paper:
"A Novel Feature Extraction Model to Enhance Underwater Image Classification"
DOI: 10.1007/978-3-030-43364-2_8

Some of the kernel sizes and filter numbers have been changed, however the 
general structure remains the same.
"""

import tensorflow as tf
import keras
from keras import layers, optimizers, losses

def get_irfan_model(input_shape=(192, 192, 1)):
    tf.keras.backend.clear_session()

    encoder_input = layers.Input(shape=input_shape, name='input')

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(encoder_input)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=2, padding='same', name='encoder_output')(x)


    x = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=1, padding='same')(x)
    decoder_output = layers.Activation('sigmoid')(x)

    irfan2020 = keras.Model(
        inputs=encoder_input,
        outputs=decoder_output,
        name='irfan_2020'
    )

    return irfan2020

if __name__ == '__main__':
 
    irfan = get_irfan_model(input_shape=(192, 192, 3))

    irfan.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.MeanSquaredError(),
        metrics=tf.image.psnr
    )

    print(irfan.summary())