import tensorflow as tf
import keras
from keras.layers import \
    Conv2D, LeakyReLU, MaxPool2D, \
    UpSampling2D, Concatenate, Input

def down_block(x, filters, initializer, use_maxpool=True):
    x = Conv2D(filters, kernel_size=(3, 3), padding="same", kernel_initializer=initializer)(x) 
    x = LeakyReLU(alpha=0.1)(x) 

    x = Conv2D(filters, kernel_size=(3, 3), padding="same", kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=0.1)(x) 

    if use_maxpool:
        return MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x), x
    else:
        return x
    
def up_block(x, y, filters, initializer):
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=3)([x, y])

    x = Conv2D(filters, 3, padding="same", kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=0.1)(x) 

    x = Conv2D(filters, 3, padding="same", kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=0.1)(x) 

    return x

def get_unet_model(input_shape=(256, 256, 3)):
    filters = [64, 128, 256, 512, 1024]
    initializer = tf.keras.initializers.HeNormal(seed=44)

    # ENCODER
    model_input = Input(shape=input_shape)
    x, temp1 = down_block(model_input, filters[0], initializer=initializer)
    x, temp2 = down_block(x, filters[1], initializer=initializer)
    x, temp3 = down_block(x, filters[2], initializer=initializer)
    x, temp4 = down_block(x, filters[3], initializer=initializer)

    # BOTTLENECK
    x = down_block(x, filters[4], initializer=initializer, use_maxpool=False)

    # DECODER
    x = up_block(x, temp4, filters[3], initializer=initializer)
    x = up_block(x, temp3, filters[2], initializer=initializer)
    x = up_block(x, temp2, filters[1], initializer=initializer)
    x = up_block(x, temp1, filters[0], initializer=initializer)

    output = Conv2D(input_shape[-1], kernel_size=(1, 1), activation='linear', kernel_initializer=initializer)(x)

    model = keras.models.Model(inputs=model_input, outputs=output, name='unet')

    return model

if __name__ == "__main__":
    model = get_unet_model(input_shape=(192, 192, 1))
    print(model.summary())
    
    # input_image = tf.random.uniform([1, 512, 512, 3])
    # output = model(input_image)
    # print(output)