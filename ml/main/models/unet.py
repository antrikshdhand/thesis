import keras
from keras.layers import \
    Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, \
    UpSampling2D, Concatenate, Input, Dropout

def down_block(x, filters, use_maxpool=True):
    x = Conv2D(filters, kernel_size=(3,3), padding="same")(x) # Note that original UNet did not use padding='same'
    x = BatchNormalization()(x) # EXPERIMENT WITH BATCHNORM BEFORE/AFTER ACTIVATION 
    x = LeakyReLU()(x) # Original UNet used ReLU

    x = Conv2D(filters, kernel_size=(3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    if use_maxpool:
        return MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x), x
    else:
        return x
    
def up_block(x, y, filters):
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=3)([x, y])

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x

def get_unet_model(input_size=(256, 256, 3), *, classes, dropout):
    filters = [64, 128, 256, 512, 1024]
    # filters = [8, 16, 32, 64, 128]

    # ENCODER
    model_input = Input(shape=input_size)
    x, temp1 = down_block(model_input, filters[0])
    x, temp2 = down_block(x, filters[1])
    x, temp3 = down_block(x, filters[2])
    x, temp4 = down_block(x, filters[3])
    x = down_block(x, filters[4], use_maxpool=False)

    # DECODER
    x = up_block(x, temp4, filters[3])
    x = up_block(x, temp3, filters[2])
    x = up_block(x, temp2, filters[1])
    x = up_block(x, temp1, filters[0])

    x = Dropout(dropout)(x)
    output = Conv2D(classes, kernel_size=(1, 1))(x)
    model = keras.models.Model(inputs=model_input, outputs=output, name='unet')

    return model