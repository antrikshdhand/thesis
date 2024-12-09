from keras.layers import \
    Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, \
    UpSampling2D, Concatenate, Input, Dropout, GaussianNoise, ReLU
from keras.models import Model

def get_paths(x, filters):
    path1 = Conv2D(filters, kernel_size=(1, 1), padding="same")(x) 
    path1 = BatchNormalization()(path1)
    path1 = LeakyReLU(alpha=0.1)(path1) # Original paper used ReLU
    # path1 = ReLU()(path1)

    path2 = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    path2 = BatchNormalization()(path2)
    path2 = LeakyReLU(alpha=0.1)(path2) 
    # path2 = ReLU()(path2)

    path3 = Conv2D(filters, kernel_size=(5, 5), padding="same")(x)
    path3 = BatchNormalization()(path3)
    path3 = LeakyReLU(alpha=0.1)(path3) 
    # path3 = ReLU()(path3)

    path4 = Conv2D(filters, kernel_size=(7, 7), padding="same")(x)
    path4 = BatchNormalization()(path4)
    path4 = LeakyReLU(alpha=0.1)(path4) 
    # path4 = ReLU()(path4)

    return path1, path2, path3, path4

def mscm(x, filters):
    path1, path2, path3, path4 = get_paths(x, filters=filters)
    concat_paths = Concatenate()([path1, path2, path3, path4])
    
    concat_with_input = Concatenate()([x, concat_paths])

    path5, path6, path7, path8 = get_paths(concat_with_input, filters=filters)
    concat_paths_2 = Concatenate()([path5, path6, path7, path8])

    return concat_paths_2

def down_block(x, filters, use_maxpool=True):

    x = GaussianNoise(0.1)(x)

    concat_paths = mscm(x, filters=filters)

    if use_maxpool:
        pooled_output = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(concat_paths)
        return pooled_output, concat_paths  
    else:
        return concat_paths  # Final down block doesn’t use max pooling

def up_block(x, skip, filters):
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, skip])

    x = GaussianNoise(0.1)(x)

    x = mscm(x, filters=filters)

    return x

def get_unetpro_model(input_size, dropout):
    filters = [8, 16, 32, 64, 128]

    # ENCODER
    model_input = Input(shape=input_size)

    x, skip1 = down_block(model_input, filters[0])
    x = Dropout(dropout)(x)

    x, skip2 = down_block(x, filters[1])
    x = Dropout(dropout)(x)

    x, skip3 = down_block(x, filters[2])
    x = Dropout(dropout)(x)

    x, skip4 = down_block(x, filters[3])
    x = Dropout(dropout)(x)

    x = down_block(x, filters[4], use_maxpool=False)  # Bottom layer doesn’t use max pool

    # DECODER
    x = up_block(x, skip4, filters[3])
    x = Dropout(dropout)(x)

    x = up_block(x, skip3, filters[2])
    x = Dropout(dropout)(x)

    x = up_block(x, skip2, filters[1])
    x = Dropout(dropout)(x)

    x = up_block(x, skip1, filters[0])
    x = Dropout(dropout)(x)
    
    # Output layer
    output = Conv2D(
        input_size[-1],
        kernel_size=(1, 1),
        activation="sigmoid"
    )(x) 

    # Define model
    model = Model(inputs=model_input, outputs=output, name='unet')

    return model 