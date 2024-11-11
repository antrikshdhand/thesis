import keras.optimizers
import tensorflow as tf
import keras
from keras.layers import (
    Reshape,
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    Activation,
    BatchNormalization,
    LayerNormalization,
    LSTM,
)
from keras import metrics

def compile_model(model, loss=None, optimizer=None, metrics=None):
    if optimizer is None or loss is None:
        return

    if metrics is None:
        metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def callback_early_stop(patience=10, restore_best_weights=True):
    return tf.keras.callbacks.EarlyStopping(
        monitor="loss", 
        patience=patience, 
        restore_best_weights=restore_best_weights
    )


def _cnn_2d_layer(
        input_layer,
        filters,
        kernel_size,
        batch_norm=True,
        activation=None,
        max_pool=True,
        pool_size=None,
        strides=None,
):

    if max_pool and pool_size is None and strides is None:
        raise ValueError(
            "Must Set Pool Size and Strides if Max Pool is Not None"
        )

    step = Conv2D(
        filters=filters, 
        kernel_size=kernel_size, 
        padding="same"
    )(input_layer)

    if batch_norm:
        step = BatchNormalization()(step)

    if activation is not None:
        step = Activation(activation)(step)

    if max_pool:
        step = MaxPool2D(
            pool_size=pool_size, 
            strides=strides, 
            padding="same"
        )(step)

    return step


def _get_cnn_layers(input_layer, filter_size=(5, 5), num_filters=128): 
    conv_2d = _cnn_2d_layer(
        input_layer,
        filters=num_filters,
        kernel_size=filter_size,
        batch_norm=True,
        activation="relu",
        max_pool=True,
        pool_size=2,
        strides=2,
    )

    conv_2d = _cnn_2d_layer(
        conv_2d,
        filters=num_filters,
        kernel_size=filter_size,
        batch_norm=True,
        activation="relu",
        max_pool=True,
        pool_size=(4, 2),
        strides=(4, 2),
    )

    conv_2d = _cnn_2d_layer(
        conv_2d,
        filters=2 * num_filters,
        kernel_size=filter_size,
        batch_norm=True,
        activation="relu",
        max_pool=True,
        pool_size=(2, 2),
        strides=(2, 2),
    )

    conv_2d = _cnn_2d_layer(
        conv_2d,
        filters=2 * num_filters,
        kernel_size=filter_size,
        batch_norm=True,
        activation="relu",
        max_pool=True,
        pool_size=(4, 2),
        strides=(4, 2),
    )

    return conv_2d


# def get_cnn_attn_model(
#         input_shape=(216, 2, 59),
#         num_classes=4,
#         filter_size=(3, 3),
#         num_filters=64,
#         verbose=False,
# ):
#     keras.backend.clear_session()

#     input_layer = Input(shape=input_shape, dtype=tf.float32, name="CQT Layer")

#     conv_2d = _get_cnn_layers(
#         input_layer, filter_size=filter_size, num_filters=num_filters
#     )

#     flatten = Flatten()(conv_2d)
#     expand_dims = keras.layers.Reshape(
#         target_shape=(1, flatten.shape[1])
#     )(flatten)
#     self_attention = keras.layers.Attention()([expand_dims, expand_dims])
#     layer_norm = LayerNormalization()(self_attention)

#     flatten = Flatten()(layer_norm)
#     classification_layer = Dense(num_classes, activation="softmax")(flatten)

#     model = keras.Model(input_layer, classification_layer, name="cnn_attention")

#     if verbose:
#         print(model.summary())
#     return model


def get_cnn_lstm(
        input_shape,
        input_name='image',
        num_classes=4,
        verbose=False,
):

    """ Images/Spectral Representations """
    input_layer = Input(shape=input_shape, dtype=tf.float32, name=input_name)

    conv_2d = _get_cnn_layers(input_layer, filter_size=(5, 5), num_filters=128)

    layer_norm = LayerNormalization()(conv_2d)
    layer_norm = Reshape(
        [-1, layer_norm.shape[2] * layer_norm.shape[3]]
    )(layer_norm)
    # flatten = Flatten(2, 3, name="flatten")(layer_norm)

    """ LSTM Embeddings """
    fwd_lstm_layer = LSTM(512, activation="tanh")(layer_norm)

    flatten = Flatten(name="flatten")(fwd_lstm_layer)

    """ Classification """
    dense = Dense(64, activation="relu", name="dense_all")(flatten)
    classification = Dense(
        num_classes, 
        activation="softmax", 
        name="classification"
    )(dense)

    model = keras.Model(
        [input_layer],
        [classification],
        name="cnn_lstm",
    )

    if verbose:
        print(model.summary())

    return model


if __name__ == "__main__":
    model = get_cnn_lstm(
        input_shape=(116, 100, 1),
        input_name='spectrogram',
        num_classes=4, 
        verbose=True
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            metrics.Accuracy(),
            metrics.Precision(),
            metrics.F1Score(average="weighted")
        ]
    )

    print(model.summary())
