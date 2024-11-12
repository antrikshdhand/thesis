import tensorflow as tf
from keras import layers, regularizers, models

def get_sae_model(input_shape=(192, 192, 1), sparsity_coefficient=1e-5):

    # Encoder
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = layers.Flatten()(encoder_input)
    
    x = layers.Dense(256, activation='relu', activity_regularizer=regularizers.l1(sparsity_coefficient))(x)
    x = layers.Dense(128, activation='relu', activity_regularizer=regularizers.l1(sparsity_coefficient))(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=regularizers.l1(sparsity_coefficient))(x)

    # Bottleneck
    bottleneck = layers.Dense(32, activation='relu', activity_regularizer=regularizers.l1(sparsity_coefficient), name="bottleneck")(x)

    # Decoder
    x = layers.Dense(64, activation='relu')(bottleneck)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(tf.reduce_prod(input_shape), activation='sigmoid')(x)

    decoder_output = layers.Reshape(input_shape)(x)

    # Model
    autoencoder = models.Model(encoder_input, decoder_output, name="sparse_autoencoder")

    return autoencoder