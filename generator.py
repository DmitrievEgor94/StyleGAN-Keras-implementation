import keras.backend as K
import numpy as np
from keras import Model
from keras.engine import Layer
from keras.layers import Dense, Input, LeakyReLU, Lambda, Conv2D, UpSampling2D, SeparableConv2D


def subtract_mean(x, axis=None):
    return Lambda(lambda tensor: tensor - K.mean(tensor, axis=axis, keepdims=True))(x)


def divide_on_std(x, axis=None):
    return Lambda(lambda tensor: tensor / K.std(tensor, axis=axis, keepdims=True))(x)


def mapping_network(x, w_size):
    x = subtract_mean(x, axis=-1)
    x = divide_on_std(x, axis=-1)

    for _ in range(8):
        x = Dense(units=w_size)(x)
        x = LeakyReLU()(x)

    return x


class B(Layer):
    def __init__(self, w_size, image_size, batch_size, **kwargs):
        self.image_size = image_size
        self.w_size = w_size
        self.batch_size = batch_size
        self.scalar_weights = []

        super(B, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scalar_weights = self.add_weight('scales',
                                              shape=(1, 1, 1, self.w_size),
                                              initializer='uniform',
                                              trainable=True)

        super(B, self).build(input_shape)

    def call(self, x):
        return K.reshape(x * self.scalar_weights, (self.batch_size, self.image_size, self.image_size, self.w_size))

    def compute_output_shape(self, input_shape):
        return input_shape


def AdaIn(x, w_vector, feature_maps, batch_size):
    x = subtract_mean(x, axis=(1, 2))
    x = divide_on_std(x, axis=(1, 2))

    y_s = Dense(feature_maps)(w_vector)
    y_b = Dense(feature_maps)(w_vector)

    y_s = K.reshape(y_s, (batch_size, 1, 1, feature_maps))
    y_b = K.reshape(y_b, (batch_size, 1, 1, feature_maps))

    x = Lambda(lambda tensor: tensor * y_s + y_b)(x)

    return x


def add_noise(x, image_size, feature_maps, batch_size):
    noise = K.variable(np.random.normal(0, 1, (batch_size, image_size, image_size, 1)))
    noise = K.repeat_elements(noise, feature_maps, axis=-1)

    noise = Lambda(lambda _: noise)([])
    scaled_noise = B(feature_maps, image_size, batch_size)([noise])

    return Lambda(lambda tensor: tensor + x)(scaled_noise)


def block_of_synthesis_network(x, image_size, w_vector, feature_maps, batch_size):
    x = UpSampling2D(interpolation='bilinear')(x)
    x = Conv2D(filters=feature_maps, kernel_size=(3, 3), padding='same')(x)
    x = add_noise(x, image_size, feature_maps, batch_size)
    x = AdaIn(x, w_vector, feature_maps, batch_size)
    x = Conv2D(filters=feature_maps, kernel_size=(3, 3), padding='same')(x)
    x = add_noise(x, image_size, feature_maps, batch_size)
    x = AdaIn(x, w_vector, feature_maps, batch_size)

    return x


def synthesis_network(w_vector, batch_size):
    constant_tensor = K.variable(np.ones((batch_size, 4, 4, 512)))

    overall_feature_maps = 8192
    max_feature_maps = 512

    image_size = 4

    x = add_noise(constant_tensor, image_size, max_feature_maps, batch_size)
    x = AdaIn(x, w_vector, max_feature_maps, batch_size)
    x = Conv2D(filters=max_feature_maps, kernel_size=(3, 3), padding='same')(x)
    x = AdaIn(x, w_vector, max_feature_maps, batch_size)

    for _ in range(8):
        image_size *= 2
        feature_maps = min(overall_feature_maps // (image_size // 2), max_feature_maps)
        x = block_of_synthesis_network(x, image_size, w_vector, feature_maps, batch_size)

    x = SeparableConv2D(filters=3, kernel_size=(3, 3), padding='same')(x)

    return x


def get_generator(w_size=512, batch_size=3):
    latent_code = Input(batch_shape=(batch_size, w_size))

    x = mapping_network(latent_code, w_size)
    x = synthesis_network(x, batch_size)

    return Model(inputs=latent_code, outputs=x)
