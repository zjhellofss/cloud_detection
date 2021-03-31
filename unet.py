import math
import random

import tensorflow as tf
from resnet import get_resnet50_encoder
from tensorflow import keras
import tensorflow.keras.backend as K
import pix2pix
IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def attention_concat(x, skip):
    concatenate = tf.keras.layers.Concatenate()([x, skip])
    # return concatenate
    n_filters = K.int_shape(x)[-1]

    branch0 = conv_batch(concatenate, n_filters=n_filters, kernel=(3, 3), padding='same')
    branch_1 = conv_act(branch0, n_filters=n_filters, pooling=True, activation='relu')
    branch_1 = conv_act(branch_1, n_filters=n_filters, pooling=False, activation='sigmoid')

    x = multiply([branch0, branch_1])
    return tf.keras.layers.Add()([branch0, x])


def conv_batch(x, n_filters=64, kernel=(2, 2), strides=(1, 1), padding='valid', activation='relu'):
    """

    :param x: 输入的特征
    :param n_filters: 输出的通道数量
    :return: 经过卷积和批处理操作的特征图
    """
    filters = n_filters

    conv_ = Conv2D(filters=filters,
                   kernel_size=kernel,
                   strides=strides,
                   padding=padding)

    batch_norm = keras.layers.BatchNormalization()
    activation = keras.layers.Activation(activation)
    x = conv_(x)
    x = batch_norm(x)
    x = activation(x)
    return x


from tensorflow.keras.layers import *


def conv_act(x, n_filters, kernel=(1, 1), activation='relu', pooling=False):
    poolingLayer = keras.layers.AveragePooling2D(pool_size=(1, 1), padding='same')
    convLayer = Conv2D(filters=n_filters,
                       kernel_size=kernel,
                       strides=1)
    activation = keras.layers.Activation(activation)
    if pooling:
        x = poolingLayer(x)
    x = convLayer(x)
    x = activation(x)
    return x


def bn_relu(input_tensor):
    """It adds a Batch_normalization layer before a Relu
    """
    input_tensor = BatchNormalization(axis=3)(input_tensor)
    return Activation("relu")(input_tensor)


def channel_layer(inputs_tensor=None, num=None, gamma=2, b=1, **kwargs):
    """
    ECA-NET
    :param inputs_tensor: input_tensor.shape=[batchsize,h,w,channels]
    :param num:
    :param gamma:
    :param b:
    :return:
    """
    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels, 1))(x_global_avg_pool)
    num = random.randint(0, 99)
    x = Conv1D(1, kernel_size=k, padding="same", name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_conv1_relu_' + str(num))(x)  # shape=[batch,chnnels,1]
    x = Reshape((1, 1, channels))(x)
    output = multiply([inputs_tensor, x])
    return output


def contr_arm(input_tensor, filters, kernel_size):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """

    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = bn_relu(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = bn_relu(x)
    x = channel_layer(x)

    x1 = Conv2D(filters, 1, padding='same')(input_tensor)
    x1 = bn_relu(x1)

    x = keras.layers.add([x, x1])
    x = Activation("relu")(x)
    return x


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=384,
          input_width=384, channels=3):
    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    # o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = attention_concat(o, f3)
    o = contr_arm(o, 256, 3)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = attention_concat(o, f2)
    o = contr_arm(o, 128, 3)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        # o = (concatenate([o, f1], axis=MERGE_AXIS))
        o = attention_concat(o, f1)

    o = contr_arm(o, 64, 3)

    o = pix2pix.upsample(32, 3)(o)
    o = tf.keras.layers.Conv2D(n_classes, 1, activation='sigmoid')(o)
    model = tf.keras.Model(inputs=img_input, outputs=o)
    model.summary()
    return model


def resnet50_unet(n_classes, input_height=384, input_width=384,
                  encoder_level=3, channels=3):
    model = _unet(n_classes, get_resnet50_encoder, input_height=input_height, input_width=input_width,
                  channels=channels)
    model.model_name = "resnet50_unet"
    return model
