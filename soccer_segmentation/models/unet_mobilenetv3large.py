from keras.models import Model
from keras.layers import *
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.applications import MobileNetV3Large

from soccer_segmentation.models.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from classification_models.keras import Classifiers


def down_block(x, filters, use_maxpool=True):
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if use_maxpool == True:
        return MaxPooling2D(strides=(2, 2))(x), x
    else:
        return x


def up_block(x, y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis=3)([x, y])
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def unet_mobilenetv3large(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax",
        dropout=0.1):

    filter = [64, 128, 256, 512]

    base_model = MobileNetV3Large(input_shape=input_shape, weights='imagenet')

    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()
    for layer in base_model.layers:
        layer.trainable = False

    base_model = Model(base_model.input, base_model.layers[-3].output)
    base_model.summary()

    temp1 = base_model.get_layer("re_lu_2").output
    temp2 = base_model.get_layer("re_lu_6").output
    temp3 = base_model.get_layer("multiply_1").output
    temp4 = base_model.get_layer("multiply_13").output
    x = base_model.get_layer("multiply_19").output
    x = up_block(x, temp4, 672)
    x = up_block(x, temp3, 240)
    x = up_block(x, temp2, 72)
    x = up_block(x, temp1, 64)
    x = Dropout(dropout)(x)
    x = Conv2D(n_labels, 1, activation='softmax')(x)
    output = Reshape(
        (int(input_shape[0] * input_shape[1] / 4), n_labels),
        input_shape=(int(input_shape[0] / 2), int(input_shape[1] / 2), n_labels))(x)
    model = Model(inputs=base_model.inputs, outputs=output, name="UNetMobileNetv3Large")

    return model


if __name__ == "__main__":
    resseg = unet_mobilenetv3large(input_shape=(256, 256, 3), n_labels=3)
    resseg.summary()
