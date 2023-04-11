from keras.models import Model
from keras.layers import Input, BatchNormalization, Add, MaxPool2D
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.applications import MobileNetV3Large

from soccer_segmentation.models.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from classification_models.keras import Classifiers


def segnet_mobilenetlarge(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):

    base_model = MobileNetV3Large(input_shape=input_shape, weights='imagenet')

    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()
    for layer in base_model.layers:
        layer.trainable = False

    base_model = Model(base_model.input, base_model.layers[-4].output)
    base_model.summary()

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(base_model.get_layer("re_lu_2").output)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(base_model.get_layer("re_lu_6").output)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(base_model.get_layer("multiply_1").output)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(base_model.get_layer("multiply_13").output)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(base_model.get_layer("multiply_19").output)

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(960, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(960, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(672, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(672, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(672, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(240, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(240, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(240, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(72, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(72, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape(
        (int(input_shape[0] * input_shape[1] / 4), n_labels),
        input_shape=(int(input_shape[0] / 2), int(input_shape[1] / 2), n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)

    model = Model(inputs=base_model.inputs, outputs=outputs, name="SegNetMobileNetLarge")

    return model


if __name__ == "__main__":
    resseg = segnet_mobilenetlarge(input_shape=(256, 256, 3), n_labels=3)
    resseg.summary()
