import tensorflow as tf


@tf.function
def load_image_train(datapoint, IMG_SIZE = 256, MASK_SIZE = 512):
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (int(MASK_SIZE/2), int(MASK_SIZE/2)))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_mask = tf.reshape(input_mask, (int(MASK_SIZE * MASK_SIZE / 4), 1))

    return input_image, input_mask


@tf.function
def load_image_test(datapoint, IMG_SIZE = 256, MASK_SIZE = 512):
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (int(MASK_SIZE/2), int(MASK_SIZE/2)))
    input_mask = tf.reshape(input_mask, (int(MASK_SIZE * MASK_SIZE / 4), 1))
    return input_image, input_mask
