import tensorflow as tf
import numpy as np


def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "segmentation")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)

    mask = tf.image.decode_png(mask, channels=1)

    return {'image': image, 'segmentation_mask': mask}


def load_data(dataset_path, training_data, val_data, test_data):
    N_CHANNELS = 3
    N_CLASSES = 151
    SEED = 42

    train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*", seed=SEED)
    train_dataset = train_dataset.map(parse_image)

    val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*", seed=SEED)
    val_dataset = val_dataset.map(parse_image)

    test_dataset = tf.data.Dataset.list_files(dataset_path + test_data + "*", seed=SEED)
    test_dataset = test_dataset.map(parse_image)

    return train_dataset, val_dataset, test_dataset


def load_test_data(dataset_path, test_data):
    N_CHANNELS = 3
    N_CLASSES = 151
    SEED = 42

    test_dataset = tf.data.Dataset.list_files(dataset_path + test_data + "*", seed=SEED)
    test_dataset = test_dataset.map(parse_image)

    return test_dataset
