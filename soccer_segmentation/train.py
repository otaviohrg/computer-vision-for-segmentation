from soccer_segmentation.models.segnet_resnet18 import segnet_resnet
from soccer_segmentation.models.segnet import segnet
from soccer_segmentation.models.unet import Unet
import tensorflow as tf
import numpy as np
from glob import glob
from keras.metrics import Recall, Precision, IoU
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import json


from soccer_segmentation.data import load_data
from soccer_segmentation.preprocessing import load_image_train, load_image_test
from utils.visualize import *
from definitions import ROOT_DIR

dataset_path = ROOT_DIR + "/data/processed_data/"
training_data = "train_images/"
val_data = "val_images/"
test_data = "test_images/"

TRAINSET_SIZE = len(glob(dataset_path + training_data + "*"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

VALSET_SIZE = len(glob(dataset_path + val_data + "*"))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

TESTSET_SIZE = len(glob(dataset_path + test_data + "*"))
print(f"The Test Dataset contains {TESTSET_SIZE} images.")

train_dataset, val_dataset, test_dataset = load_data(dataset_path, training_data, val_data, test_data)

LR = 1e-4
EPOCHS = 50
metrics = ['sparse_categorical_accuracy', IoU(num_classes=3, target_class_ids=[1], sparse_y_pred=False), IoU(num_classes=3, target_class_ids=[2], sparse_y_pred=False)]
#metrics = ['sparse_categorical_accuracy']
BATCH_SIZE = 4
BUFFER_SIZE = 1000
N_CHANNELS = 3
N_CLASSES = 3
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

# -- Train Dataset --#
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#-- Validation Dataset --#
dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

#-- Test Dataset --#
dataset['test'] = dataset['test'].map(load_image_test)
dataset['test'] = dataset['test'].batch(BATCH_SIZE)
dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)


#m = segnet_resnet(input_shape=(256, 256, 3), n_labels=3)
#m = segnet(input_shape=(256, 256, 3), n_labels=3)
m = Unet(input_shape=(256, 256, 3), n_labels=3, dropout=0.2)

#for layer in m.layers:
#    layer.trainable = True

m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='sparse_categorical_crossentropy', metrics=metrics)

callbacks = [
    ModelCheckpoint(ROOT_DIR + "/files/unet.h5"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
    CSVLogger(ROOT_DIR + "/files/data.csv"),
    TensorBoard(),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
]

STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

#STEPS_PER_EPOCH = 10
#VALIDATION_STEPS = 5

#m.load_weights(ROOT_DIR + "/files/resnet_segnet.h5")

history = m.fit(
    dataset['train'],
    epochs=10,
    validation_data=dataset['val'],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks
)

#history_dict = history.history
# Save it under the form of a json file
#json.dump(history_dict, open(ROOT_DIR + "/files/segnet_history.txt", 'w'))

#evaluation = m.evaluate(dataset["test"], return_dict=True)

#for name, value in evaluation.items():
#  print(f"{name}: {value:.4f}")
