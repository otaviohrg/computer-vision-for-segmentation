from glob import glob
from collections import Counter

import cv2
import numpy as np
from keras.metrics import Recall, Precision, IoU
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import tensorflow as tf
from tqdm.keras import TqdmCallback


from .data import load_data, load_test_data
from .models.segnet_resnet18 import segnet_resnet18
from .models.segnet import segnet
from .models.unet import Unet
from .models.segnet_resnet50 import segnet_resnet50
from .models.unet_resnet18 import unet_resnet18
from .models.unet_resnet50 import unet_resnet50
from .models.segnet_mobilenetv3small import segnet_mobilenetsmall
from .models.segnet_mobilenetv3large import segnet_mobilenetlarge
from .models.unet_mobilenetv3small import unet_mobilenetv3small
from .models.unet_mobilenetv3large import unet_mobilenetv3large
from .models.segnet_vgg16 import segnet_vgg16
from .models.segnet_vgg19 import segnet_vgg19
from .models.unet_vgg16 import unet_vgg16
from .models.unet_vgg19 import unet_vgg19

from definitions import ROOT_DIR
from .preprocessing import load_image_test

def test():
    models = [
        #{"m": segnet(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet"},
        #{"m": segnet_resnet18(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet_resnet18"},
        #{"m": segnet_resnet50(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet_resnet50"},
        #{"m": segnet_mobilenetsmall(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet_mobilenetsmall"},
        #{"m": segnet_mobilenetlarge(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet_mobilenetlarge"},
        #{"m": segnet_vgg16(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet_vgg16"},
        #{"m": segnet_vgg19(input_shape=(256, 256, 3), n_labels=3),
        # "name": "segnet_vgg19"},
        #{"m": Unet(input_shape=(256, 256, 3), n_labels=3, dropout=0.2),
        # "name": "unet"},
        #{"m": unet_resnet18(input_shape=(256, 256, 3), n_labels=3),
        # "name": "unet_resnet18"},
        #{"m": unet_resnet50(input_shape=(256, 256, 3), n_labels=3),
        # "name": "unet_resnet50"},
        #{"m": unet_mobilenetv3small(input_shape=(256, 256, 3), n_labels=3),
        # "name": "unet_mobilenetv3small"},
        {"m": unet_mobilenetv3large(input_shape=(256, 256, 3), n_labels=3),
         "name": "unet_mobilenetv3large"},
        #{"m": unet_vgg16(input_shape=(256, 256, 3), n_labels=3),
        # "name": "unet_vgg16"},
        #{"m": unet_vgg19(input_shape=(256, 256, 3), n_labels=3),
        # "name": "unet_vgg19"}
    ]

    dataset_path = ROOT_DIR + "/data/processed_data/"
    test_data = "test_images/"
    val_data = "val_images/"

    TESTSET_SIZE = len(glob(dataset_path + test_data + "*"))
    print(f"The Test Dataset contains {TESTSET_SIZE} images.")

    test_dataset = load_test_data(dataset_path, test_data)
    val_dataset = load_test_data(dataset_path, val_data)

    LR = 1e-4
    EPOCHS = 50
    metrics = ['sparse_categorical_accuracy', IoU(num_classes=3, target_class_ids=[0], sparse_y_pred=False), IoU(num_classes=3, target_class_ids=[2], sparse_y_pred=False)]

    # metrics = ['sparse_categorical_accuracy']
    BATCH_SIZE = 4
    BUFFER_SIZE = 1000
    N_CHANNELS = 3
    N_CLASSES = 3
    SEED = 42
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # -- Test Dataset --#
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    #-- Validation Dataset --#
    val_dataset = val_dataset.map(load_image_test)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)


    for model in models:
        model["m"].compile(optimizer=tf.keras.optimizers.Adam(LR), loss='sparse_categorical_crossentropy', metrics=metrics)

        callbacks = [
            ModelCheckpoint(ROOT_DIR + "/files/" + model["name"] + ".h5"),
            ModelCheckpoint(ROOT_DIR + "/files/best_" + model["name"] + ".h5", monitor='val_loss', save_best_only=True,
                            save_weights_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
            CSVLogger(ROOT_DIR + "/files/" + model["name"] + "_test_data.csv"),
            TensorBoard(),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False),
            TqdmCallback(verbose=2)
        ]

        STEPS_PER_EPOCH = TESTSET_SIZE // BATCH_SIZE
        model["m"].load_weights(ROOT_DIR + "/files/best_" + model["name"] + ".h5")



        # history_dict = history.history
        # Save it under the form of a json file
        # json.dump(history_dict, open(ROOT_DIR + "/files/segnet_history.txt", 'w'))

        evaluation = model["m"].evaluate(test_dataset, return_dict=True)
        evaluation = model["m"].evaluate(val_dataset, return_dict=True)

        #x = model["m"].predict(test_dataset)
        #y = np.zeros_like(x[0,:,:])
        #y[np.arange(len(x[0,:,:])), x[0,:,:].argmax(1)] = 1
        #y = y.reshape((256,256,3))*100
        #y[:,:,1] = np.zeros((256,256))
        #y[:, :, 0] = y[:,:,0] + y[:,:,2]
        #y[:, :, 1] = y[:, :, 1] + y[:,:,0]
        #y[:, :, 1] = y[:, :, 1] + y[:, :, 2]
        #y = cv2.resize(y, (200, 150))
        #cv2.imwrite(dataset_path+model["name"]+".png", y)
        #with open(ROOT_DIR + "/files/test_results.txt", "a") as file:
        #    file.write("\n" + model["name"] + "\n")
        #    file.write(str(evaluation) + "\n")

        #x = test_dataset.take(1).get_single_element()
        #print(np.unique(np.array(x[0][0, :, :, :]), return_counts=True))
        #model["m"].predict(x[0])
