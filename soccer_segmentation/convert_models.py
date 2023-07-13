import tensorflow as tf
from keras.metrics import Recall, Precision, IoU
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

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


def convert():
    path_models = ROOT_DIR + "/models/"
    #m = segnet_resnet18(input_shape=(256, 256, 3), n_labels=3)
    #m = segnet(input_shape=(256, 256, 3), n_labels=3)
    #m = Unet(input_shape=(256, 256, 3), n_labels=3, dropout=0.2)
    #m = segnet_resnet50(input_shape=(256, 256, 3), n_labels=3)
    #m = unet_resnet18(input_shape=(256, 256, 3), n_labels=3)
    #m = unet_resnet50(input_shape=(256, 256, 3), n_labels=3, dropout=0.2)
    #m = segnet_mobilenetsmall(input_shape=(256, 256, 3), n_labels=3)
    #m = segnet_mobilenetlarge(input_shape=(256, 256, 3), n_labels=3)
    #m = unet_mobilenetv3small(input_shape=(256,256,3), n_labels=3)
    #m = unet_mobilenetv3large(input_shape=(256, 256, 3), n_labels=3)
    #m = segnet_vgg16(input_shape=(256, 256, 3), n_labels=3)
    #m = segnet_vgg19(input_shape=(256, 256, 3), n_labels=3)
    #m = unet_vgg16(input_shape=(256, 256, 3), n_labels=3)
    m = unet_vgg19(input_shape=(256, 256, 3), n_labels=3)
    model_name = "unet_vgg19"

    LR = 1e-4
    metrics = ['sparse_categorical_accuracy', IoU(num_classes=3, target_class_ids=[1], sparse_y_pred=False), IoU(num_classes=3, target_class_ids=[2], sparse_y_pred=False)]

    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='sparse_categorical_crossentropy', metrics=metrics)
    m.load_weights(ROOT_DIR + "/files/best_" + model_name + ".h5")

    full_model = tf.function(lambda x: m(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(m.inputs[0].shape, m.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=path_models,
                      name=model_name+".pb",
                      as_text=False)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=path_models,
                      name=model_name + ".pbtxt",
                      as_text=True)
