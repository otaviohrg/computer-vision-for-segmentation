import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def display_sample(img_list):
    plt.figure(figsize=(18,18))
    title = ['Input Image', "True Mask", "Predicted Mask"]

    for i in range(len(img_list)):
        plt.subplot(1, len(img_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img_list[i]))
        plt.axis('off')

    plt.show()


def create_mask(pred_mask):
    """
        [h,w,class] ==> [h,w] ==> [h,w,1]
    """
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = tf.expand_dims(pred_mask, axis= -1)
    return pred_mask


def show_prediction(model , dataset = None, num = 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], mask, create_mask(pred_mask)])
    else:
        print("\n\n[-- INFO --] No Dataset Provided !\n\n")


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x