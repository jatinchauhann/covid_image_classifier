from tensorflow.keras.models import Sequential, Model, load_model
import efficientnet.tfkeras as efn

import os

import cv2

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shutil

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, Input, \
    MaxPooling2D, GlobalMaxPooling2D, concatenate, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import precision_score, recall_score

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import gc
from focal_loss import BinaryFocalLoss

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score


def function_pred_grad_cam(IMAGE_PATH):
    data_gen = ImageDataGenerator(rescale=1 / 255)

    img = load_img(IMAGE_PATH, target_size=(348, 348))  # this is a PIL image

    x = img_to_array(img)

    x = x.reshape((1,) + x.shape)

    test_generator = data_gen.flow(x, batch_size=1)

    model_load = load_model('NEW_B3_model_2.hdf5')

    pred_value = model_load.predict(test_generator, verbose=1)

    """
    if pred_value.argmax(axis =1)[0] == 0:
        print("COVID")
    else:
        print("NON-COVID") 

    print(pred_value)
    print(pred_value.max(axis =1))

    """
    ### ---------- ### -----------#### ---------

    LAYER_NAME = 'top_conv'
    COVID_CLASS_INDEX = 0

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(348, 348))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model = tf.keras.Model(model_load.inputs, model_load.get_layer(LAYER_NAME).output)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, COVID_CLASS_INDEX]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (348, 348))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT)
    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    cv2.imwrite('cam.png', output_image)
    # The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
    # Import image

    image = cv2.imread("cam.png")
    img_ = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(348, 348))

    # The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
    # Show the image with matplotlib

    plt.imshow(img_)
    # plt.show()
    plt.savefig('input.png')
    # Show the image with matplotlib
    plt.imshow(image)
    # plt.show()
    plt.savefig('output.png')

    return "DONE"