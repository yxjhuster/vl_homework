from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.contrib import eager as tfe

import util

from IPython import embed

from sklearn.neighbors import NearestNeighbors

import scipy
from scipy import ndimage
import scipy.misc

import matplotlib

matplotlib.use('agg') 
import matplotlib.pyplot as plt

import scipy.misc
from sklearn.manifold import TSNE

from operator import itemgetter

from collections import OrderedDict

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

import h5py

f = h5py.File('./checkpoints/03/ckpt58.h5')

class Caffenet(keras.Model):
    def __init__(self, num_classes=10):
        super(Caffenet, self).__init__(name='Caffenet')
        self.num_classes = num_classes

        self.conv1 = layers.Conv2D(filters=96,
                                   strides=4,
                                   kernel_size=[11, 11],
                                   padding="valid",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d']['Caffenet/conv2d/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d']['Caffenet/conv2d/bias:0'].value))

        self.pool1 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.conv2 = layers.Conv2D(filters=256,
                                   strides=1,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_1']['Caffenet/conv2d_1/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_1']['Caffenet/conv2d_1/bias:0'].value))

        self.pool2 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.conv3 = layers.Conv2D(filters=384,
                                   strides=1,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_2']['Caffenet/conv2d_2/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_2']['Caffenet/conv2d_2/bias:0'].value))
        self.conv4 = layers.Conv2D(filters=384,
                                   strides=1,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_3']['Caffenet/conv2d_3/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_3']['Caffenet/conv2d_3/bias:0'].value))
        self.conv5 = layers.Conv2D(filters=256,
                                   strides=1,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_4']['Caffenet/conv2d_4/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_4']['Caffenet/conv2d_4/bias:0'].value))

        self.pool3 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu',
                                    kernel_initializer = keras.initializers.Constant(f['dense']['Caffenet/dense/kernel:0'].value),
                                    bias_initializer=keras.initializers.Constant(f['dense']['Caffenet/dense/bias:0'].value))
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu',
                                    kernel_initializer = keras.initializers.Constant(f['dense_1']['Caffenet/dense_1/kernel:0'].value),
                                    bias_initializer=keras.initializers.Constant(f['dense_1']['Caffenet/dense_1/bias:0'].value))
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes,
                                  kernel_initializer = keras.initializers.Constant(f['dense_2']['Caffenet/dense_2/kernel:0'].value),
                                  bias_initializer=keras.initializers.Constant(f['dense_2']['Caffenet/dense_2/bias:0'].value))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        flat_x = self.flat(x)
        # out = flat_x
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        # out = self.dropout2(out, training=training)
        # out = self.dense3(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def randomly_crop(image):
    num_images = image.shape[0]
    image_cropped = tf.image.random_crop(image, [num_images,224, 224, 3])
    # image_cropped2 = tf.image.random_crop(image, [224, 224, 3])
    # image = tf.concat([image_cropped1, image_cropped2],0)
    # label = tf.concat([label, label],0)
    # weight = tf.concat([weight, weight],0)
    return image_cropped

tf.enable_eager_execution()
data_dir='VOCdevkit/VOC2007/'
print('start loading!')
test_images, test_labels, test_weights = util.load_pascal(data_dir,
                                                          class_names=CLASS_NAMES,
                                                          split='test')
# print(test_weights[0])
# embed()
print('finish loading!')
model = Caffenet(num_classes=len(CLASS_NAMES))

f.close()

size=test_images.shape[0]
layer_output=np.zeros((size,4096))

sample = test_images[0:1000,:,:,:]
sample = randomly_crop(sample)

logits = model(sample)

labels = test_labels[0:1000,:]

color = np.random.rand(20,3)

color_list = []

l=[]

for i in labels:
    label_list = np.argwhere(i >= 1)
    l.append(np.where(i==1)[0][0])
    temp_color = np.zeros((1,3))
    for i in label_list:
        temp_color=temp_color+color[i,:]

    color_list.append(tuple(temp_color/len(label_list)))

labels = []

label=list(itemgetter(*l)(CLASS_NAMES))

projection=TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(logits)

fig, ax = plt.subplots(figsize=(10, 10))
for i in range(0,1000):
    ax.scatter(projection[i, 0], projection[i, 1],c=color_list[i],label=label[i],alpha=0.8)

#  code refered from Jianren Wang for ploting the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

ax.grid(True)
plt.title('tSNE Projection')
plt.savefig('tSNE.jpg')

