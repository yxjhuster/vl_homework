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

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

import h5py

f = h5py.File('./checkpoints/05/ckpt.h5')

class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='SimpleCNN')
        self.num_classes = num_classes
        self.conv1_1 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d']['SimpleCNN/conv2d/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d']['SimpleCNN/conv2d/bias:0'].value))
        # print(self.conv1_1.get_weights()[0].shape)

        self.conv1_2 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_1']['SimpleCNN/conv2d_1/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_1']['SimpleCNN/conv2d_1/bias:0'].value))

        self.pool1 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.conv2_1 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_2']['SimpleCNN/conv2d_2/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_2']['SimpleCNN/conv2d_2/bias:0'].value))
        # print(self.conv1_1.get_weights()[0].shape)
        self.conv2_2 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_3']['SimpleCNN/conv2d_3/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_3']['SimpleCNN/conv2d_3/bias:0'].value))
        # print(self.conv1_1.get_weights()[0].shape)
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.conv3_1 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_4']['SimpleCNN/conv2d_4/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_4']['SimpleCNN/conv2d_4/bias:0'].value))
        # print(self.conv1_1.get_weights()[0].shape)
        self.conv3_2 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_5']['SimpleCNN/conv2d_5/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_5']['SimpleCNN/conv2d_5/bias:0'].value))
        self.conv3_3 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_6']['SimpleCNN/conv2d_6/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_6']['SimpleCNN/conv2d_6/bias:0'].value))
        # print(self.conv1_1.get_weights()[0].shape)

        self.pool3 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.conv4_1 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_7']['SimpleCNN/conv2d_7/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_7']['SimpleCNN/conv2d_7/bias:0'].value))

        self.conv4_2 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_8']['SimpleCNN/conv2d_8/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_8']['SimpleCNN/conv2d_8/bias:0'].value))
        self.conv4_3 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_9']['SimpleCNN/conv2d_9/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_9']['SimpleCNN/conv2d_9/bias:0'].value))

        self.pool4 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.conv5_1 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_10']['SimpleCNN/conv2d_10/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_10']['SimpleCNN/conv2d_10/bias:0'].value))
        self.conv5_2 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_11']['SimpleCNN/conv2d_11/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_11']['SimpleCNN/conv2d_11/bias:0'].value))
        self.conv5_3 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_initializer = keras.initializers.Constant(f['conv2d_12']['SimpleCNN/conv2d_12/kernel:0'].value),
                                   bias_initializer=keras.initializers.Constant(f['conv2d_12']['SimpleCNN/conv2d_12/bias:0'].value))

        self.pool5 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu',
                                  kernel_initializer = keras.initializers.Constant(f['dense']['SimpleCNN/dense/kernel:0'].value),
                                  bias_initializer=keras.initializers.Constant(f['dense']['SimpleCNN/dense/bias:0'].value))
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu',
                                  kernel_initializer = keras.initializers.Constant(f['dense_1']['SimpleCNN/dense_1/kernel:0'].value),
                                  bias_initializer=keras.initializers.Constant(f['dense_1']['SimpleCNN/dense_1/bias:0'].value))
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes,
                                  kernel_initializer = keras.initializers.Constant(f['dense_2']['SimpleCNN/dense_2/kernel:0'].value),
                                  bias_initializer=keras.initializers.Constant(f['dense_2']['SimpleCNN/dense_2/bias:0'].value))

    def call(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)

        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)

        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)

        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.pool5(x)

        flat_x = self.flat(x)
        out = flat_x
        # out = self.dense1(flat_x)
        # out = self.dropout1(out, training=training)
        # out = self.dense2(out)
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

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
test_dataset = test_dataset.batch(100)


model = Caffenet(num_classes=len(CLASS_NAMES))

f.close()

size=test_images.shape[0]

near=[]
t=[]

m=size%100
outputs=[]

# collect the output of the network

for batch, (images, labels, weights) in enumerate(train_dataset):
    images = randomly_crop(images)
    logits = model(images)
    outputs.extend(logits.numpy())


# select 20 different image with different classification
for i in range(20):
    for j in range(size):
        if test_labels[j,:,i]==1:
            if j not in t: 
                t.append(j)
                break

# nearest neighbour
for i in t:
    test=layer[i, :]
    samples=np.delete(layer,i,axis=0)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples) 
    nearest=neigh.kneighbors(test.reshape(1,-1))[1][0][0]
    if nearest>=i:
        nearest=nearest+1
    near.append(nearest)

for i in t:
    plt.imshow(test_images[i,:,:,:])
    plt.savefig(str(i)+'origin.jpg')
for i, j in zip(near, t):
    plt.imshow(test_images[i,:,:,:])
    plt.savefig(str(i)+'near.jpg')
