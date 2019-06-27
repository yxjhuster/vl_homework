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

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class Caffenet(keras.Model):
    def __init__(self, num_classes=10):
        super(Caffenet, self).__init__(name='Caffenet')
        self.num_classes = num_classes

        self.conv1 = layers.Conv2D(filters=96,
                                   strides=4,
                                   kernel_size=[11, 11],
                                   padding="valid",
                                   activation='relu')

        self.pool1 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.conv2 = layers.Conv2D(filters=256,
                                   strides=1,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')

        self.pool2 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.conv3 = layers.Conv2D(filters=384,
                                   strides=1,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv4 = layers.Conv2D(filters=384,
                                   strides=1,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv5 = layers.Conv2D(filters=256,
                                   strides=1,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')

        self.pool3 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

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
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def randomly_flip(image, label, weight):
    image_flipped = tf.image.random_flip_left_right(image)
    # image = tf.concat([image, image_flipped],0)
    # label = tf.concat([label, label],0)
    # weight = tf.concat([weight, weight],0)
    return image_flipped, label, weight

def randomly_crop(image, label, weight):
    num_images = image.shape[0]
    image_cropped = tf.image.random_crop(image, [num_images,224, 224, 3])
    # image_cropped2 = tf.image.random_crop(image, [224, 224, 3])
    # image = tf.concat([image_cropped1, image_cropped2],0)
    # label = tf.concat([label, label],0)
    # weight = tf.concat([weight, weight],0)
    return image_cropped, label, weight

def center_crop(image, label, weight):
    image_cropped = tf.image.central_crop(image, 224./256.)
    # image_cropped2 = tf.image.central_crop(image, 224./256.)
    # image = tf.concat([image_cropped1, image_cropped2],0)
    # label = tf.concat([label, label],0)
    # weight = tf.concat([weight, weight],0)
    return image_cropped, label, weight

def mean_normalization(image, label, weight):
    mean = np.array([123.68, 116.78, 103.94])
    image_normalized = image - mean
    return image_normalized, label, weight

def test(model, dataset):
    test_loss = tfe.metrics.Mean()
    accuracy = []
    for batch, (images, labels, weights) in enumerate(dataset):
        images, labels, weights = center_crop(images, labels, weights)
        logits = model(images)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights)
        prediction = tf.math.sigmoid(logits)
        # embed()
        if np.sum(prediction.numpy()) == 0:
            pass
        else:
            accuracy.append(util.compute_ap(labels.numpy(), prediction.numpy(), weights.numpy(), average = None))
            test_loss(loss_value)
        # print(batch)
        # print(np.sum(prediction.numpy()))
    accuracy_mean = np.nanmean(accuracy)
    return test_loss.result(), accuracy_mean

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval-interval', type=int, default=250,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log-dir', type=str, default='tb/03',
                        help='path for logging directory')
    parser.add_argument('--data-dir', type=str, default='./VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/03/',
                        help='Path to checkpoints storage')
    parser.add_argument('--save-epoch', type=int, default=2,
                        help='How many batch to wait before storing checkpoints')
    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    ## TODO modify the following code to apply data augmentation here
    print('start_loading!')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    train_dataset = train_dataset.shuffle(20000).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.batch(100)

    model = Caffenet(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    learning_rate_decay = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum = 0.9)

    train_log = {'iter': [], 'loss': []}
    test_log = {'iter': [], 'loss': [], 'accuracy': []}
    print('start training!')

    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        # epoch_accuracy = tfe.metrics.Accuracy()
        for batch, (images, labels, weights) in enumerate(train_dataset):
            images, labels, weights = mean_normalization(images, labels, weights)
            images, labels, weights = randomly_crop(images, labels, weights)
            images, labels, weights = randomly_flip(images, labels, weights)
            with tf.contrib.summary.record_summaries_every_n_global_steps(100):
                tf.contrib.summary.image("sample_image", images, max_images=3)


            loss_value, grads = util.cal_grad(model,
                                              loss_func=tf.losses.sigmoid_cross_entropy,
                                              inputs=images,
                                              targets=labels,
                                              weights=weights)
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables),
                                      global_step)
            learning_rate_decay = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5)
            with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                    tf.contrib.summary.scalar('learning_rate', learning_rate_decay())
            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                for grad, var in zip(grads,model.trainable_variables):
                    tf.contrib.summary.histogram("{}/grad_histogram".format(var.name), grad)

            with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                tf.contrib.summary.scalar('training_loss', loss_value)

            epoch_loss_avg(loss_value)
            
            if global_step.numpy() % args.log_interval == 0:
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}'
                                                        .format(ep,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         epoch_loss_avg.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())
                # tf.contrib.summary.scalar('training_loss', epoch_loss_avg.result())
                # train_log['accuracy'].append(epoch_accuracy.result())
            if global_step.numpy() % args.eval_interval == 0:
                test_loss, test_acc = test(model, test_dataset)
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.eval_interval):
                    tf.contrib.summary.scalar('testing_acc', test_acc)
                test_log['iter'].append(global_step.numpy())
                test_log['loss'].append(test_loss)
                test_log['accuracy'].append(test_acc)
                # tf.contrib.summary.scalar('testing_loss', test_loss)
                # tf.contrib.summary.scalar('testing_loss', test_acc)
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Testing Loss:{3:.4f} Testing Accuracy:{4:.4f}'.format(ep,
                                                                            args.epochs, global_step.numpy(), test_loss, test_acc))
        if ep % args.save_epoch == 0:
            # checkpoint = tfe.Checkpoint(optimizer=optimizer,
            #                     model=model,
            #                     optimizer_step=tf.train.get_or_create_global_step())
            checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt" + str(ep) + '.h5')
            model.save_weights(checkpoint_prefix)

    AP, mAP = util.eval_dataset_map(model, test_dataset)
    rand_AP = util.compute_ap(
        test_labels, np.random.random(test_labels.shape),
        test_weights, average=None)

    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                model=model,
                                optimizer_step=tf.train.get_or_create_global_step())
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt")
    checkpoint.save(file_prefix=checkpoint_prefix)

    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = util.compute_ap(test_labels, test_labels, test_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    print('Obtained {} mAP'.format(mAP))
    print('Per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, util.get_el(AP, cid)))
    writer.close()


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()