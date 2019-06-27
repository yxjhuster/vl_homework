import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras

from PIL import Image

def set_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    return session


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_pascal(data_dir, class_names, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        class_names (list): list of class names
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 256px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    ## TODO Implement this function
    labels_path = data_dir + '/ImageSets/Main'
    images_path = data_dir + '/JPEGImages'

    name_path = labels_path + '/' + class_names[0] + '_' + split + '.txt'
    index_list = np.genfromtxt(name_path, dtype='str')[:,0]

    labels_list = []
    # load label information
    for names in class_names:
        name_path = labels_path + '/' + names + '_' + split + '.txt'
        labels_list.append(np.int32(np.loadtxt(name_path)[:,1]))
    labels_list = np.array(labels_list)

    labels = np.where(labels_list >= 1, np.int32(1), np.int32(0))
    labels = labels.T

    weights = np.where(labels_list <= -1,  np.int32(1),  np.int32(0)) + np.where(labels_list >= 1, np.int32(1), np.int32(0))
    weights = weights.T

    images = []
    for index in index_list:
        sampled_image = Image.open(images_path + '/' + str(index) + '.jpg')
        sampled_image = sampled_image.resize((256, 256))
        sampled_image_array = np.array(sampled_image, dtype= np.float32)
        images.append(sampled_image_array)
    images = np.array(images)

    return images, labels, weights


def cal_grad(model, loss_func, inputs, targets, weights=1.0):
    """
    Return the loss value and gradients
    Args:
         model (keras.Model): model
         loss_func: loss function to use
         inputs: image inputs
         targets: labels
         weights: weights of the samples
    Returns:
         loss and gradients
    """

    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_value = loss_func(targets, logits, weights)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    ## TODO implement the code here
    AP = []
    for batch, (images, labels, weights) in enumerate(dataset):
        images, labels, weights = center_crop(images, labels, weights)
        logits = model(images)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights)
        prediction = tf.math.sigmoid(logits)
        # embed()
        if np.sum(prediction.numpy()) == 0:
            pass
        else:
            AP.append(np.nanmean(compute_ap(labels.numpy(), prediction.numpy(), weights.numpy(), average = None)))
    mAP = np.nanmean(AP)
    return AP, mAP

def center_crop(image, label, weight):
    image_cropped = tf.image.central_crop(image, 224./256.)
    # image_cropped2 = tf.image.central_crop(image, 224./256.)
    # image = tf.concat([image_cropped1, image_cropped2],0)
    # label = tf.concat([label, label],0)
    # weight = tf.concat([weight, weight],0)
    return image_cropped, label, weight


def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr