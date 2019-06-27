from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import visdom

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from tensorboardX import SummaryWriter
from test import test_net
import gc
import pdb

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


classes = np.asarray([
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
])


# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
imdb_test_name = 'voc_2007_test'
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 500

start_step = 0
end_step = 30000
lr_decay_steps = {150000}
lr_decay = 1. / 10

rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = True

remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE * 10
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load imdb and create data later
imdb = get_imdb(imdb_name)
imdb_test = get_imdb(imdb_test_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
print(net)
network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
log_dir = './tensorboardx/task2'
own_state = net.state_dict()
for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue

# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
params = list(net.parameters())
optimizer = torch.optim.SGD(params[2:], lr=lr,
                            momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logdir = os.path.join(log_dir,
                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
writer = SummaryWriter(logdir)

vis = visdom.Visdom(port='6006')

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step + 1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']

    # forward
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.item()
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps, lr,
            momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    # TODO: evaluate the model every N iterations (N defined in handout)
    if step % 5000 == 0:
        net.eval()
        aps = test_net(name = "{}_{}", net = net, imdb = imdb_test, logger = writer, step = step, visualize=True)
        if step == start_step:
            opt_map = dict(legend=['test'], title='Test mAP',
                           xlabel='Step', ylabel='mAP')
            vis_map = vis.line(Y=np.array([np.mean(aps)]), opts=opt_map)
            flag_map = 1
        else:
            vis.line(X=np.array([step]), Y=np.array(
                [np.mean(aps)]), opts=opt_map, win=vis_map, update='append')
        writer.add_scalar('test/mAP', np.mean(aps), step)
        for i, ap in enumerate(aps):
            writer.add_scalar(classes[i], ap, step)
        net.train()

        # TODO: Perform all visualizations here
        # You can define other interval variable if you want (this is just an
        # example)
        # The intervals for different things are defined in the handout
    if visualize and step % vis_interval == 0:
        # TODO: Create required visualizations
        if use_tensorboard:
            print('Logging to Tensorboard')
            writer.add_scalar('loss', train_loss / step_cnt, step)
        if use_visdom:
            print('Logging to visdom')
            if step == start_step:
                opt_loss = dict(legend=['train'], title='Train Loss',
                                xlabel='Step', ylabel='Loss')
                vis_loss = vis.line(Y=np.array(
                    [train_loss / step_cnt]), opts=opt_loss)
            else:
                vis.line(X=np.array([step]), Y=np.array(
                    [train_loss / step_cnt]), opts=opt_loss, win=vis_loss, update='append')

    if step % 2000 == 0:
        for name, param in net.named_parameters():
            writer.add_histogram(
                name, param.clone().cpu().data.numpy(), step)

    # Save model occasionally
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(
            output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX, step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
