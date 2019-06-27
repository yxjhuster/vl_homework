import h5py
import numpy as np
import matplotlib

matplotlib.use('agg') 
import matplotlib.pyplot as plt

from IPython import embed

def plot_filter(weight_file_path):
    s = h5py.File(weight_file_path)
    kernel = s['conv2d']['Caffenet/conv2d/kernel:0'].value
    bias = s['conv2d']['Caffenet/conv2d/bias:0'].value
    s.close()
    # print(kernel.shape)
    # x_min = kernel.min()
    # x_max = kernel.max()
    # kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
    kernel_0_to_1 = kernel
    kernel_0_to_1 = np.transpose(kernel_0_to_1, (3,0,1,2))
    # embed()
    # idx = np.random.randint(96, size=1)
    idx = [18]
    plt.imshow(kernel_0_to_1[idx[0]])
    plt.savefig('conv2d1_' + str(idx) + '.png')

    idx = [39]
    plt.imshow(kernel_0_to_1[idx[0]])
    plt.savefig('conv2d1_' + str(idx) + '.png')

    idx = [48]
    plt.imshow(kernel_0_to_1[idx[0]])
    plt.savefig('conv2d1_' + str(idx) + '.png')