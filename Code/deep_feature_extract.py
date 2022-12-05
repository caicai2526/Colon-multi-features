# coding=utf-8

#author:caichengfei

caffe_root = '/home/ccf/CaffeMex_densenet_focal_loss/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import matplotlib.image as mpimg
import openslide
import time
import math
import cv2
import os
import glob
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import skimage
from skimage import measure,morphology
import matplotlib.pyplot as plt
# from scipy import misc

caffe.set_mode_gpu()
caffe.set_device(0)

start = time.time()

###################可视化#####################
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()
    plt.axis('off')

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename

#====================================================
deploy = '/home/ccf/CCF/Colorecal-cancer/SMU_data_v4_densenet/prototxt/DenseNet_201_focal_loss_deploy.prototxt'
model = '/home/ccf/CCF/Colorecal-cancer/SMU_data_v4_densenet/model_densnet_loss/caffe_colorecal_DenseNet_201_iter_100000.caffemodel'
img_dir = '/home/ccf/CCF/Colorecal-cancer/2011_survival/3000X3000/'
save_path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/deep_feature/'
meanfile = '/home/ccf/CCF/Colorecal-cancer/SMU_data_V3/data/train_lmdb/trainset_mean.binaryproto'
#=====================================================
blob = caffe.proto.caffe_pb2.BlobProto()
data_mean = open(meanfile, 'rb').read()
blob.ParseFromString(data_mean)
array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
#======================================================
net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mean_npy.mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

caffe.set_mode_gpu()
caffe.set_device(0)
#=======================================================
img_path = glob.glob(os.path.join(img_dir, '*.tif'))
for img_name in img_path:
    name = get_filename_from_path(img_name)
    image = OpenSlide(img_name)
    w, h = image.dimensions
    m = int(math.floor(h/150))
    n = int(math.floor(w/150))
    npy_feature = []
    for i in range(1, m):
        for j in range(n):
            batch_img = skimage.img_as_float(np.array(image.read_region((0+150*(i-1), 0+150*j), 0, (150, 150)))).astype(np.float32)
            net.blobs['data'].data[...] = transformer.preprocess('data', batch_img)
            out = net.forward()
            filters = net.params['conv1'][0].data
            # vis_square(filters.transpose(0, 2, 3, 1))# 可视化卷积权重和偏值
            feature = net.blobs['pool5'].data[0]
            # vis_square(feature) # 可视化特征图
            feature = feature.tolist()
            npy_feature = npy_feature + feature

            # plt.imshow(batch_img)
            # plt.show()
    np.save(save_path + name[0: 9] + '_100000.npy', npy_feature)
    print('has done...')
    print (name)