# coding=utf-8

import numpy as np
import os
import scipy.io as sio
import glob

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename

file_path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/deep_feature/'
save_path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/deep_feature_mat/'

file_name_list = glob.glob(os.path.join(file_path, '*.npy'))
for file_name in file_name_list:
    file_npy = np.load(file_name)
    name = get_filename_from_path(file_name)
    sio.savemat(save_path + name, {'label': file_npy})
    print 'sucess'


