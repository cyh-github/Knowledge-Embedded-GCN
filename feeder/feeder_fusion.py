# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
# visualization

# operation

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 joint_data_path,
                 part_data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=100,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.joint_data_path = joint_data_path
        self.part_data_path = part_data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
            #self.label = np.load(self.label_path)
        # load data
        if mmap:
            self.joint_data = np.load(self.joint_data_path, mmap_mode='r')
            self.part_data = np.load(self.part_data_path, mmap_mode='r')
        else:
            self.joint_data = np.load(self.joint_data_path)
            self.part_data = np.load(self.part_data_path)


        #self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        joint_data_numpy = np.array(self.joint_data[index])
        part_data_numpy = np.array(self.part_data[index])
        label = self.label[index]

        return joint_data_numpy, part_data_numpy, label
