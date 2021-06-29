import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as io
import sys
from time import time
import h5py
from scipy import sparse
from args import args, device
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return args.nmc
    def __getitem__(self, idx):
        idx = idx+1
        dir = '/KLE_100_mcs'
        #input
        self.input = io.loadmat(dir+'/input_1_test/input_1_%d.mat'%idx)
        self.input = self.input['input_1']
        self.input = np.transpose(self.input)

        if args.data == 'KLE':
            self.input = np.log(self.input)
        elif args.data == 'channel':
            self.input = self.input
        self.input = torch.from_numpy(self.input)
        
        #basis_patch
        self.basis_patch = io.loadmat(dir+'/output/output_1_%d.mat'%idx)
        self.basis_patch = self.basis_patch['output_1']  
        self.basis_patch = np.transpose(self.basis_patch)
        self.basis_patch = torch.from_numpy(self.basis_patch) 
        #A-matrix
        self.A_matrix = io.loadmat(dir+'/A_matrix_test/A_matrix1_%d.mat'%idx)
        self.A_matrix = torch.from_numpy(self.A_matrix['A_matrix1'])
        #q-matrix
        self.q_matrix = io.loadmat(dir+'/q_matrix_test/q_matrix1_%d.mat'%idx)
        self.q_matrix = self.q_matrix['q_matrix1']
        #B-matrix
        self.B_matrix = io.loadmat(dir+'/B_matrix_test/B_matrix_%d.mat'%idx)
        self.B_matrix = self.B_matrix['Extract_B']
        #target-matrix
        self.target_P = io.loadmat(dir+'/P_value_test/P_value_%d.mat'%idx)
        self.target_P = self.target_P['P_value']
        return self.input, self.basis_patch, self.A_matrix, self.B_matrix, self.target_P, self.q_matrix

def mcs_load_data():
    

    kwargs = {'num_workers': 4,
                'pin_memory': True} if torch.cuda.is_available() else {}
    s=Dataset()
    loader= torch.utils.data.DataLoader(s,
                                        batch_size=100, shuffle=False,
                                        **kwargs)
    return loader

