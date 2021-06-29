"""
Stein Variational Gradient Descent forHM-DenseED.
Reference: https://github.com/zabaras/cnn-surrogate/tree/master/models
"""  

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.misc import log_sum_exp, parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')
from time import time
import copy
import sys
import os
import gc
import math
import scipy.io as io
from args import args, device
from models.bdsmm import bdsmm
from torch.autograd import Variable
n_samples = args.n_samples
lr = args.lr
lr_noise = args.lr_noise
ntrain = args.ntrain
dir = './models'

class Bayesian_model_train(object):
    """Base class for Stein Variational Gradient Descent, with for-loops...
    The Bayesian neural network is defined in `Bayesian_model_NN.BayesNN` class.    

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, Bayesian_model_NN, train_loader):
        """
        For-loop implementation of SVGD.

        Args:
            Bayesian_model_NN (nn.Module): Bayesian NN
            train_loader (utils.data.DataLoader): Training data loader
            logger (dict)

        """
        self.Bayesian_model_NN = Bayesian_model_NN
        self.train_loader = train_loader
        self.n_samples = n_samples
        self.optimizers, self.schedulers = self._optimizers_schedulers(
                                            lr, lr_noise)

    def train(self, epoch):
        print('epoch..............................................',epoch)

        self.Bayesian_model_NN.train()
        mse2 = 0.
        mse_train = 0.
        for batch_idx, (input,basis_patch,A_matrix, B_matrix,target_P, q_matrix) in enumerate(self.train_loader):
            input_rr12,output_basis12,A1_transformed12,B1_transformed12, target_pressure12, q1_transformed12 = input.to(device),basis_patch.to(device),A_matrix.to(device),B_matrix.to(device), target_P.to(device), q_matrix.to(device)
            input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed = input_rr12.float(),output_basis12.float(),A1_transformed12.float(),B1_transformed12.float(), target_pressure12.float(), q1_transformed12.float()
            #================================================================================
            output_basis = output_basis.view(144,1,15,15)
            input_rr = input_rr.view(144,1,15,15)
            A_app  = []
            for i in range(1):
                A_torch = A1_transformed1[i,:,:]
                A_torch1 = A_torch[:,0:2]
                A_torch2 = A_torch[:,2]
                A_torch1 = A_torch1.type(torch.LongTensor).to(device)
                A_torch_final = torch.sparse.FloatTensor(A_torch1.t(), A_torch2, torch.Size([16384,16384]))
                A_app.append(A_torch_final)
            A1_transformed = torch.stack(A_app,dim=0)
            #================================================
            C = io.loadmat(dir+'/matlab_index_save_1.mat')
            C = C['basis_save']
            C = np.squeeze(C)
            X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1]))
            for i in range(X.shape[0]):
                X[i] = C[i]
            # -1 because of matlab and python
            X1 = X.reshape(144,225)-1

            X2 = np.zeros((144,225))
            for i in range(144):
                var2 = np.zeros((15,15))
                ele = X1[i,0]
                for varu in range(15):
                    var1 = ele+128*(varu)
                    for vm in range(15):
                        var2[varu,vm] = var1+vm
                var3 = var2.reshape(1,225)
                X2[i,:] = var3 
            X2 = torch.Tensor(X2)
            target_BB = output_basis
            output_pr = []
            # all gradients of log joint probability: (S, P)
            grad_log_joint = []
            # all model parameters (particles): (S, P)
            theta = []
            # store the joint probabilities
            log_joint = 0.
            for i in range(self.n_samples):
                B1_transformed = Variable(B1_transformed,requires_grad=False)
                self.Bayesian_model_NN[i].zero_grad()

                X2_i = X2
                q1_transformed_i = q1_transformed
                B1_transformed_i = B1_transformed
                output_i = self.Bayesian_model_NN[i].forward(input_rr)               
                output_i = output_i.view(1,144,225)
                target_BB = target_BB.view(1,144,225)
                loss1 = F.mse_loss(output_i, target_BB, size_average=False)              

                #============================================================================
                for RRRR in range (1):
                    output_RR_i = output_i[RRRR,:,:]    
                    output_RR_i = output_RR_i.reshape(144,225)
                    B1_p_out_i = B1_transformed[RRRR,:,:]
                    B1_p_out_i = B1_p_out_i.reshape(16384, 256)
                    ss_i = 0
                    for ii in range(12):
                        jjj = 35+16*ii
                        for k in range (jjj,jjj+12):
                            ss_i =ss_i+1
                            s_i = ss_i-1
                            basis_temp_i = X2_i[s_i,:]
                            tem_i = B1_p_out_i[:,k-1]
                            basis_temp_i = basis_temp_i.type(torch.LongTensor)
                            tem_i = tem_i.type(torch.cuda.FloatTensor)
                            temp_variable_i = output_RR_i[143-s_i,:]/torch.max(output_RR_i[143-s_i,:])
                            tem_i[basis_temp_i] = temp_variable_i
                            B1_p_out_i[:,k-1] = tem_i               
                    B1_transformed[RRRR,:,:] = B1_p_out_i
                #====================================================
                temp11 = []
                for kkk in range(1):
                    B1_transformed_temp = B1_transformed[kkk,:,:]
                    B1_transformed1 = torch.transpose(B1_transformed_temp,0,1)
                    dim = torch.sum(B1_transformed1,dim=0)
                    B1_transformed2 = torch.div(B1_transformed1,dim)
                    B1_transformed22 = torch.transpose(B1_transformed2,0,1)
                    temp11.append(B1_transformed22)
                B1_transformed = temp11
                B1_transformed = torch.stack(temp11,dim=0)
                #============================================ 

                R1_transformed = torch.transpose(B1_transformed,1,2) #check here
                A1_transformed = torch.transpose(A1_transformed,1,2)
                R1_transformed = torch.transpose(R1_transformed,1,2)
                A_c_transformed = torch.matmul(torch.transpose(bdsmm(A1_transformed,R1_transformed),1,2),B1_transformed)
                R1_transformed = torch.transpose(R1_transformed,1,2) #transform back to the old R1_transformed
                temp1_transformed = torch.matmul(R1_transformed,q1_transformed)
                temp2_transformed,LU = torch.solve(temp1_transformed,A_c_transformed)
                temp3_transformed = torch.matmul(B1_transformed,temp2_transformed)
                predict_pressure = temp3_transformed
                target_pressure = target_pressure.view(1,1,128,128)
                predict_pressure = predict_pressure.view(1,1,128,128)
                loss2 = F.mse_loss(predict_pressure,target_pressure, size_average=False)
                predict_pressure12 = predict_pressure.cpu().detach()
                output_pr.append(predict_pressure12)
                #==========
                predict_pressure_i = predict_pressure
                log_joint_i = self.Bayesian_model_NN._log_joint(i, predict_pressure_i, target_pressure, ntrain)

                # backward to compute gradients of log joint probabilities
                log_joint_i.backward()
                # monitoring purpose
                log_joint += log_joint_i.item()
                # for name, param in self.Bayesian_model_NN[i].named_parameters():
                #     if param.requires_grad:
                #         print (name)
                # backward frees memory for computation graph
                # computation below does not build computation graph
                # extract parameters and their gradients out from models
                vec_param, vec_grad_log_joint = parameters_to_vector(
                    self.Bayesian_model_NN[i].parameters(), both=True)
                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                theta.append(vec_param.unsqueeze(0))
            output_pr = torch.stack(output_pr,dim=0)
            # calculating the kernel matrix and its gradients
            theta = torch.cat(theta)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint)
            # this line needs S x P memory
            grad_logp = torch.mm(Kxx, grad_log_joint)
            # negate grads here!!!
            grad_theta = - (grad_logp + dxKxx) / self.n_samples
            # explicitly deleting variables does not release memory :(
       
            # update param gradients
            for i in range(self.n_samples):
                vector_to_parameters(grad_theta[i],
                    self.Bayesian_model_NN[i].parameters(), grad=True)
                self.optimizers[i].step()
            # WEAK: no loss function to suggest when to stop or
            # approximation performance
            output_tr = target_pressure.cpu().detach()
            mse2 += F.mse_loss(output_pr.mean(0), output_tr).item()
        rmse_train2 = np.sqrt(mse2 / len(self.train_loader))
        return rmse_train2

    def _squared_dist(self, X):
        """Computes squared distance between each row of `X`, ||X_i - X_j||^2

        Args:
            X (Tensor): (S, P) where S is number of samples, P is the dim of 
                one sample

        Returns:
            (Tensor) (S, S)
        """
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)


    def _Kxx_dxKxx(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.

        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self._squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx

    
    def _optimizers_schedulers(self, lr, lr_noise):
        """Initialize Adam optimizers and schedulers (ReduceLROnPlateau)

        Args:
            lr (float): learning rate for NN parameters `w`
            lr_noise (float): learning rate for noise precision `log_beta`
        """
        optimizers = []
        schedulers = []
        for i in range(self.n_samples):
            parameters = [{'params': [self.Bayesian_model_NN[i].log_beta], 'lr': lr_noise},
                    {'params': self.Bayesian_model_NN[i].features.parameters()}]
            optimizer_i = torch.optim.Adam(parameters, lr=lr)
            optimizers.append(optimizer_i)
            schedulers.append(ReduceLROnPlateau(optimizer_i, 
                    mode='min', factor=0.1, patience=10, verbose=True))
        return optimizers, schedulers