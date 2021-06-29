"""
Stein Variational Gradient Descent for HM-DenseED (UQ).
"""

import torch
import numpy as np
from time import time
from args import args, device
import os
from models.model_det import DenseED
from models.Bayesian_model_NN import Bayesian_model_NN
from models.model_train import Bayesian_model_train
from utils.train_data_upload import train_load_data
from utils.test_data_upload import test_load_data
from utils.misc import mkdirs, logger
from utils.plot import plot_prediction_det
import json
import scipy.io as io
import sys

# deterministic NN


n_out_pixels_train = args.ntrain*128*128
n_out_pixels_test = args.ntest*128*128
dir = './models'

# Bayesian NN
Bayesian_model = torch.load('model_%d.pt'%args.ntrain).to(device)
KLE_val = 16384

# load data
test_loader = test_load_data()
print('Loaded data!')


def test(epoch, logger, test_fixed=None):
    """Evaluate model during training. 
    Print predictions including 4 rows:
        1. target
        2. predictive mean
        3. error of the above two
        4. two sigma of predictive variance

    Args:
        test_fixed (Tensor): (2, N, *), `test_fixed[0]` is the fixed test input, 
            `test_fixed[1]` is the corresponding target
    """
    Bayesian_model.eval()
    mse_test, nlp_test = 0., 0.
    mse_test_final = 0.
    nlp_test_final = 0.
    final_predict = []   
    mse_test, nlp_test = 0., 0.
    final_target_UQ = []
    final_predict_UQ = []
    nlp_test_val = []
    for batch_idx, (input,basis_patch,A_matrix, B_matrix,target_P, q_matrix) in enumerate(test_loader):
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input.float(),basis_patch.float(),A_matrix.float(),B_matrix.float(), target_P.float(), q_matrix.float()
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input_rr.to(device),output_basis.to(device),A1_transformed1.to(device),B1_transformed.to(device), target_pressure.to(device), q1_transformed.to(device)
        #================================================================================ 
        tocc = time()
        output_basis = output_basis.view(args.batchs*144,1,15,15)
        input_rr = input_rr.view(args.batchs*144,1,15,15)
        A_app  = []
        for i in range(args.batchs):
            A_torch = A1_transformed1[i,:,:]
            A_torch1 = A_torch[:,0:2]
            A_torch2 = A_torch[:,2]
            A_torch1 = A_torch1.type(torch.LongTensor).to(device)
            A_torch_final = torch.sparse.FloatTensor(A_torch1.t(), A_torch2, torch.Size([16384,16384]))
            A_app.append(A_torch_final)
        A1_transformed = torch.stack(A_app,dim=0).to(device)
        #================================================================================
        C = io.loadmat(dir+'/matlab_index_save_1.mat')
        C = C['basis_save']
        C = np.squeeze(C)
        X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1]))
        for i in range(X.shape[0]):
            X[i] = C[i]
        # -1 because of matlab and python
        X1 = X.reshape(144,225)-1

        #==============
        #If tanining un-comment below part
        #==============
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
        mse, nlp, output, target = Bayesian_model.test_model(A1_transformed, B1_transformed,q1_transformed,input_rr, target_pressure,batch_idx, X2,
                            size_average=True, out=True)
        y_noise_var = (- Bayesian_model.log_beta).exp().mean()

        mse_test += mse.item()
        nlp_test += nlp.item()
        nlp1 = nlp.cpu().detach().numpy()
        nlp_test_val.append(nlp1)
        final_predict_UQ.append(output)
        final_target_UQ.append(target)
        ticc = time()
        print('total time',ticc-tocc)
    save_pred = np.array(final_predict_UQ)
    save_tar = np.array(final_target_UQ)
    # # print('old................',save_pred.shape)
    # save_pred = np.swapaxes(save_pred,0,1)
    # # print('new................',save_pred.shape)
    mse_test_final += mse_test
    nlp_test_final += nlp_test
    nlp_test_val = np.array(nlp_test_val)
    return mse_test_final, nlp_test_final, save_pred, save_tar


def predict(Bayesian_model, y):
    """
    Predictive mean and variance at x_test. (only average over w and beta)
    Args:
        x_test (Tensor): [N, *], test input
    """
    # S x N x oC x oH x oW
    y = y.to(device)
    y_pred_mean = y.mean(0)
    # compute predictive variance per pixel
    # N x oC x oH x oW
    EyyT = (y ** 2).mean(0)
    EyEyT = y_pred_mean ** 2
    beta_inv = (- Bayesian_model.log_beta).exp()
    y_pred_var = beta_inv.mean() + EyyT - EyEyT

    return y_pred_mean, y_pred_var

#==========================================================
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#==========================================================

def main():
    args.batchs = 64
    print('Start training.........................................................')
    tic = time()
    #result_plot
    mkdir('result_plot')
    #results
    mkdir('results')
    for epoch in range(1):
        print ('epoch number .......................................',epoch)
        with torch.no_grad():
            mse_tot_test, nlp_tot_test, predict_val,target_val  = test(epoch, logger)
            predict_val = np.array(predict_val)
            target_val = np.array(target_val)
            print('MSE:',mse_tot_test)
            RMSE = np.sqrt(mse_tot_test)
            print('RMSE:',RMSE)
    training_time = time() - tic
    print('Finished testing:\n{} epochs\n{} data\n{} samples (SVGD)\n{} seconds'
        .format(args.epochs, args.ntrain, args.n_samples, training_time))

    final_mnlp = np.array(nlp_tot_test)

    print('total_nlp',final_mnlp)

    print('mean_nlp',final_mnlp/4.0)

    #Now
    predict_val1 = predict_val.reshape(256,20,1,128,128)
    predict_val2 = np.swapaxes(predict_val1,0,1)
    predict_val2 = torch.Tensor(predict_val2)
    y_pred_mean, y_pred_var = predict(Bayesian_model, predict_val2)
    y_pred_var = y_pred_var.cpu().detach().numpy()
    y_pred_mean = y_pred_mean.cpu().detach().numpy()
    io.savemat('./results/predict_mean.mat', dict([('predict_mean',np.array(y_pred_mean))]))
    io.savemat('./results/predict_var.mat', dict([('predict_var',np.array(y_pred_var))]))
    target_val1 = target_val.reshape(256,1,128,128)
    target_val2 = target_val1
    io.savemat('./results/target_val.mat', dict([('target_val',np.array(target_val2))]))


    if args.kle == 100:
        img_val = 3
    elif args.kle == 1000:
        img_val = 3
    elif args.kle == 16384:
        img_val = 10
    elif args.data == 'channel':
        img_val = 8
    target1 = target_val2[img_val,0,:,:].reshape(1,128,128)
    mean_predict1 = y_pred_mean[img_val,0,:,:].reshape(1,128,128)
    std_predict1 = y_pred_var[img_val,0,:,:].reshape(1,128,128)
    target1 = np.swapaxes(target1,1,2)
    mean_predict1 = np.swapaxes(mean_predict1,1,2)
    std_predict1 = np.swapaxes(std_predict1,1,2)
    plot_prediction_det(target1, mean_predict1, std_predict1, img_val, KLE_val, plot_fn='imshow')

if __name__ == "__main__":
    main()