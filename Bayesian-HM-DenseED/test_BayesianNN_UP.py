"""
Stein Variational Gradient Descent for Deep ConvNet on GPU.
Current implementation is mainly using for-loops over model instances.
"""

import torch
import numpy as np
from time import time
from args import args, device
import h5py
import os
from models.model_det import DenseED
from models.Bayesian_model_NN import Bayesian_model_NN
from models.model_train import Bayesian_model_train
from utils.misc import mkdirs, logger
from utils.plot1 import plot_prediction_det1
from utils.plot import plot_prediction_det
from utils.mcs_data_upload import mcs_load_data
import json
import scipy.io as io
import sys



n_out_pixels_train = args.ntrain*128*128
n_out_pixels_test = args.ntest*128*128
dir = './models'

# Bayesian NN
Bayesian_model = torch.load('model_%d.pt'%args.ntrain)
KLE_val = 100
# load data
test_loader = mcs_load_data()
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
        output_basis = output_basis.view(144*args.batchs,1,15,15)
        input_rr = input_rr.view(144*args.batchs,1,15,15)
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
    mse_test_final += mse_test
    nlp_test_final += nlp_test
    nlp_test_val = np.array(nlp_test_val)
    return mse_test_final, nlp_test_final, save_pred, save_tar


def UP(Bayesian_model, y):
    y = torch.tensor(y).to(device)
    cond_Ey = y.mean(1)
    cond_Eyy = y.pow(2).mean(1)
    beta_inv = (- Bayesian_model.log_beta).exp()
    print('Noise variances: {}'.format(beta_inv))
    
    y_cond_pred_var = cond_Eyy - cond_Ey ** 2 \
                    + beta_inv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # compute statistics of conditional statistics
    return cond_Ey.mean(0), cond_Ey.var(0), \
            y_cond_pred_var.mean(0), y_cond_pred_var.var(0)

#==========================================================
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#==========================================================

def main():
    args.batchs = 100
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
            nlp_tot_test
            print('MSE:',mse_tot_test)
            RMSE = np.sqrt(mse_tot_test/n_out_pixels_test)
            print('RMSE:',RMSE)
    training_time = time() - tic
    print('Finished testing:\n{} epochs\n{} data\n{} samples (SVGD)\n{} seconds'
        .format(args.epochs, args.ntrain, args.n_samples, training_time))

    final_tar = predict_val
    final_pred = target_val
    final_mnlp = np.array(nlp_tot_test)

    predict_val1 = predict_val.reshape(args.nmc,20,1,128,128)
    predict_val1 = np.swapaxes(predict_val1,0,1)
    predict_val1 = np.swapaxes(predict_val1,3,4)
    target_val1 = target_val.reshape(args.nmc,1,128,128)
    target_val1 = np.swapaxes(target_val1,2,3)


    hf = h5py.File('predict_val1.hdf5', 'w')
    hf.create_dataset('predict', data=predict_val1)
    hf.close()

    hf = h5py.File('target_val1.hdf5', 'w')
    hf.create_dataset('target', data=target_val1)
    hf.close()

    predict_val2 = torch.Tensor(predict_val1)

    y_pred_EE, y_pred_VE, y_pred_EV, y_pred_VV = UP(Bayesian_model, predict_val2)
    y_pred_EE = y_pred_EE.cpu().detach().numpy()
    y_pred_VE = y_pred_VE.cpu().detach().numpy()
    y_pred_EV = y_pred_EV.cpu().detach().numpy()
    y_pred_VV = y_pred_VV.cpu().detach().numpy()


    two_sigma = 2 * np.sqrt(y_pred_VV)
    two_sigma_VE = 2 * np.sqrt(y_pred_VE)

    actual_mean = np.mean(target_val1,axis=0)
    actual_mean = actual_mean.reshape(128,128)
    actual_var = np.var(target_val1,axis=0)
    actual_var = actual_var.reshape(128,128)

    target1 = actual_mean.reshape(1,128,128)
    mean_predict1 = y_pred_EE.reshape(1,128,128)
    std_predict1 = two_sigma_VE.reshape(1,128,128)

    
    plot_prediction_det(target1, mean_predict1, std_predict1, args.ntrain, KLE_val, plot_fn='imshow')

    target1v = actual_var.reshape(1,128,128)
    mean_predict1v = y_pred_EV.reshape(1,128,128)
    std_predict1v = two_sigma.reshape(1,128,128)

    
    plot_prediction_det1(target1v, mean_predict1v, std_predict1v, args.ntrain, KLE_val, plot_fn='imshow')

    #Save files
    io.savemat('./result_plot/target1.mat', dict([('target1',np.array(target1))]))
    io.savemat('./result_plot/mean_predict1.mat', dict([('mean_predict1',np.array(mean_predict1))]))
    io.savemat('./result_plot/std_predict1.mat', dict([('std_predict1',np.array(std_predict1))]))
    io.savemat('./result_plot/target1v.mat', dict([('target1v',np.array(target1v))]))
    io.savemat('./result_plot/mean_predict1v.mat', dict([('mean_predict1v',np.array(mean_predict1v))]))
    io.savemat('./result_plot/std_predict1v.mat', dict([('std_predict1v',np.array(std_predict1v))]))




if __name__ == "__main__":
    main()
