"""
Train deterministic convolutional encoder-decoder networks
"""
from time import time
import torch
import os
import numpy as np
import scipy.io as io
import sys
from args import args, device
from models.model import DenseED
from utils.mcs_data_upload import mcs_load_data
from models.bdsmm import bdsmm
from plot.velocity_pdf import velocity_post_pdf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')
import seaborn as sns
torch.set_default_tensor_type('torch.FloatTensor')
dir = './plot'
# load data
mcs_loader = mcs_load_data()
print('Loaded data!')

def mcs_test(model):
    model.eval()
    final_target_pressure = []
    final_predict_pressure = []
    final_target_velocityx = []
    final_predict_velocityx = []
    final_target_velocityy = []
    final_predict_velocityy = []
    for batch_idx, (input,basis_patch,A_matrix, B_matrix,target_P, q_matrix, T_val, ft_val) in enumerate(mcs_loader):
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input.float(),basis_patch.float(),A_matrix.float(),B_matrix.float(), target_P.float(), q_matrix.float()
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input_rr.to(device),output_basis.to(device),A1_transformed1.to(device),B1_transformed.to(device), target_pressure.to(device), q1_transformed.to(device)
        #================================================================================
        output_basis = output_basis.view(50*144,1,15,15)
        input_rr = input_rr.view(50*144,1,15,15)
        A_app  = []
        for i in range(50):
            A_torch = A1_transformed1[i,:,:]
            A_torch1 = A_torch[:,0:2]
            A_torch2 = A_torch[:,2]
            A_torch1 = A_torch1.type(torch.LongTensor).to(device)
            A_torch_final = torch.sparse.FloatTensor(A_torch1.t(), A_torch2, torch.Size([16384,16384]))
            A_app.append(A_torch_final)
        A1_transformed = torch.stack(A_app,dim=0).to(device)
        C = io.loadmat(dir+'/basis_save_1.mat')
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
        #================================================================================
        with torch.no_grad():
            output = model(input_rr)
        #================================================================================
        output = output.view(50,144,225)
        output_basis = output_basis.view(50,144,225)
        #============================================================================
        for RRRR in range (50):
            output_RR = output[RRRR,:,:].reshape(144,225)    
            B1_p_out = B1_transformed[RRRR,:,:].reshape(16384, 256)  
            jjj=0
            ss = 0
            for ii in range(12):
                jjj = 35+16*ii
                for k in range (jjj,jjj+12):
                    ss =ss+1
                    s = ss-1
                    basis_temp = X2[s,:]
                    tem = B1_p_out[:,k-1]
                    basis_temp = basis_temp.type(torch.LongTensor)
                    tem = tem.type(torch.cuda.FloatTensor)
                    temp_variable = output_RR[143-s,:]/torch.max(output_RR[143-s,:])
                    tem[basis_temp] = temp_variable
                    B1_p_out[:,k-1] = tem       
            B1_transformed[RRRR,:,:] = B1_p_out        
        #====================================================
        temp11 = []
        for kkk in range(50):
            B1_transformed_temp = B1_transformed[kkk,:,:]
            B1_transformed1 = torch.transpose(B1_transformed_temp,0,1)
            dim = torch.sum(B1_transformed1,dim=0)
            B1_transformed2 = torch.div(B1_transformed1,dim)
            B1_transformed22 = torch.transpose(B1_transformed2,0,1)
            temp11.append(B1_transformed22)
        B1_transformed = torch.stack(temp11,dim=0).to(device) 
        #============================================  
        R1_transformed = torch.transpose(B1_transformed,1,2) 
        A1_transformed = torch.transpose(A1_transformed,1,2)
        R1_transformed = torch.transpose(R1_transformed,1,2)
        A_c_transformed = torch.matmul(torch.transpose(bdsmm(A1_transformed,R1_transformed),1,2),B1_transformed)
        R1_transformed = torch.transpose(R1_transformed,1,2) 
        temp1_transformed = torch.matmul(R1_transformed,q1_transformed)
        temp2_transformed,LU = torch.solve(temp1_transformed,A_c_transformed)
        temp3_transformed = torch.matmul(B1_transformed,temp2_transformed)
        predict_pressure = temp3_transformed.view(50,16384)
        target_pressure = target_pressure.view(50,16384)
        target_pressure = target_pressure.type(torch.cuda.FloatTensor).view(50,16384)
        predict_pressure = predict_pressure.type(torch.cuda.FloatTensor).view(50,16384)  
        predict_press = predict_pressure
        predict_press = predict_press.cpu().detach().numpy() 
        target_press = target_pressure
        target_press = target_press.cpu().detach().numpy()            
        interior_basis = output_basis.cpu().detach().numpy()   
        vx_tar = []
        vy_tar = []
        vx_pred= []
        vy_pred= []
        for kkk in range(50):
            vx,vy, vxpred, vypred = velocity_post_pdf(target_press[kkk,:], predict_press[kkk,:],T_val[kkk,:],ft_val[kkk,:],batch_idx,kkk)
            vx_tar.append(vx)
            vy_tar.append(vy)
            vx_pred.append(vxpred)
            vy_pred.append(vypred)
        vx_tar = np.array(vx_tar)
        vy_tar = np.array(vy_tar)
        vx_pred= np.array(vx_pred)
        vy_pred= np.array(vy_pred)

        final_target_pressure.append(target_press)
        final_predict_pressure.append(predict_press)

        final_target_velocityx.append(vx_tar)
        final_predict_velocityx.append(vx_pred)

        final_target_velocityy.append(vy_tar)
        final_predict_velocityy.append(vy_pred)
    
    final_target_pressure = np.array(final_target_pressure).reshape(args.nmcs,16384)

    final_predict_pressure = np.array(final_predict_pressure).reshape(args.nmcs,16384)

    final_target_velocityx = np.array(final_target_velocityx).reshape(args.nmcs,16384)
    
    final_predict_velocityx = np.array(final_predict_velocityx).reshape(args.nmcs,16384)

    final_target_velocityy = np.array(final_target_velocityy).reshape(args.nmcs,16384)

    final_predict_velocityy = np.array(final_predict_velocityy).reshape(args.nmcs,16384)

    plt.figure()
    marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='tab:red')
    sns.kdeplot(final_target_pressure[:, 7529], color='b', ls='--', label='Monte Carlo')
    sns.kdeplot(final_predict_pressure[:, 7529], color='none', marker='o', markersize='2', markerfacecolor='none',markeredgewidth='5', markeredgecolor='green', label='Hybrid DenseED-MsRSB')
    plt.legend()
    plt.savefig('./results/KLE_{}_pressure_{}.png'.format(args.kle,args.ntrain),dpi=300, bbox_inches='tight')
    plt.close()



    plt.figure()
    marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='tab:red')
    sns.kdeplot(final_target_velocityx[:, 7529], color='b', ls='--', label='Monte Carlo')
    sns.kdeplot(final_predict_velocityx[:, 7529], color='none', marker='o', markersize='2', markerfacecolor='none',markeredgewidth='5', markeredgecolor='green', label='Hybrid DenseED-MsRSB')
    plt.legend()
    plt.savefig('./results/KLE_{}_velocity_x_{}.png'.format(args.kle,args.ntrain),dpi=300, bbox_inches='tight')
    plt.close()


    plt.figure()
    marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='tab:red')
    sns.kdeplot(final_target_velocityy[:, 7529], color='b', ls='--', label='Monte Carlo')
    sns.kdeplot(final_predict_velocityy[:, 7529], color='none', marker='o', markersize='2', markerfacecolor='none',markeredgewidth='5', markeredgecolor='green', label='Hybrid DenseED-MsRSB')
    plt.legend()
    plt.savefig('./results/KLE_{}_velocity_y_{}.png'.format(args.kle,args.ntrain),dpi=300, bbox_inches='tight')
    plt.close()






