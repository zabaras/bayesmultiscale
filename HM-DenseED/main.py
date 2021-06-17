"""
Train deterministic HM-DenseED
"""
from time import time
import torch
import os
import numpy as np
import scipy.io as io
import sys
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from args import args, device
from models.model import DenseED
from utils.train_data_upload import train_load_data
from utils.test_data_upload import test_load_data
from models.bdsmm import bdsmm
from plot.velocity_plot import velocity_post_process
from plot.plot_PDF import mcs_test
from plot.basis_plot import basis_plot
import matplotlib.pyplot as plt
import scipy.io as io
import matplotlib.ticker as ticker
plt.switch_backend('agg')

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# initialize DenseED model
model = DenseED(in_channels=1, out_channels=1, 
                blocks=args.blocks,
                growth_rate=args.growth_rate, 
                init_features=args.init_features,
                drop_rate=args.drop_rate,
                bn_size=args.bn_size,
                bottleneck=args.bottleneck,
                out_activation='Sigmoid').to(device)
print(model)

n_out_pixels_train = args.ntrain * args.imsize
n_out_pixels_test = args.ntest * args.imsize

dir = './plot'

# load data
train_loader = train_load_data()
test_loader = test_load_data()

print('...................Loaded data!...........................')

optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.01, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)

#train
def train(epoch):
    model.train()
    mse = 0.
    mse_total = 0.
    #===================================================================================
    for batch_idx, (input,basis_patch,A_matrix, B_matrix,target_P, q_matrix) in enumerate(train_loader):
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input.float(),basis_patch.float(),A_matrix.float(),B_matrix.float(), target_P.float(), q_matrix.float()
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input_rr.to(device),output_basis.to(device),A1_transformed1.to(device),B1_transformed.to(device), target_pressure.to(device), q1_transformed.to(device)
        #================================================================================
        output_basis = output_basis.view(144,1,15,15)
        input_rr = input_rr.view(144,1,15,15)
        #load the A matrix from sparse to dense
        A_app  = []
        for i in range(1):
            A_torch = A1_transformed1[i,:,:]
            A_torch1 = A_torch[:,0:2]
            A_torch2 = A_torch[:,2]
            A_torch1 = A_torch1.type(torch.LongTensor).to(device)
            A_torch_final = torch.sparse.FloatTensor(A_torch1.t(), A_torch2, torch.Size([16384,16384]))
            A_app.append(A_torch_final)
        A1_transformed = torch.stack(A_app,dim=0)
        #Obtain the indices for the non-interior support region
        C = io.loadmat(dir+'/basis_save_1.mat')
        C = C['basis_save']
        C = np.squeeze(C)
        X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1]))
        for i in range(X.shape[0]):
            X[i] = C[i]
        # -1 because of matlab and python
        X1 = X.reshape(144,225)-1
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
        #================================================================================
        model.zero_grad()
        #obtain the output from the DenseED model
        output = model(input_rr)
        #================================================================================
        output = output.view(1,144,225)
        #============================================================================
        for RRRR in range (1):
            output_RR = output[RRRR,:,:]    
            output_RR = output_RR.reshape(144,225)
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
        R1_transformed = torch.transpose(B1_transformed,1,2) 
        A1_transformed = torch.transpose(A1_transformed,1,2)
        R1_transformed = torch.transpose(R1_transformed,1,2)
        A_c_transformed = torch.matmul(torch.transpose(bdsmm(A1_transformed,R1_transformed),1,2),B1_transformed)
        R1_transformed = torch.transpose(R1_transformed,1,2) 
        temp1_transformed = torch.matmul(R1_transformed,q1_transformed)
        temp2_transformed,LU = torch.gesv(temp1_transformed,A_c_transformed)
        temp3_transformed = torch.matmul(B1_transformed,temp2_transformed)
        predict_pressure = temp3_transformed
        target_pressure = target_pressure.view(1,16384)
        predict_pressure = predict_pressure.view(1,16384)    
        target_pressure = target_pressure.type(torch.cuda.FloatTensor)
        predict_pressure = predict_pressure.type(torch.cuda.FloatTensor)           
        loss2 = F.mse_loss(predict_pressure,target_pressure, size_average=False)
        loss2.backward()
        optimizer.step()
        mse += loss2.item()        
    rmse = np.sqrt(mse / n_out_pixels_train)
    scheduler.step(rmse)          
    mse_total += mse 
    return mse_total

#test
def test(epoch):
    model.eval()
    mse = 0.
    mse_total = 0.
    final_target = []
    final_predict = []
    for batch_idx, (input,basis_patch,A_matrix, B_matrix,target_P, q_matrix, T_val, ft_val) in enumerate(test_loader):
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input.float(),basis_patch.float(),A_matrix.float(),B_matrix.float(), target_P.float(), q_matrix.float()
        input_rr,output_basis,A1_transformed1,B1_transformed, target_pressure, q1_transformed \
            = input_rr.to(device),output_basis.to(device),A1_transformed1.to(device),B1_transformed.to(device), target_pressure.to(device), q1_transformed.to(device)
        #================================================================================
        output_basis = output_basis.view(144*64,1,15,15)
        input_rr = input_rr.view(144*64,1,15,15)
        A_app  = []
        for i in range(64):
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
        #================================================================================
        with torch.no_grad():
            output = model(input_rr)
        #================================================================================
        output = output.view(64,144,225)
        output_basis = output_basis.view(64,144,225)
        #============================================================================
        for RRRR in range (64):
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
        for kkk in range(64):
            B1_transformed_temp = B1_transformed[kkk,:,:]
            B1_transformed1 = torch.transpose(B1_transformed_temp,0,1)
            dim = torch.sum(B1_transformed1,dim=0)
            B1_transformed2 = torch.div(B1_transformed1,dim)
            B1_transformed22 = torch.transpose(B1_transformed2,0,1)
            temp11.append(B1_transformed22)
        B1_transformed = temp11
        B1_transformed = torch.stack(temp11,dim=0).to(device)
        temp_save = B1_transformed.cpu().detach().numpy()  
        #============================================  
        R1_transformed = torch.transpose(B1_transformed,1,2)
        A1_transformed = torch.transpose(A1_transformed,1,2)
        R1_transformed = torch.transpose(R1_transformed,1,2)
        A_c_transformed = torch.matmul(torch.transpose(bdsmm(A1_transformed,R1_transformed),1,2),B1_transformed)
        R1_transformed = torch.transpose(R1_transformed,1,2) 
        temp1_transformed = torch.matmul(R1_transformed,q1_transformed)
        temp2_transformed,LU = torch.gesv(temp1_transformed,A_c_transformed)
        temp3_transformed = torch.matmul(B1_transformed,temp2_transformed)
        predict_pressure = temp3_transformed.view(64,16384)
        target_pressure = target_pressure.view(64,16384).type(torch.cuda.FloatTensor)
        predict_pressure = predict_pressure.type(torch.cuda.FloatTensor)
        loss2 = F.mse_loss(predict_pressure,target_pressure, size_average=False)     
        predict_press = predict_pressure.cpu().detach().numpy() 
        target_press = target_pressure
        target_press = target_press.cpu().detach().numpy()            
        if epoch % args.epochs == 0:
            if batch_idx == 0:
                interior_basis = output_basis.cpu().detach().numpy()
                io.savemat('./result_data/test_interior_basis_%d.mat'%epoch, dict([('interior_basis',np.array(interior_basis))]))
                io.savemat('./result_data/test_prolongation_%d.mat'%epoch, dict([('prolongation_operator',np.array(temp_save))]))              
                kkk = 28
                velocity_x_tar, velocity_y_tar, velocity_x_pred, velocity_y_pred = \
                         velocity_post_process(target_press[kkk,:], predict_press[kkk,:],T_val[kkk,:],ft_val[kkk,:],epoch,kkk)
        mse += loss2.item()
        final_target.append(target_press)
        final_predict.append(predict_press)
    final_target = np.array(final_target)
    final_predict = np.array(final_predict)
    output_new = output.view(64,144,15,15)
    predict_new = output_basis.view(64,144,15,15)
    output_new = output_new.cpu().detach().numpy() 
    predict_new = predict_new.cpu().detach().numpy()       
    mse_total += mse 
    return mse_total, final_target, final_predict

#plot rmse
def train_test_error(r2_train,r2_test,epoch):
    plt.figure()
    plt.plot(r2_train, label="Train: {:.3f}".format(np.mean(r2_train[-5:])))
    plt.plot(r2_test, label="Test: {:.3f}".format(np.mean(r2_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel(r'RMSE')
    plt.legend(loc='lower right')
    plt.savefig("./results/rmse.pdf", dpi=600)
    plt.close()
    np.savetxt("./results/rmse_train.txt", r2_train)
    np.savetxt("./results/rmse_test.txt", r2_test)
#==========================================================
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#==========================================================
#main
def main():
    mkdir('results')
    mkdir('result_data')
    print('Start training and testing........................................................')
    loss_train_all = []
    loss_test_all = []
    for epoch in range(1, args.epochs + 1):
        print ('epoch number .......................................',epoch)
        mse_tot = train(epoch)
        with torch.no_grad():
            mse_tot_test, final_target, final_predict = test(epoch) 
        rmse = np.sqrt(mse_tot / n_out_pixels_train)
        rmse1 = np.sqrt(mse_tot_test / n_out_pixels_test)
        loss_train_all.append(rmse)
        loss_test_all.append(rmse1)
    loss_train_all = np.array(loss_train_all)
    loss_test_all = np.array(loss_test_all)
    print('.............training and testing done....................')
    print('\n')
    print('saving the training error and testing error')
    io.savemat('./result_data/training_loss.mat', dict([('training_loss',np.array(loss_train_all))]))
    io.savemat('./result_data/test_loss.mat', dict([('testing_loss',np.array(loss_test_all))]))
    print('.........................plotting results.........................................')
    train_test_error(loss_train_all,loss_test_all, 1)
    print('.........................saving model..............................................')
    torch.save(model, 'KLE_100.pt')
    print('.............................plotting basis.........................................')
    basis_plot()
    """clean up gpu memory"""
    torch.cuda.empty_cache()
    print('.............................plotting PDF............................................')
    mcs_test(model)

if __name__ == '__main__':
    main()


