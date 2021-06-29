import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from args import args, device

dir = './plot/'

dir_1 = './result_data/'

dir_2 = './result_data/'

dir_3 = './result_data/'

ntrain = 64
# img_val = 7

if args.kle == 100:
    img_val = 28
    val_p = 96
elif args.kle == 1000:
    img_val = 23
    val_p = 82
elif args.kle == 16384:
    img_val = 28
    val_p = 126
elif args.data == 'channel':
    img_val = 7
    val_p = 79

C = io.loadmat(dir+'basis_save_1.mat')
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

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def plot_basis(tar1,pred1,ntrain,val_p):
    plt.figure(figsize=(15,15))
    plt.subplot(331)
    im1 = plt.imshow(tar1.reshape(15,15), cmap='jet', origin='lower',
                                        interpolation='bilinear')
    plt.title("Ground truth")
    plt.axis('off')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im1, cax=cax) 
    plt.clim(0,1)
    plt.subplot(332)
    im2 = plt.imshow(pred1.reshape(15,15), cmap='jet', origin='lower',
                                        interpolation='bilinear')
    plt.title("Predicted")
    plt.axis('off')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax) 
    plt.clim(0,1)
    plt.subplot(333)
    im = plt.imshow((tar1.reshape(15,15)-pred1.reshape(15,15)), cmap='jet', origin='lower',
                                        interpolation='bilinear')
    plt.title("Error: actual-predicted")
    plt.axis('off')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax) 
    plt.savefig('./results/Basis_plot_KLE_100_%d_%d.png'%(ntrain,val_p),bbox_inches='tight')
#======================================================================
def basis_plot():
    pro1 = io.loadmat(dir_1+'test_prolongation_%d.mat'%args.epochs)
    pro1 = pro1['prolongation_operator']
    B1_p_out1 = pro1[img_val,:,:]
    B1_p_out1 = B1_p_out1.reshape(16384, 256)
    jjj=0
    ss = 0
    temp11 = []
    for ii in range(12):
        jjj = 35+16*ii
        for k in range (jjj,jjj+12):
            ss =ss+1
            s = ss-1
            basis_temp1 = X2[s,:]
            tem1 = B1_p_out1[:,k-1]
            inds1 = indices(tem1, lambda x: x > 0)
            AA1 = tem1[inds1]
            temp11.append(AA1)
    temp11 = np.array(temp11)
    # print(temp1.shape)
    tar1 = io.loadmat(dir_1+'test_interior_basis_%d.mat'%args.epochs)
    tar1 = tar1['interior_basis']
    #========================================================================
    #========================================================================
    #========================================================================
    tar15 = tar1
    # val_p = 79
    pred1 = temp11[143-val_p,:].reshape(15,15)
    pred1 = pred1.transpose()
    tar1 = tar15[img_val,val_p,:].reshape(15,15)
    tar1 = tar1.transpose()
    plot_basis(tar1, pred1,ntrain,val_p)

