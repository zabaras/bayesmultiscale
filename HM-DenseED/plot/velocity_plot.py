#to post-process 
import numpy as np
import sys
# from plot.velocity_src.velocity_single import velocity_post

from plot.velocity_src.velocity_single import velocity_post
from plot.test_filter import GaussianFilter
#======================================================================
import scipy.io as io 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
plt.switch_backend('agg')
import torch

def plot_prediction_det(sample_num, target, predict,velocity_x_pred, velocity_y_pred, velocity_x_tar, velocity_y_tar, epoch, index, 
                        plot_fn='contourf'):
    """Plot prediction for one input (`index`-th at epoch `epoch`)
    Args:
        save_dir: directory to save predictions
        target (np.ndarray): (3, 65, 65)
        prediction (np.ndarray): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """

    
    # 9 x 65 x 65
    axis_val = predict
    samples = np.concatenate((target,velocity_y_tar,velocity_x_tar, predict,velocity_y_pred, velocity_x_pred, target - predict,  velocity_y_pred-velocity_y_tar,  velocity_x_pred-velocity_x_tar), axis=0)
    vmin1 = [np.amin(velocity_y_tar)]
    vmax1 = [np.amax(velocity_y_tar)]

    vmin2 = [np.amin(velocity_x_tar)]
    vmax2 = [np.amax(velocity_x_tar)]

    fig, _ = plt.subplots(3, 3, figsize=(15, 11))
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        if j < 6:
            #print(j)
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap='jet',
                                  vmin=vmin[j % 1], vmax=vmax[j % 1])
            elif plot_fn =='imshow':
                if j == 1:
                    cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                    interpolation='bilinear', vmin=vmin1[j % 1], vmax=vmax1[j % 1])   

                elif j == 4:
                    cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                    interpolation='bilinear', vmin=vmin1[j % 1], vmax=vmax1[j % 1])   

                elif j == 2:
                    cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                    interpolation='bilinear', vmin=vmin2[j % 1], vmax=vmax2[j % 1])   
                elif j == 5:
                    cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                    interpolation='bilinear', vmin=vmin2[j % 1], vmax=vmax2[j % 1])   
                else:
                    cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                    interpolation='bilinear')   

        else:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap='jet')
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                interpolation='bilinear')
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_offset_position('left')
        # cbar.ax.tick_params(labelsize=5)
        cbar.update_ticks()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig('./results/pred_epoch{}_{}.png'.format(epoch, sample_num),
                dpi=100, bbox_inches='tight')
    plt.close(fig)

sigma = 1.8
truncate = 4.0
gaussian_filter = GaussianFilter(sigma=sigma, truncate=truncate, order=0)


def velocity_post_process(target, predict,T,ft,sample_num,epoch):
    #filter the pressure
    pred = predict.reshape(128,128)
    pred_old = predict.reshape(128,128)
    pred_old = pred_old.transpose()
    pred_old = pred_old.reshape(1,128,128)
    v_x_smoothed = gaussian_filter(torch.FloatTensor(pred).unsqueeze(0).unsqueeze(0))
    v_x_smoothed = v_x_smoothed.numpy()[0, 0]

    T = np.array(T)
    ft = np.array(ft)
    ft = ft.reshape(33024,)
    predict = v_x_smoothed.reshape(16384,1)
    velocity = velocity_post(predict,T,ft)
    v1 = velocity[:,1]*1000
    v2 = velocity[:,0]*1000
    v1 = v1.reshape(128,128)
    v2 = v2.reshape(128,128)
    v1 = np.transpose(v1)
    v2 = np.transpose(v2)
    velocity_x_pred = v2
    velocity_y_pred = v1

    target = target.reshape(16384,1)
    velocity = velocity_post(target,T,ft)
    v1 = velocity[:,1]*1000
    v2 = velocity[:,0]*1000
    v1 = v1.reshape(128,128)
    v2 = v2.reshape(128,128)
    v1 = np.transpose(v1)
    v2 = np.transpose(v2)
    velocity_x_tar = v2
    velocity_y_tar = v1
    predict = predict.reshape(1,128,128)
    predict = np.swapaxes(predict,1,2)
    target = target.reshape(1,128,128)
    target = np.swapaxes(target,1,2)


    velocity_x_pred = velocity_x_pred.reshape(1,128,128)
    velocity_y_pred = velocity_y_pred.reshape(1,128,128)
    velocity_x_tar = velocity_x_tar.reshape(1,128,128)
    velocity_y_tar = velocity_y_tar.reshape(1,128,128)


    plot_prediction_det(sample_num, target, pred_old,velocity_x_pred, velocity_y_pred, velocity_x_tar, velocity_y_tar, epoch, 1, 
                            plot_fn='imshow')
    return velocity_x_tar, velocity_y_tar, velocity_x_pred, velocity_y_pred
