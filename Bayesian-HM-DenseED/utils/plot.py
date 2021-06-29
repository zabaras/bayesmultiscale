import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
plt.switch_backend('agg')

import scipy.io as io

def plot_prediction_det(target, mean_predict, std_predict, epoch, index, 
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
    samples = np.concatenate((target,mean_predict,target-mean_predict, std_predict), axis=0)

    # vmin1 = [np.amin(velocity_y_tar)]
    # vmax1 = [np.amax(velocity_y_tar)]
    fig, _ = plt.subplots(2, 2, figsize=(12, 8))
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        if j < 6:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap='jet')
            elif plot_fn =='imshow':
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
    plt.savefig('./result_plot/Mean_KLE_{}_sample_{}.png'.format(index,epoch),dpi=300, bbox_inches='tight')
    plt.close(fig)


# if __name__ == "__main__":
#     #load the matlab files
#     # target = io.loadmat('target_0.mat')

#     # predict = io.loadmat('predict_0.mat')

#     # predict = predict['predict_pressure1_0']

#     # target = target['target_pressure1_0']

#     # print(predict.shape)
#     # print(target.shape)
#     # target = np.array(target)
#     # predict = np.array(predict)
#     # mean_predict = np.mean(predict,axis=0)
#     # std_predict = np.std(predict,axis=0)
#     target = io.loadmat('target_val')
#     target = target['target_val']
#     mean_predict = io.loadmat('predict_mean.mat')
#     std_predict  = io.loadmat('predict_var')

#     mean_predict = mean_predict['predict_mean']
#     std_predict = std_predict['predict_var']



#     for ii in range(16):
#         target1 = target[ii,0,:,:].reshape(1,128,128)
#         mean_predict1 = mean_predict[ii,0,:,:].reshape(1,128,128)
#         std_predict1 = std_predict[ii,0,:,:].reshape(1,128,128)

#         target1 = np.swapaxes(target1,1,2)
#         mean_predict1 = np.swapaxes(mean_predict1,1,2)
#         std_predict1 = np.swapaxes(std_predict1,1,2)
#         plot_prediction_det(target1, mean_predict1, std_predict1, ii, 100, plot_fn='imshow')