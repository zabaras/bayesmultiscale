#to post-process PDF
import numpy as np
import sys
from plot.velocity_src.velocity_single import velocity_post
from plot.test_filter import GaussianFilter
#======================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
plt.switch_backend('agg')
import torch

sigma = 1.8
truncate = 4.0
gaussian_filter = GaussianFilter(sigma=sigma, truncate=truncate, order=0)

def velocity_post_pdf(target, predict,T,ft,sample_num,epoch):
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
    return velocity_x_tar, velocity_y_tar, velocity_x_pred, velocity_y_pred
