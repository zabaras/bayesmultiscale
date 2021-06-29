"""
Stein Variational Gradient Descent for HM-DenseED.
"""

import torch
import numpy as np
from time import time
from args import args, device
from models.model_det import DenseED
from models.Bayesian_model_NN import Bayesian_model_NN
from models.model_train import Bayesian_model_train
from utils.train_data_upload import train_load_data
import json
import scipy.io as io
import sys
import os

# deterministic NN
dense_ed = DenseED(in_channels=1, 
                    out_channels=1, 
                    blocks=args.blocks,
                    growth_rate=args.growth_rate, 
                    init_features=args.init_features,
                    drop_rate=args.drop_rate,
                    bn_size=args.bn_size,
                    bottleneck=args.bottleneck,
                    out_activation='Sigmoid')

n_out_pixels_train = args.ntrain*128*128

# Bayesian NN
Bayesian_model = Bayesian_model_NN(dense_ed, n_samples=args.n_samples).to(device)
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path) 


# load data
train_loader = train_load_data()
print('Loaded data!')

# Initialize SVGD
Bayesian_train = Bayesian_model_train(Bayesian_model, train_loader)

def main():
    print('Start training.........................................................')
    tic = time()
    train_loss = []
    for epoch in range(1, args.epochs + 1):
        print ('epoch number .......................................',epoch)
        ticc = time()
        mse_tot2 = Bayesian_train.train(epoch)
        print('Training RMSE 2:.......',mse_tot2)
        train_loss.append(mse_tot2)
        tocc = time()
        print('.............training done....................', tocc-ticc)   
    train_loss = np.array(train_loss)
    io.savemat('train_loss.mat', dict([('train_loss',np.array(train_loss))]))
    training_time = time() - tic
    print('Finished training:\n{} epochs\n{} data\n{} samples (SVGD)\n{} seconds'
        .format(args.epochs, args.ntrain, args.n_samples, training_time))

    torch.save(Bayesian_model, 'model_%d.pt'%args.ntrain)


if __name__ == '__main__':
    main()