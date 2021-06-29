"""
Particle approximations for posterior of Bayesian neural net used in SVGD.

References:
   1) https://github.com/zabaras/cnn-surrogate/tree/master/models
   2) Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
    A general purpose bayesian inference algorithm." NIPS. 2016.

methods:
    __init__
    forward
Note: 
`torch.distributions` is not much used in this implementation to keep simple.
Also we trade computing for memory by using for-loop rather than in a batch way.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Gamma
from utils.misc import parameters_to_vector, vector_to_parameters, log_sum_exp
import scipy.io as io
import math
import numpy
import copy
import sys
from time import time
import numpy as np
from models.bdsmm import bdsmm
from args import args, device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#for uq: args.batchs = 64
#for up: args.batchs = 100
class Bayesian_model_NN(nn.Module):
    """Class for Bayesian NNs with Stein Variational Gradient Descent.
    Not for usage independently.
    
    Bayesian NNs: y = f(x, w) + n

    uncertain weights:
            w_i ~ Normal(w_i | mu=0, 1 / alpha) 
            alpha ~ Gamma(alpha | shape=1, rate=0.05) (shared)
            --> w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
            Parameterization of StudentT in Bishop p.103 Eq. (2.159)

    Assumptions on noise:
        Additive, Gaussian, homoscedastic (independent of input), 
        output wise (same for every pixels in the output).
            n ~ Normal(0, 1 / beta)
            beta ~ Gamma(beta | shape=2, rate=2e-6)

    Hyperparameters for weights and noise are pre-defined based on heuristic.

    Given a deterministic `model`, initialize `n_samples` replicates
    of the `model`. (plus `n_samples` of noise precision realizations)

    `model` must implement `reset_parameters` method for the replicates
    to have different initial realizations of uncertain parameters.

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The deterministic NN to be instantiated `n_samples` 
            times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """
    def __init__(self, model, n_samples=20):
        super(Bayesian_model_NN, self).__init__()
        self.n_samples = n_samples
        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # for efficiency, represent StudentT params using Gamma params
        self.w_prior_shape = 1.
        self.w_prior_rate = 0.03 
        # noise variance 1e-6: beta ~ Gamma(beta | shape, rate)
        self.beta_prior_shape = 3.
        self.beta_prior_rate = 1.e-6
        # replicate `n_samples` instances with the same network as `model`
        instances = []
        for i in range(n_samples):
            new_instance = copy.deepcopy(model)
            # initialize each model instance with their defualt initialization
            # instead of the prior
            new_instance.reset_parameters()
            print('Reset parameters in model instance {}'.format(i))
            instances.append(new_instance)
        self.nnets = nn.ModuleList(instances)
        del instances
        # log precision (Gamma) of Gaussian noise
        log_beta = Gamma(self.beta_prior_shape, 
                         self.beta_prior_rate).sample((self.n_samples,)).log()
        for i in range(n_samples):
            self.nnets[i].log_beta = Parameter(log_beta[i])


    def __getitem__(self, idx):
        return self.nnets[idx]

    @property
    def log_beta(self):
        return torch.tensor([self.nnets[i].log_beta.item() 
            for i in range(self.n_samples)], device=device)

    def forward(self, input):
        """Computes all the `n_samples` NN output
        Args:
            input: N x iC x iH x iW

        Return:
            output: S x N x oC x oH x oW
        """
        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(input))
        output = torch.stack(output)

        return output

    def _log_joint(self, index, output, target, ntrain):
        """Log joint probability or unnormalized posterior for single model
        instance. Ignoring constant terms for efficiency.
        Can be implemented in batch computation, but memory is the bottleneck.
        Thus here we trade computation for memory, e.g. using for loop.

        Args:
            index (int): model index, 0, 1, ..., `n_samples`
            output (Tensor): B x oC x oH x oW
            target (Tensor): B x oC x oH x oW
            ntrain (int): total number of training data, mini-batch is used to
                evaluate the log joint prob

        Returns:
            Log joint probability (zero-dim tensor)
        """
        # Normal(target | output, 1 / beta * I)
        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * (target - output).pow(2).sum()
                            + 0.5 * target.numel() * self.nnets[index].log_beta)
        # log prob of prior of weights, i.e. log prob of studentT
        log_prob_prior_w = torch.tensor(0.).to(device)
        for param in self.nnets[index].features.parameters():
            log_prob_prior_w += \
                torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        # log prob of prior of log noise-precision (NOT noise precision)
        log_prob_prior_log_beta = ((self.beta_prior_shape-1.0) * self.nnets[index].log_beta \
                    - self.nnets[index].log_beta.exp() * self.beta_prior_rate)
        return log_likelihood + log_prob_prior_w + log_prob_prior_log_beta


    def test_model(self,A1_transformed, B1_transformed,q1_transformed,input, target,e, X2,size_average=True, out=False):
        """Evaluate the MSE and Negative Log Probability.

        Args:
            input (Tensor): (N, iC, iH, iW)
            target (Tensor): (N, oC, oH, oW)
            size_average (bool)
            out (bool): If True, return output of `bayes_nn` w. `input`

        Returns:
            (mse, nlp) if `out` is False, o.w. (mse, nlp, output)
            where output is of size (S, N, oC, oH, oW)
        """
        # S x N x oC x oH x oW
        output12 = self.forward(input)
        print('20 X batch size X 144 X 225',output12.shape)
        output2 = output12.view(20,args.batchs,144,225)
        final_predict = []
        for vv in range(20):
            output1 = output2[vv,:,:,:]
            output1 = output1.view(args.batchs,144,225) 
            for RRRR in range (args.batchs):
                output_RR = output1[RRRR,:,:]    
                output_RR = output_RR.reshape(144,225)
                B1_p_out = B1_transformed[RRRR,:,:]
                B1_p_out = B1_p_out.reshape(16384, 256)
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
            for kkk in range(args.batchs):
                B1_transformed_temp = B1_transformed[kkk,:,:]
                B1_transformed1 = torch.transpose(B1_transformed_temp,0,1)
                dim = torch.sum(B1_transformed1,dim=0)
                B1_transformed2 = torch.div(B1_transformed1,dim)
                B1_transformed22 = torch.transpose(B1_transformed2,0,1)
                temp11.append(B1_transformed22)
            B1_transformed = temp11
            B1_transformed = torch.stack(temp11,dim=0).to(device)
            #============================================  
            R1_transformed = torch.transpose(B1_transformed,1,2) #check here
            A1_transformed = torch.transpose(A1_transformed,1,2)
            R1_transformed = torch.transpose(R1_transformed,1,2)
            A_c_transformed = torch.matmul(torch.transpose(bdsmm(A1_transformed,R1_transformed),1,2),B1_transformed)
            R1_transformed = torch.transpose(R1_transformed,1,2) #transform back to the old R1_transformed
            temp1_transformed = torch.matmul(R1_transformed,q1_transformed)
            temp2_transformed,LU = torch.solve(temp1_transformed,A_c_transformed)
            temp3_transformed = torch.matmul(B1_transformed,temp2_transformed)
            predict_pressure = temp3_transformed.view(args.batchs,16384)
            predict_pressure = predict_pressure.type(torch.cuda.FloatTensor)
            final_predict.append(predict_pressure)
        output = torch.stack(final_predict)
        output = output.view(20,args.batchs,1,128,128)
        output = output.type(torch.cuda.FloatTensor)
        target_pressure = target.type(torch.cuda.FloatTensor)
        target = target_pressure.view(args.batchs,1,128,128)
        # S x 1 x 1 x 1 x 1
        log_beta = self.log_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        log_2pi_S = torch.tensor(0.5 * target[0].numel() * math.log(2 * math.pi)
                       + math.log(self.n_samples), device=device)
        # S x N
        exponent = - 0.5 * (log_beta.exp() * ((target - output) ** 2)).view(
            self.n_samples, target.size(0), -1).sum(-1) \
                   + 0.5 * target[0].numel() * self.log_beta.unsqueeze(-1)

        nlp = - log_sum_exp(exponent, dim=0).mean() + log_2pi_S
        mse = ((target - output.mean(0)) ** 2).mean()
        if not size_average:
            mse *= target.numel()
            nlp *= target.size(0)
        if not out:
            return mse, nlp
        else:
            predict_val = output.cpu().detach().numpy()
            target_val = target.cpu().detach().numpy()
            predict_val = np.swapaxes(predict_val,0,1)
            return mse, nlp, predict_val, target_val
