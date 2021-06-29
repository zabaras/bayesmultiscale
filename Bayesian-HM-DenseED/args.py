import argparse
import torch
import json
import random
from pprint import pprint
import time

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Bayesian Convolutional Encoder-Decoder Networks with SVGD')
        self.add_argument('--post', action='store_true', default=False, help='enables post analysis mode, default is training mode')
        self.add_argument('--debug', action='store_true', default=False, help='enters debug mode, to check GPU memory usage, try different hyperparams, etc.')
        self.add_argument('--exp-name', type=str, default='Bayesian', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')       

        # model
        self.add_argument('-ns', '--n-samples', type=int, default=20, help='(20-30) number of model instances in SVGD')        
        self.add_argument('--blocks', type=list, default=[4, 8, 4], help='list of number of layers in each dense block')
        self.add_argument('--growth-rate', type=int, default=16, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=48, help='number of initial features after the first conv layer')        
        self.add_argument('--drop-rate', type=float, default=0., help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enables bottleneck design in the dense blocks')

        # data
        self.add_argument('--data', type=str, default="KLE", help='dataset KLE or channel')
        self.add_argument('--data-dir', type=str, default="./dataset", help='directory to dataset')
        self.add_argument('--kle', type=int, default=100, help='num of KLE terms. Make this 0 when data is "channel"')
        self.add_argument('--ntrain', type=int, default=64, help="number of training data")
        self.add_argument('--batchs', type=int, default=100, help="batch size for UQ is 64 and UP is 100")
        self.add_argument('--ntest', type=int, default=256, help="number of test data") 
        self.add_argument('--nmc', type=int, default=10000, help="number of MC test data for uncertainty propagation")
        ## more details on dataset
        self.add_argument('--nic', type=int, default=1, help="number of input channels")
        self.add_argument('--noc', type=int, default=1, help="number of output channels")
        self.add_argument('--image-size', type=int, default=[15, 15], help="image resolution")
        
        # training
        self.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=2e-3, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=0.01, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=16, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=100, help='batch size for testing')
        self.add_argument('--mc-batch-size', type=int, default=1000, help='batch size for Monte Carlo')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')

        

    def parse(self):
        args = self.parse_args()
        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
