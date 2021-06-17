import argparse
import torch
import json
import random
from pprint import pprint

# always uses cuda if avaliable
class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Hybrid Miltiscale-Dense Encoder-Decoder Networks')
        # network
        self.add_argument('--blocks', type=list, default=[4, 8, 4], help='list of number of layers in each dense block')
        self.add_argument('--growth-rate', type=int, default=16, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=64, help='number of initial features after the first conv layer')        
        self.add_argument('--drop-rate', type=float, default=0., help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enables bottleneck design in the dense blocks')
        # data 
        self.add_argument('--ntrain', type=int, default=64, help="number of training data")
        self.add_argument('--data-dir', type=str, default="./dataset", help='directory to dataset')
        self.add_argument('--kle', type=int, default=100, help='num of KLE terms')
        self.add_argument('--imsize', type=int, default=128*128, help='data dimension')
        self.add_argument('--ntest', type=int, default=256, help="number of test data") 
        self.add_argument('--nmcs', type=int, default=10000, help="number of test data") 
        # training
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
        self.add_argument('--lr', type=float, default=1e-4, help='learnign rate')
        self.add_argument('--weight-decay', type=float, default=1e-5, help="weight decay")
        self.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
        self.add_argument('--mcs-batch', type=int, default=50, help='mcs batch size for tesing (default: 50)')
        self.add_argument('--test-batch-size', type=int, default=100, help='input batch size for testing (default: 100)')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')


    def parse(self):
        args = self.parse_args()
        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print('Arguments:')
        pprint(vars(args))
        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
