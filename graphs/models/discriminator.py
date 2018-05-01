"""
DCGAN discriminator model
based on the paper: https://arxiv.org/pdf/1511.06434.pdf
date: 30 April 2018
"""
import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.LeakyReLU(self.config.relu_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.num_filt_d, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=self.config.num_filt_d, out_channels=self.config.num_filt_d * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(self.config.num_filt_d*2)

        self.conv3 = nn.Conv2d(in_channels=self.config.num_filt_d*2, out_channels=self.config.num_filt_d * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(self.config.num_filt_d*4)

        self.conv4 = nn.Conv2d(in_channels=self.config.num_filt_d*4, out_channels=self.config.num_filt_d*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.config.num_filt_d*8)

        self.conv5 = nn.Conv2d(in_channels=self.config.num_filt_d*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)

        self.out = nn.Sigmoid()


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm1(out)
        out =  self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm2(out)
        out =  self.relu(out)

        out = self.conv4(out)
        out = self.batch_norm3(out)
        out =  self.relu(out)

        out = self.conv5(out)
        out = self.out(out)

        return out.view(-1, 1).squeeze(1)


"""
netD testing
"""
def main():
    config = json.load(open('../../configs/GAN_config.json'))
    config = edict(config)
    inp  = torch.autograd.Variable(torch.randn(config.batch_size, config.input_channels, config.image_size, config.image_size))
    print (inp.shape)
    netD = Discriminator(config)
    out = netD(inp)
    print (out)

if __name__ == '__main__':
    main()

"""
#########################
Architecture:
#########################

Input: (N, 3, 64, 64)

conv1: (N, 64, 32, 32)   ==> H/2, W/2
conv2: (N, 128, 16, 16)  ==> H/4, W/4
conv3: (N, 256, 8, 8)    ==> H/8, W/8
conv4: (N, 512, 4, 4)    ==> H/16, W/16
conv5: (N, 1, 1, 1)      ==> H/64, W/64

out: (N)
----
torch.Size([4, 3, 64, 64])
torch.Size([4, 64, 32, 32])
torch.Size([4, 128, 16, 16])
torch.Size([4, 256, 8, 8])
torch.Size([4, 512, 4, 4])
torch.Size([4, 1, 1, 1])
"""