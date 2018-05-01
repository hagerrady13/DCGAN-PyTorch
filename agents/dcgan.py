import numpy as np

from tqdm import tqdm
import shutil

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from graphs.models.generator import Generator
from graphs.models.discriminator import Discriminator
from graphs.losses.loss import BinaryCrossEntropy
from datasets.celebA import CelebADataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate
from utils.misc import print_cuda_statistics

cudnn.benchmark = True

class DCGANAgent:

    def __init__(self, config):
        self.config = config

        # define models ( generator and discriminator)
        self.netG = Generator(self.config)
        self.netD = Discriminator(self.config)

        # define dataloader
        self.dataloader = CelebADataLoader(self.config)

        # define loss
        self.loss = BinaryCrossEntropy()

        # define optimizers for both generator and discriminator
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.config.learning_rate,betas=(self.config.beta1, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.config.learning_rate,betas=(self.config.beta1, 0.999))

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_mean_iou = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        torch.manual_seed(self.config.seed)

        if self.cuda:
            print("Program will run on *****GPU-CUDA***** ")
            torch.cuda.manual_seed_all(self.config.seed)
            print_cuda_statistics()

            self.vgg_model = self.vgg_model.cuda()
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
        else:
            print("Program will run on *****CPU***** ")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='DCGAN')

    def load_checkpoint(self, D_file_name, G_file_name):
        pass

    def save_checkpoint(self, D_file_name = "D_checkpoint.pth.tar", G_file_name = "G_checkpoint.pth.tar", is_best = 0):
        pass

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        pass

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.dataloader.finalize()
