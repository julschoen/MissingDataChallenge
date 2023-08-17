import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
import os
import copy
import argparse

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler
from pytorch_msssim import ssim

import torchvision
import torchvision.utils as vutils

from BigGANdeep import Discriminator, Generator
from BigGANdeep2 import Discriminator as D2
from BigGANdeep2 import Generator as G2
from dataset import Cats
from inpaint_tools import read_file_list


class Trainer(object):
    def __init__(self, dataset, params):
        ### Misc ###
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        self.p = params

        ### Make Models ###
        if self.p.biggan2:
            self.netG = G2(dim_z=self.p.z_size).to(self.device)
            self.y = torch.zeros(self.p.batch_size, device=self.p.device).reshape(self.p.batch_size, 1)
        else:
            self.netG = Generator(dim_z=self.p.z_size).to(self.device)

        self.scaler = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def log_interpolation(self, ims):
        torchvision.utils.save_image(
            vutils.make_grid(fake, padding=2, normalize=True)
            , os.path.join(self.images_dir, f'{step}.png'))

    def load_gen(self):
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        
        state_dict = torch.load(checkpoint)
        self.netG.load_state_dict(state_dict['modelG_state_dict'])


    def G_batch(self, ims, mask, names):
        z = torch.randn((ims.shape[0], self.p.z_size), device=self.p.device)
        z = torch.nn.Parameter(z, requires_grad=True)
        opt_ims = torch.optim.SGD([z], lr=self.p.lr, momentum=0.5)

        for i in range(self.p.niter):
            with autocast():
                if self.p.biggan2:
                    fake = self.netG(z, self.y)
                else:
                    fake = self.netG(z)

                if self.p.full:
                    loss = 1- ssim(fake*mask+1, ims+1, data_range=2)
                else:
                    loss = 1- ssim(fake*mask+ims+1, ims+1, data_range=2)

            self.scalerG.scale(loss).backward()
            self.scalerG.step(opt_ims)
            self.scalerG.update()



    def train(self):
        self.load_gen()
        for p in self.netG.parameters():
            p.requires_grad = False

        print("Starting Training...", flush=True)
        for ims, mask, name in self.dataset_train:
            self.G_batch(ims, mask, names)
        print('...Done', flush=True)


def main():
    parser = argparse.ArgumentParser()
    ## MISC & Hyper
    parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--z_size', type=int, default=128, help='Latent space dimension')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate G')
    parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
    parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
    parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
    parser.add_argument('--biggan2', type=bool, default=False)
    parser.add_argument('--data', type=str, default='validation_200')
    args = parser.parse_args()


    file_list = os.path.join("./MissingDataOpenData/", "data_splits", f"{args.data}.txt")
    file_ids = read_file_list(file_list)
    dataset_train = Cats(path="./MissingDataOpenData/", files_orig=file_ids)

    trainer = Trainer(dataset_train, params=args)
    trainer.train()

if __name__ == '__main__':
    main()



