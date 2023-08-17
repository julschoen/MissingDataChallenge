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

class Params(object):
    self.niters=5000
    self.batch_size=16
    self.z_size=128
    self.lr=1e-3
    self.device='cuda'
    self.biggan2=True
    self.full=True


class Trainer(object):
    def __init__(self, dataset, config):
        ### Misc ###
        params = Params()
        self.config = config
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


        input_data_dir = settings["dirs"]["input_data_dir"]
        output_data_dir = settings["dirs"]["output_data_dir"]
        data_set = settings["data_set"]

        inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")
        pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

        self.inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def log(self, ims, names):
        for i, idx in enumerate(names):
            im = ims[i]
            out_image_name = os.path.join(self.inpainted_result_dir, f"{idx}.png")
            io.imsave(out_image_name, im)
    
    def load_gen(self):
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        
        state_dict = torch.load(checkpoint)
        self.netG.load_state_dict(state_dict['modelG_state_dict'])

    def G_batch(self, ims, mask, names):
        z = torch.randn((ims.shape[0], self.p.z_size), device=self.p.device)
        z = torch.nn.Parameter(z, requires_grad=True)
        opt_ims = torch.optim.SGD([z], lr=self.p.lr, momentum=0.5)

        mask = (mask -1)*-1

        for i in range(self.p.niters):
            with autocast():
                if self.p.biggan2:
                    fake = self.netG(z, self.y)
                else:
                    fake = self.netG(z)

                if self.p.full:
                    fake[torch.where(mask == 1)] = 0.0
                    loss = 1- ssim((fake+1, ims+1, data_range=2))
                else:
                    fake[torch.where(mask == 0)] = 0
                    loss = 1- ssim(fake+ims+1, ims+1, data_range=2)

            if i%100 == 0:
                print('[%d|%d] Loss: %.4f' % (step, self.p.niters, loss.detach().item()), flush=True)

            self.scalerG.scale(loss).backward()
            self.scalerG.step(opt_ims)
            self.scalerG.update()

        self.log(fake, names)



    def train(self):
        self.load_gen()
        for p in self.netG.parameters():
            p.requires_grad = False

        print("Starting Training...", flush=True)
        for ims, mask, name in self.dataset_train:
            self.G_batch(ims, mask, names)
        print('...Done', flush=True)


def main():
    args = argparse.ArgumentParser(description='InpaintImages')
    config = InPaintConfig(args)

    file_list = os.path.join("./MissingDataOpenData/", "data_splits", f"{args.data_set}.txt")
    file_ids = read_file_list(file_list)
    dataset_train = Cats(path="./MissingDataOpenData/", files_orig=file_ids, file=True)

    trainer = Trainer(dataset_train, params=config)
    trainer.train()

if __name__ == '__main__':
    main()



