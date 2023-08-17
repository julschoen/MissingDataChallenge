import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
import os
import copy
import argparse
import pathlib
from skimage import io

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler
from pytorch_msssim import ssim, ms_ssim

import torchvision
import torchvision.utils as vutils

from BigGANdeep import Discriminator, Generator
from BigGANdeep2 import Discriminator as D2
from BigGANdeep2 import Generator as G2
from dataset import Cats
from inpaint_tools import read_file_list
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list

class Params(object):
    def __init__(self):
        self.niters=5000
        self.batch_size=16
        self.z_size=128
        self.lr=1e-1
        self.device='cuda'
        self.biggan2=True
        self.full=True
        self.models_dir='./big2/models/'


class Trainer(object):
    def __init__(self, dataset, settings):
        ### Misc ###
        params = Params()
        self.config = settings
        self.device = params.device

        ### Make Dirs ###
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
        ims = ims.permute(0,2,3,1).detach().cpu().numpy()
        ims = (ims+1)/2
        ims = ims*255
        ims = ims.astype(np.uint8)
        for i, idx in enumerate(names):
            im = ims[i]
            out_image_name = os.path.join(self.inpainted_result_dir, f"{idx}.png")
            io.imsave(out_image_name, im)
    
    def load_gen(self):
        checkpoint = os.path.join(self.p.models_dir, 'checkpoint.pt')
        
        state_dict = torch.load(checkpoint)
        self.netG.load_state_dict(state_dict['modelG_state_dict'])

    def G_batch(self, ims, mask, names):
        z = torch.randn((ims.shape[0], self.p.z_size), device=self.p.device)
        z = torch.nn.Parameter(z, requires_grad=True)
        opt_ims = torch.optim.SGD([z], lr=self.p.lr, momentum=0.5)
        mask = mask/255
        mask = (mask -1)*-1

        for i in range(self.p.niters):
            opt_ims.zero_grad()
            with autocast():
                if self.p.biggan2:
                    fake = self.netG(z, self.y)
                else:
                    fake = self.netG(z)

                if self.p.full:
                    fake_ = fake * ((mask-1)*-1)
                    loss = 1- ssim(fake_+1, ims+1, data_range=2)
                else:
                    fake = (fake * mask)+ims
                    loss = 1- ms_ssim(fake+1, ims+1, data_range=2)

            if i%100 == 0:
                print('[%d|%d] Loss: %.4f' % (i, self.p.niters, loss.detach().item()), flush=True)

            self.scaler.scale(loss).backward()
            self.scaler.step(opt_ims)
            self.scaler.update()

        self.log((fake * mask)+ims, names)


    def train(self):
        self.load_gen()
        for p in self.netG.parameters():
            p.requires_grad = False

        print("Starting Training...", flush=True)
        for i, x in enumerate(self.generator_train):
            ims, mask, names = x
            ims = ims/255
            ims = (ims*2)-1
            ims = ims.permute(0,3,1,2).to(self.p.device)
            mask = mask.unsqueeze(1).to(self.p.device)
            self.G_batch(ims, mask, names)
        print('...Done', flush=True)


def main():
    args = argparse.ArgumentParser(description='InpaintImages')
    config = InPaintConfig(args).settings

    file_list = os.path.join("./MissingDataOpenData/", "data_splits", f"{config['data_set']}.txt")
    file_ids = read_file_list(file_list)
    dataset_train = Cats(path="./MissingDataOpenData/", files_masked=file_ids, files_mask=file_ids, file=True)

    trainer = Trainer(dataset_train, config)
    trainer.train()

if __name__ == '__main__':
    main()



