import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os

class Cats(Dataset):
  def __init__(self, input_data_dir, files_orig=None, files_masked=None, files_mask=None):
    self.files_masked = files_masked
    self.files_mask = files_mask
    self.file_list = file_list
    self.len = len(self.file_list)
    self.input_data_dir = input_data_dir

  def __getitem__(self, index):
    ret = []
    if self.file_list:
      idx = self.file_list[index]
      p = os.path.join(self.input_data_dir, "originals", f"{idx}.jpg")
      im = io.imread(p)
      im = im.astype(np.float32)
      im = im/255
      im = (im*2)-1
      im = torch.from_numpy(im).float()
      ret.append(im)

    if self.files_masked:
      idx = self.files_masked[index]
      p = os.path.join(self.input_data_dir, "masked", f"{idx}_stroke_masked.png")
      im = io.imread(p)
      im = im.astype(np.float32)
      im = im/255
      im = (im*2)-1
      im = torch.from_numpy(im).float()
      ret.append(im)

    if self.files_mask:
      idx = self.mask[index]
      p = os.path.join(self.input_data_dir, "masks", f"{idx}_stroke_mask.png")
      im = io.imread(p)
      im = im.astype(np.float32)
      im = torch.from_numpy(im).float()
      ret.append(im)

    return ret

  def __len__(self):
    return self.len