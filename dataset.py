import numpy as np
import torch
from skimage import io
from torch.utils.data.dataset import Dataset
import os

class Cats(Dataset):
  def __init__(self, path, files_orig=None, files_masked=None, files_mask=None):
    self.files_masked = files_masked
    self.files_mask = files_mask
    self.files_orig = files_orig

    if self.files_orig:
      self.len = len(self.files_orig)

    if self.files_masked:
      self.len = len(self.files_masked)

    if self.files_masked:
      self.len = len(self.files_masked)
    self.input_data_dir = path

  def __getitem__(self, index):
    ret = []
    if self.files_orig:
      idx = self.files_orig[index]
      p = os.path.join(self.input_data_dir, "originals", f"{idx}.jpg")
      im = io.imread(p)
      im = im.astype(np.float32)
      #im = im/255
      #im = (im*2)-1
      im = torch.from_numpy(im).float()
      ret.append(im)

    if self.files_masked:
      idx = self.files_masked[index]
      p = os.path.join(self.input_data_dir, "masked", f"{idx}_stroke_masked.png")
      im = io.imread(p)
      im = im.astype(np.float32)
      #im = im/255
      #im = (im*2)-1
      im = torch.from_numpy(im).float()
      ret.append(im)

    if self.files_mask:
      idx = self.mask[index]
      p = os.path.join(self.input_data_dir, "masks", f"{idx}_stroke_mask.png")
      im = io.imread(p)
      im = im.astype(np.float32)
      im = torch.from_numpy(im).float()
      ret.append(im)

    if len(ret) == 1:
      return ret[0]
    if len(ret) == 2:
      return ret[0], ret[1]
    if len(ret) == 3:
      return ret[0], ret[1], ret[2]

    return ret

  def __len__(self):
    return self.len