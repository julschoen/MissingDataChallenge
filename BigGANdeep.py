import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d

# BigGAN-deep: uses a different resblock and pattern


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.

# Channel ratio is the ratio of 
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=None,
               upsample=None, channel_ratio=4):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.in_channels // channel_ratio
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(self.in_channels)
    self.bn2 = self.which_bn(self.hidden_channels)
    self.bn3 = self.which_bn(self.hidden_channels)
    self.bn4 = self.which_bn(self.hidden_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x):
    # Project down to channel ratio
    h = self.conv1(self.activation(self.bn1(x)))
    # Apply next BN-ReLU
    h = self.activation(self.bn2(h))
    # Drop channels in x if necessary
    if self.in_channels != self.out_channels:
      x = x[:, :self.out_channels]      
    # Upsample both h and x at this point  
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    # 3x3 convs
    h = self.conv2(h)
    h = self.conv3(self.activation(self.bn3(h)))
    # Final 1x1 conv
    h = self.conv4(self.activation(self.bn4(h)))
    return h + x

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}

  arch[360] = {'in_channels' :  [ch*item for item in [16, 8, 8, 4, 2, 1, 1]],
               'out_channels' : [item * ch for item in [8, 8, 4, 2, 1]],
               'upsample' : [True] * 6 + [False],
               'resolution' : [5 , 11, 22, 45, 90, 180, 360],
                'attention' : {45:45}}

  return arch

class Generator(nn.Module):
    def __init__(self, G_ch=64, G_depth=2, dim_z=512, resolution=360,
               G_kernel_size=3, G_attn='64', n_classes=1,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',
               **kwargs):
      super(Generator, self).__init__()
      # Channel width mulitplier
      self.ch = G_ch
      # Number of resblocks per stage
      self.G_depth = G_depth
      # Dimensionality of the latent space
      self.dim_z = dim_z
      # The initial spatial dimensions
      self.bottom_width = 5
      # Resolution of the output
      self.resolution = 360
      # Kernel size?
      self.kernel_size = G_kernel_size
      # Attention?
      self.attention = G_attn
      # number of classes, for use in categorical conditional generation
      self.n_classes = n_classes
      # Use shared embeddings?
      self.G_shared = G_shared
      # Dimensionality of the shared embedding? Unused if not using G_shared
      self.shared_dim = shared_dim if shared_dim > 0 else dim_z
      # Hierarchical latent space?
      self.hier = hier
      # Cross replica batchnorm?
      self.cross_replica = cross_replica
      # Use my batchnorm?
      self.mybn = mybn
      # nonlinearity for residual blocks
      self.activation = G_activation
      # Initialization style
      self.init = G_init
      # Parameterization style
      self.G_param = G_param
      # Normalization style
      self.norm_style = norm_style
      # Epsilon for BatchNorm?
      self.BN_eps = BN_eps
      # Epsilon for Spectral Norm?
      self.SN_eps = SN_eps
      # fp16?
      self.fp16 = G_fp16
      # Architecture dict
      self.arch = G_arch(self.ch, self.attention)[resolution]


      # Which convs, batchnorms, and linear layers to use
      if self.G_param == 'SN':
        self.which_conv = functools.partial(layers.SNConv2d,
                            kernel_size=3, padding=1,
                            num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                            eps=self.SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                            num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                            eps=self.SN_eps)
      else:
        self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
        self.which_linear = nn.Linear
        
      # We use a non-spectral-normed embedding here regardless;
      # For some reason applying SN to G's embedding seems to randomly cripple G
      self.which_embedding = nn.Embedding
      bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                   else self.which_embedding)


      # Prepare model
      # If not using shared embeddings, self.shared is just a passthrough
      self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                      else layers.identity())
      # First linear layer
      self.linear = self.which_linear(self.dim_z, self.arch['in_channels'][0] * (self.bottom_width **2))

      # self.blocks is a doubly-nested list of modules, the outer loop intended
      # to be over blocks at a given resolution (resblocks and/or self-attention)
      # while the inner loop is over a given block
      self.blocks = []
      for index in range(len(self.arch['out_channels'])):
        self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                               out_channels=self.arch['in_channels'][index] if g_index==0 else self.arch['out_channels'][index],
                               which_conv=self.which_conv,
                               activation=self.activation,
                               upsample=(functools.partial(F.interpolate, scale_factor=2)
                                         if self.arch['upsample'][index] and g_index == (self.G_depth-1) else None))]
                         for g_index in range(self.G_depth)]

        # If attention on this block, attach it to the end
        if self.arch['resolution'][index] == 45:
          print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
          self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

      # Turn self.blocks into a ModuleList so that it's all properly registered.
      self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

      # output layer: batchnorm-relu-conv.
      # Consider using a non-spectral conv here
      self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                  cross_replica=self.cross_replica,
                                                  mybn=self.mybn),
                                      self.activation,
                                      self.which_conv(self.arch['out_channels'][-1], 3))

      # Initialize weights. Optionally skip init for testing.
      if not skip_init:
        self.init_weights()

      # Set up optimizer
      # If this is an EMA copy, no need for an optim, so just return now
      if no_optim:
        return
      self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
      if G_mixed_precision:
        print('Using fp16 adam in G...')
        import utils
        self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0,
                             eps=self.adam_eps)
      else:
        self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0,
                             eps=self.adam_eps)

      # LR scheduling, left here for forward compatibility
      # self.lr_sched = {'itr' : 0}# if self.progressive else {}
      # self.j = 0

    # Initialize
    def init_weights(self):
      self.param_count = 0
      for module in self.modules():
        if (isinstance(module, nn.Conv2d) 
            or isinstance(module, nn.Linear) 
            or isinstance(module, nn.Embedding)):
          if self.init == 'ortho':
            init.orthogonal_(module.weight)
          elif self.init == 'N02':
            init.normal_(module.weight, 0, 0.02)
          elif self.init in ['glorot', 'xavier']:
            init.xavier_uniform_(module.weight)
          else:
            print('Init style not recognized...')
          self.param_count += sum([p.data.nelement() for p in module.parameters()])
      print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    # NOTE: The z vs y dichotomy here is for compatibility with not-y
    def forward(self, z):
      h = self.linear(z)
      # Reshape
      h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)    
      # Loop over blocks
      for index, blocklist in enumerate(self.blocks):
        # Second inner loop in case block has multiple layers
        for block in blocklist:
          h = block(h)
          print(h.shape)
          
      # Apply batchnorm-relu-conv-tanh at output
      out = self.output_layer(h)
      print(out.shape)
      return torch.tanh(out)

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=layers.SNConv2d, wide=True,
               preactivation=True, activation=None, downsample=None,
               channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
    
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)
    
# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}

  arch[360] = {'in_channels' :  [3]+[ch*item for item in [1, 1, 2, 4, 8, 8]],
               'out_channels' : [item * ch for item in [1, 1, 2, 4, 8, 8, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [180, 90, 45, 22, 11, 5, 5],
               'attention' : {45:45}}

  return arch

class Discriminator(nn.Module):

  def __init__(self, D_ch=64, D_wide=True, D_depth=2, resolution=360,
               D_kernel_size=3, D_attn='64', n_classes=1,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # How many resblocks per stage?
    self.D_depth = D_depth
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]


    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    
    
    # Prepare model
    # Stem convolution
    self.input_conv = self.which_conv(3, self.arch['in_channels'][0])
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index] if d_index==0 else self.arch['out_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=True,
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] and d_index==0 else None))
                       for d_index in range(self.D_depth)]]
      # If attention on this block, attach it to the end
      if self.arch['resolution'][index] == 45:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Run input conv
    h = self.input_conv(x)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    print(h.shape)
    out = self.linear(h)
    return out
