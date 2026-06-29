import math
import copy
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from torchvision import transforms as T
from PIL import Image


# import xformers, xformers.ops

from utils import *

import argparse
from diffdrr.drr import DRR
from diffdrr.data import load_example_ct

class Reconstruction(torch.nn.Module):
    def __init__(self, subject, device):
        super().__init__()
        self.drr = DRR(load_example_ct(), sdd=1020.0, height=224, delx=1.1).to(device=device)
        self.density = subject.clone()

    def forward(self, pose, **kwargs):
        source, target = self.drr.detector(pose, None)
        img = self.drr.render(self.density, source, target)
        return self.drr.reshape_transform(img, batch_size=len(pose))

