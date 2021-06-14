# coding: UTF-8


import torch
import torchinfo
from model.Lambda import UNet


model = UNet(1, 31)
summary(model, (1, 1, 48, 48))
