# coding: UTF-8


import torch
from torchinfo import summary
from model.Lambda import UNet


x = torch.rand((1, 1, 48, 48))
model = UNet(1, 31)
summary(model, (1, 1, 128, 128))
