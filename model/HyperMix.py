# coding: UTF-8


import torch
from .layers import Mix_SS_Layer, GroupConv


class HyperMixNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, *args, feature_num=64, 
                 block_num=9, group_num=1, **kwargs):
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.mix_conv = torch.nn.ModuleList([Mix_SS_Layer(feature_num, feature_num,  
                                                          chuncks, activation=activation) 
                                             for _ in range(block_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.input_conv(x)
        for i, layer in enumerate(self.mix_conv):
            x = layer(x)
        x = self.output_conv(x)
        return x


class VanillaNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, *args, feature_num=64, 
                 block_num=9, group_num=1, **kwargs):
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.vanilla_conv = torch.nn.ModuleList([GroupConv(feature_num, feature_num,  
                                                           chuncks, activation=activation) 
                                             for _ in range(block_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.input_conv(x)
        for i, layer in enumerate(self.vanilla_conv):
            x = layer(x)
        x = self.output_conv(x)
        return x
