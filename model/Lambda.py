# coding: UTF-8


import torch


class UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, layer_num=6, **kwargs):

        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        feature_num = [2 ** (5 + i) for i in range(layer_num)]
        self.input_conv = torch.nn.Conv2d(input_ch, feature_num[0], 3, 1, 1)
        self.encoder =  torch.nn.ModuleList([torch.nn.Conv2d(feature_num[i], 
                                                             feature_num[i + 1], 3, 1, 1)
                                             for i in range(len(feature_num) - 1)
                                             for _ in range(3)])

    def forward(sefl, x):
        x = self.input_conv(x)
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        return x
