# coding: UTF-8


import torch


class UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, layer_num=6, **kwargs):

        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        encode_feature_num = [2 ** (5 + i) for i in range(layer_num)]
        decode_feature_num = encode_feature_num[::-1]
        self.skip_encode_feature = [2, 7, 12, 17, 21]
        self.input_conv = torch.nn.Conv2d(input_ch, encode_feature_num[0], 3, 1, 1)
        self.encoder = torch.nn.ModuleList([
                torch.nn.Conv2d(encode_feature_num[0], encode_feature_num[0], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[0], encode_feature_num[0], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[0], encode_feature_num[0], 3, 1, 1),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(encode_feature_num[0], encode_feature_num[1], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[1], encode_feature_num[1], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[1], encode_feature_num[1], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[1], encode_feature_num[1], 3, 1, 1),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(encode_feature_num[1], encode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[2], encode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[2], encode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[2], encode_feature_num[2], 3, 1, 1),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(encode_feature_num[2], encode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[3], encode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[3], encode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[3], encode_feature_num[3], 3, 1, 1),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(encode_feature_num[3], encode_feature_num[4], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[4], encode_feature_num[4], 3, 1, 1),
                torch.nn.Conv2d(encode_feature_num[4], encode_feature_num[4], 3, 1, 1),
                torch.nn.MaxPool2d((2, 2)),
                torch.nn.Conv2d(encode_feature_num[4], encode_feature_num[5], 3, 1, 1),
                ])
        self.bottleneck1 = torch.nn.Conv2d(encode_feature_num[-1], encode_feature_num[-1], 3, 1, 1)
        self.bottleneck2 = torch.nn.Conv2d(encode_feature_num[-1], decode_feature_num[0], 3, 1, 1)
        self.skip_decode_feature = [1, 4, 8, 12, 16]
        self.decoder = torch.nn.ModuleList([
                torch.nn.Conv2d(decode_feature_num[0], decode_feature_num[0], 3, 1, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[0], decode_feature_num[1], 2, 2),
                torch.nn.Conv2d(decode_feature_num[1] + encode_feature_num[-2], 
                                decode_feature_num[1], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[1], decode_feature_num[1], 3, 1, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[1], decode_feature_num[2], 2, 2),
                torch.nn.Conv2d(decode_feature_num[2], decode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[2], decode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[2], decode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[2] + encode_feature_num[-3], 
                                decode_feature_num[2], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[3], decode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[3], decode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[3], decode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[3] + encode_feature_num[-4], 
                                decode_feature_num[3], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[4], decode_feature_num[4], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[4], decode_feature_num[4], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[4], decode_feature_num[4], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[4] + encode_feature_num[-5], 
                                decode_feature_num[4], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[5], decode_feature_num[5], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[5], decode_feature_num[5], 3, 1, 1),
                torch.nn.Conv2d(decode_feature_num[5], decode_feature_num[5], 3, 1, 1),
                ])
        self.output_conv = torch.nn.Conv2d(decode_feature_num[-1], output_ch, 3, 1, 1)

    def forward(self, x):
        x = self.input_conv(x)
        encode_feature = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_encode_feature:
                encode_feature.append(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        j = 1
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i in self.skip_decode_feature:
                print(x.size(), encode_feature[-j].size())
                x = torch.cat([x, encode_feature[-j]], dim=1)
                j += 1
        x = self.output_conv(x)
        return x
