# coding: utf-8


import torch
from torchsummary import summary


class HyperReconNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, **kwargs):
        super(HyperReconNet, self).__init__()
        feature = kwargs.get('feature')
        if feature is None:
            feature = 64
        self.start_spatial = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.spatial_layers = torch.nn.ModuleList([torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1),
                                                   torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1)])
        self.start_spectral = torch.nn.ModuleList([torch.nn.Conv2d(2, feature, 9, 1, 4),
                                                   torch.nn.Conv2d(feature, feature // 2, 1, 1, 0),
                                                   torch.nn.Conv2d(feature // 2, 1, 5, 1, 2)])
        spectral_layers = []
        for ch in range(1, output_ch - 1):
            spectral_layers.append(torch.nn.ModuleList([torch.nn.Conv2d(3, feature, 9, 1, 4),
                                                             torch.nn.Conv2d(feature, feature // 2, 1, 1, 0),
                                                             torch.nn.Conv2d(feature // 2, 1, 5, 1, 2)]))
        self.spectral_layers = torch.nn.ModuleList(spectral_layers)
        self.last_spectral = torch.nn.ModuleList([torch.nn.Conv2d(2, feature, 9, 1, 4),
                                                  torch.nn.Conv2d(feature, feature // 2, 1, 1, 0),
                                                  torch.nn.Conv2d(feature // 2, 1, 5, 1, 2)])

    def forward(self, x):
        # Spatial ##############################################################
        x_spatial = torch.relu(self.start_spatial(x))
        for spatial_layer in self.spatial_layers:
            x_spatial = torch.relu(spatial_layer(x_spatial))
        _, _, _, ch = x_spatial.size()
        ########################################################################

        x_spectral_slice = x_spatial

        # Spectral #############################################################
        x_spectral_0 = x_spectral_slice[:, :2]

        for i in range(len(self.start_spectral) - 1):
            x_spectral_0 = self.start_spectral[i](x_spectral_0)
        x_spectral = [self.start_spectral[-1](x_spectral_0)]
        for i, spectral_layer in zip(range(1, ch - 1), self.spectral_layers):
            x_spectral_input = x_spectral_slice[:, (i - 1, i, i + 1)]
            for j in range(len(spectral_layer) - 1):
                x_spectral_input = torch.relu(spectral_layer[j](x_spectral_input))
            x_spectral_input = spectral_layer[-1](x_spectral_input)
            x_spectral.append(x_spectral_input)
        x_spectral_last = x_spectral_slice[:, -2:]
        for i in range(len(self.last_spectral) - 1):
            x_spectral_last = self.last_spectral[i](x_spectral_last)
        x_spectral.append(self.last_spectral[-1](x_spectral_last))
        x_spectral = torch.cat(x_spectral, dim=1)
        # ######################################################################
        return x_spectral_slice + x_spectral


if __name__ == '__main__':

    model = HyperReconNet(1, 31)
    summary(model, (1, 64, 64))
