# coding: utf-8


from .layers import ReLU, Leaky, Swish, Mish
from .layers import Ghost_layer
import torch
from torchsummary import summary


class GhostMix(torch.nn.ModuleList):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 feature_num: int=64, layer_num: int=9, **kwargs):
        super(GhostMix, self).__init__()
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = kwargs.get('activation', 'relu').lower()
        mode = kwargs.get('mode', 'normal')
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.start_activation = activations[activation]()
        self.ghost_layers = torch.nn.ModuleList([Ghost_layer(output_ch, output_ch, mode=mode) for _ in range(layer_num)])
        self.activations = torch.nn.ModuleList([activations[activation]() for _ in range(layer_num)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.start_activation(self.start_conv(x))
        x_in = x
        for i, (layer, activation) in enumerate(zip(self.ghost_layers, self.activations)):
            x = activation(layer(x))
            x = x + x_in
        return x


if __name__ == '__main__':

    model = GhostMix(1, 31, mode='mix2').to('cpu')
    summary(model, (1, 48, 48))