# coding: utf-8


import torch


# ########################## Activation Function ##########################


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class FReLU(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride=1, **kwargs):
        super(FReLU, self).__init__()
        self.depth = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2, groups=input_ch)

    def forward(self, x):
        depth = self.depth(x)
        return torch.max(x, depth)


class ReLU(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        pass

    def forward(self, x):
        return torch.relu(x)


class Leaky(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Leaky, self).__init__()
        self.alpha = kwargs.get('alpha', .02)

    def forward(self, x):
        return torch.where(x < 0., x * self.alpha, x)


class Swish(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()
        pass

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Mish, self).__init__()
        pass

    def forward(self, x):
        return mish(x)


# ########################## Loss Function ##########################


class SAMLoss(torch.nn.Module):

    def forward(self, x, y):
        x_sqrt = torch.norm(x, dim=1)
        y_sqrt = torch.norm(y, dim=1)
        xy = torch.sum(x * y, dim=1)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class MSE_SAMLoss(torch.nn.Module):

    def __init__(self, alpha=.5, beta=.5, mse_ratio=1., sam_ratio=.01):
        super(MSE_SAMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = torch.nn.MSELoss()
        self.sam_loss = SAMLoss()
        self.mse_ratio = mse_ratio
        self.sam_ratio = sam_ratio

    def forward(self, x, y):
        return self.alpha * self.mse_ratio * self.mse_loss(x, y) + self.beta * self.sam_ratio * self.sam_loss(x, y)


# ########################## Loss Function ##########################


class Base_Module(torch.nn.Module):

    def __init__(self):
        super(Base_Module, self).__init__()

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        # elif self.activation == 'frelu':
        #     return FReLU(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x


class DW_PT_Conv(Base_Module):

    def __init__(self, input_ch, output_ch, kernel_size, activation='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activation
        self.depth = torch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.depth(x)
        x = self._activation_fn(x)
        x = self.point(x)
        x = self._activation_fn(x)
        return x


class HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(HSI_prior_block, self).__init__()
        self.activation = activation
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.spectral = torch.nn.Conv2d(output_ch, input_ch, 1, 1, 0)

    def forward(self, x):
        x_in = x
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class My_HSI_network(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(My_HSI_network, self).__init__()
        self.activation = activation
        self.spatial_1 = DW_PT_Conv(input_ch, feature, 3, activation=None)
        self.spatial_2 = DW_PT_Conv(feature, output_ch, 3, activation=None)
        self.spectral = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x_in = x
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class RAM(Base_Module):

    def __init__(self, input_ch, output_ch, ratio=None, **kwargs):
        super(RAM, self).__init__()

        if ratio is None:
            self.ratio = 2
        else:
            self.ratio = ratio
        self.activation = kwargs.get('attn_activation')
        self.spatial_attn = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1, groups=input_ch)
        self.spectral_pooling = GVP()
        self.spectral_Linear = torch.nn.Linear(input_ch, input_ch // self.ratio)
        self.spectral_attn = torch.nn.Linear(input_ch // self.ratio, output_ch)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        spatial_attn = self._activation_fn(self.spatial_attn(x))
        spectral_pooling = self.spectral_pooling(x).view(-1, ch)
        spectral_linear = torch.relu(self.spectral_Linear(spectral_pooling))
        spectral_attn = self.spectral_attn(spectral_linear).unsqueeze(-1).unsqueeze(-1)

        # attn_output = torch.sigmoid(spatial_attn + spectral_attn + spectral_pooling.unsqueeze(-1).unsqueeze(-1))
        attn_output = torch.sigmoid(spatial_attn * spectral_attn)
        output = attn_output * x
        return output


class Global_Average_Pooling2d(torch.nn.Module):

    def forward(self, x):
        bs, ch, h, w = x.size()
        return torch.nn.functional.avg_pool2d(x, kernel_size=(h, w)).view(-1, ch)


class GVP(torch.nn.Module):

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        avg_x = torch.nn.functional.avg_pool2d(x, kernel_size=(h, w))
        var_x = torch.nn.functional.avg_pool2d((x - avg_x) ** 2, kernel_size=(h, w))
        return var_x.view(-1, ch)


class SE_block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, **kwargs):
        super(SE_block, self).__init__()
        if 'ratio' in kwargs:
            ratio = kwargs['ratio']
        else:
            ratio = 2
        mode = kwargs.get('mode')
        if mode == 'GVP':
            self.pooling = GVP()
        else:
            self.pooling = Global_Average_Pooling2d()
        self.squeeze = torch.nn.Linear(input_ch, output_ch // ratio)
        self.extention = torch.nn.Linear(output_ch // ratio, output_ch)

    def forward(self, x):
        gap = torch.mean(x, [2, 3], keepdim=True)
        squeeze = self._activation_fn(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torch.sigmoid(extention) * x


class Attention_HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, **kwargs):
        super(Attention_HSI_prior_block, self).__init__()
        self.mode = kwargs.get('mode')
        ratio = kwargs.get('ratio')
        if ratio is None:
            ratio = 2
        attn_activation = kwargs.get('attn_activation')
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.attention = RAM(output_ch, output_ch, ratio=ratio, attn_activation=attn_activation)
        self.spectral = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        if self.mode is not None:
            self.spectral_attention = SE_block(output_ch, output_ch, mode=self.mode, ratio=ratio)
        self.activation = kwargs.get('activation')

    def forward(self, x):
        x_in = x
        h = self._activation_fn(self.spatial_1(x))
        h = self.spatial_2(h)
        h = self.attention(h)
        x = h + x_in
        x = self.spectral(x)
        if self.mode is not None:
            x = self.spectral_attention(x)
        return x
