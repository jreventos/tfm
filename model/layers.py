from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    """
    Create convolution layer
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param bias:
    :param padding:
    :return:
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
     Create single convolution block with non-linearity and optional batchnorm/groupnorm.
    :param in_channels (int): number of input channels
    :param out_channels (int): number of output channels
    :param kernel_size (int or tuple): size of the convolving kernel
    :param order (string): order of the layers in the convolution block
            'cr' -> conv + ReLU
            'cl' -> conv + LeakyReLU
            'gcr' -> groupnorm + conv + ReLU
            'bcr' -> batchnorm + conv + ReLU
    :param num_groups (int): number of groups for the GroupNorm
    :param padding (int or tuple):  add zero-padding added to all three sides of the input

    :return: list of tuple (name, module)
    """

    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rl', 'Non-linearity cannot be the first operation in the layer'

    # Build up the convolution block
    modules = []
    for i, char in enumerate(order): # Append layers accoring to the order of the chars

        # add ReLU activation layer
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))

        # add LeakyReLU activation layer
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))

        # add convolution layer
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))

        # add gourp normalization (better in small batches)
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels
            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))

        # add batch normalization (better in bigger batches)
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))

        # if the order does not include these layers
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm.

        :param in_channels (int): number of input channels
        :param out_channels (int): number of output channels
        :param kernel_size (int or tuple): size of the convolving kernel
        :param order (string): order of the layers in the convolution block
                'cr' -> conv + ReLU
                'cl' -> conv + LeakyReLU
                'gcr' -> groupnorm + conv + ReLU
                'bcr' -> batchnorm + conv + ReLU
        :param num_groups (int): number of groups for the GroupNorm
        :param padding (int or tuple):  add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):

    """
    Module of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

        :param in_channels (int): number of input channels
        :param out_channels (int): number of output channels
        :param encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        :param kernel_size (int or tuple): size of the convolving kernel
        :param order (string): order of the layers in the convolution block
            'cr' -> conv + ReLU
            'cl' -> conv + LeakyReLU
            'gcr' -> groupnorm + conv + ReLU
            'bcr' -> batchnorm + conv + ReLU

        :param num_groups (int): number of groups for the GroupNorm
        :param padding (int or tuple):  add zero-padding added to all three sides of the input

    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()

        if encoder:
            # ENCODER BLOCK
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels

        else:
            # DECODER BLOCK (decrease the number of channels in the 1st convolution)
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))





class Encoder(nn.Module):
    """
    Single ENCODER module consisting of the optional max/avg pooling layer.

        :param in_channels (int): number of input channels
        :param out_channels (int): number of output channels
        :param conv_kernel_size (int or tuple): size of the convolving kernel
        :param apply_pooling (bool): if True use MaxPool3d before DoubleConv
        :param pool_kernel_size (int or tuple): the size of the window
        :param pool_type (str): pooling layer: 'max' or 'avg'
        :param conv_layer_order (string): determines the order of layers
        :param num_groups (int): number of groups for the GroupNorm
        :param padding (int or tuple):  add zero-padding added to all three sides of the input

    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', conv_layer_order='gcr',
                 num_groups=2, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']

        # Pooling layer
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        # Encoder module
        self.basic_module = DoubleConv(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    Single DECODER module with the upsampling layer (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv).

        :param in_channels (int): number of input channels
        :param out_channels (int): number of output channels
        :param conv_kernel_size (int or tuple): size of the convolving kernel
        :param scale_factor (tuple): used as the multiplier for the image H/W/D in case of nn.Upsample or as stride
                                     in case of ConvTranspose3d, must reverse the MaxPool3d operation from the corresponding encoder
        :param conv_layer_order (string): determines the order of layers
        :param num_groups (int): number of groups for the GroupNorm
        :param mode (string): algorithm used for upsampling the options are the following...
                    'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
        :param padding (int or tuple):  add zero-padding added to all three sides of the input

    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1):

        super(Decoder, self).__init__()


        # interpolation for upsampling and concatenation joining
        self.upsampling = Upsampling(mode=mode)
        # concat joining
        self.joining = partial(self._joining, concat=True)

        self.basic_module = DoubleConv(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class Upsampling(nn.Module):
    """
    Upsamples a given multi-channel 3D data using interpolation

        :param mode (string): algorithm used for upsampling the options are the following...
                    'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'

    """

    def __init__(self, mode='nearest'):
        super(Upsampling, self).__init__()

        self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)