from model.layers import Encoder, Decoder
import torch
from torch import nn as nn
from torch.nn import functional as F


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

        :param in_channels (int): number of input channels
        :param out_channels (int): number of output segmentation masks (2 if softmax is used)
        :param f_maps (int, tuple):  number of feature maps at each level of the encoder; if it's an integer the number
                                    of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
                                    number of feature maps in the first conv layer of the encoder (default: 64);
                                     if tuple: number of feature maps at each level

        :param final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax.
        :param layer_order (string): determines the order of layers
        :param num_groups (int): number of groups for the GroupNorm
        :param num_levels (int): number of levels in the encoder/decoder path
        :param is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
                                       after the final convolution; if False (regression problem) the normalization layer is skipped at the end

        :param testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
                               will be applied as the last operation during the forward pass; if False the model is in training mode
                               and the `final_activation` (even if present) won't be applied; default: False

        :param conv_kernel_size (int or tuple): size of the convolving kernel
        :param pool_kernel_size (int or tuple): the size of the window (by default MAX pooling)
        :param conv_padding (int or tuple):  add zero-padding added to all three sides of the input


    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='gcr',
                 num_groups=4, num_levels=2, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):

        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
            #print(f_maps)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                # First Encoder block
                #print('In:',in_channels)
                #print('Out:',out_channels)
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,          # skip pooling in the firs encoder
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding)
            else:
                # Next Encoder blocks
                #print('In:', f_maps[i - 1])
                #print('Out:', out_feature_num)
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        #print(reversed_f_maps)

        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            #print('In:',in_feature_num)
            out_feature_num = reversed_f_maps[i + 1]
            #print('Out:',out_feature_num)

            decoder = Decoder(in_feature_num, out_feature_num,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:

            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     f_maps=f_maps, layer_order=layer_order,
                                     num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation,
                                     conv_padding=conv_padding, **kwargs)

if __name__ == '__main__':
    import time
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1,out_channels=2)

    print('Parameters:',sum(p.data.nelement() for p in net.parameters()))
    #print(net)
    net.to(device)
    data = torch.Tensor(1, 1, 32, 32, 32)
    data = data.type(torch.cuda.FloatTensor)
    data = data.to(device)
    start_time = time.time()
    out = net(data)
    print(out)
    print(out.shape)