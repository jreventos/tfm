import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_chan, out_chan, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm3d(out_chan),
        nn.ReLU(inplace=True) # TODO: also try LeakyRelu
    )

def conv_stage(in_chan, out_chan):
    return nn.Sequential(
        conv_block(in_chan, out_chan),
        conv_block(out_chan, out_chan),
    )


class UNet(nn.Module):
    """
    This is a 3D segmentation framework of UNet for medical volume by pytorch
    The network is inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation and
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = conv_stage(1, 16)
        self.enc2 = conv_stage(16, 32)
        self.enc3 = conv_stage(32, 64)
        self.enc4 = conv_stage(64, 128)
        self.enc5 = conv_stage(128, 128)
        self.pool = nn.MaxPool3d(2, 2)

        # Decoder
        self.dec4 = conv_stage(256, 64)
        self.dec3 = conv_stage(128, 32)
        self.dec2 = conv_stage(64, 16)
        self.dec1 = conv_stage(32, 16)
        self.conv_out = nn.Conv3d(16, out_channels=2, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):

        # ENCODER
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # DECODER (
        dec4 = self.dec4(torch.cat((enc4, F.interpolate(enc5, enc4.size()[2:], mode='trilinear')), 1))
        dec3 = self.dec3(torch.cat((enc3, F.interpolate(dec4, enc3.size()[2:], mode='trilinear')), 1))
        dec2 = self.dec2(torch.cat((enc2, F.interpolate(dec3, enc2.size()[2:], mode='trilinear')), 1))
        dec1 = self.dec1(torch.cat((enc1, F.interpolate(dec2, enc1.size()[2:], mode='trilinear')), 1))
        out = self.conv_out(dec1)
        out = self.final_activation(out)
        return out


#
# if __name__ == '__main__':
#     import time
#     import torch
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net = UNet()
#     print('Parameters:',sum(p.data.nelement() for p in net.parameters()))
#     print(net)
#     net.to(device)
#     data = torch.Tensor(1, 1, 32, 32, 32)
#     start_time = time.time()
#
#     out = net(data.to(device))
#     print(out)
#     print(out.shape)
#     count = 0

