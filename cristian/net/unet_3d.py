import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    
    return conv

def crop_image(tensor, target_tensor):
    target_size = target_tensor.size()[3]
    tensor_size = tensor.size()[3]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool_2x2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 16)
        self.down_conv_2 = double_conv(16, 32)
        self.down_conv_3 = double_conv(32, 64)
        self.down_conv_4 = double_conv(64, 128)
        self.down_conv_5 = double_conv(128, 256)
        
        self.up_trans_1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(256, 128)
        self.up_trans_2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(128, 64)
        self.up_trans_3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(64, 32)
        self.up_trans_4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(32, 16)
        
        self.out = nn.Conv3d(in_channels=16, out_channels=2, kernel_size=1)
        
    def forward(self, image):
        #encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        
        #decoder
        x = self.up_trans_1(x9)
        y = crop_image(x7, x)
        x = self.up_conv_1(torch.cat([x, y], axis=1))
        
        x = self.up_trans_2(x)
        y = crop_image(x5, x)
        x = self.up_conv_2(torch.cat([x, y], axis=1))
        
        x = self.up_trans_3(x)
        y = crop_image(x3, x)
        x = self.up_conv_3(torch.cat([x, y], axis=1))
        
        x = self.up_trans_4(x)
        y = crop_image(x1, x)
        x = self.up_conv_4(torch.cat([x, y], axis=1))
        
        x = self.out(x)
        return x
        
        
if __name__ == '__main__':
    image = torch.rand(1, 1, 64, 64, 64)
    model = UNet()
    print(model(image))
