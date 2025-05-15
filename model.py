import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, p=1):
        super().__init__()
        self.ks = ks
        self.p = p        

        self.block1 = self.conv_block(in_channels = in_channels, out_channels = out_channels)
        self.block2 = self.conv_block(in_channels = out_channels, out_channels = out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=self.ks, padding=self.p),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block2(self.block1(x))
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()        
        self.downsample_block = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                             UNetBlock(in_channels, out_channels))

    def forward(self, x):
        x = self.downsample_block(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, mode, upsample=None):
        super().__init__()
        assert mode.lower() in ['bilinear', 'nearest', 'tr_conv']
        if mode.lower() in ['bilinear', 'nearest']:
            upsample = True
            up_mode = mode

        self.upsample = nn.Upsample(scale_factor = 2, mode = up_mode) if upsample else nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        pad_x = x2.shape[3] - x1.shape[3]
        pad_y = x2.shape[2] - x2.shape[2]
        pad_xx, pad_yy = pad_x // 2, pad_y // 2

        x1 = F.pad(x1, pad = [pad_xx, pad_x - pad_xx, pad_yy, pad_y - pad_yy])
        concat = torch.cat([x1, x2], dim=1)

        return self.conv(concat)


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

  def __init__(self, in_channels, out_channels, num_classes, up_method):
    super().__init__()

    self.init_conv = UNetBlock(in_channels=in_channels, out_channels = out_channels, ks = 3, p = 1)

    factor = 2 if up_method.lower() in ["bilinear", "nearest"] else 1

    self.enc_block_1 = DownSampling(out_channels, out_channels * 2)
    self.enc_block_2 = DownSampling(out_channels * 2, out_channels * 4)
    self.enc_block_3 = DownSampling(out_channels * 4, out_channels * 8)
    self.enc_block_4 = DownSampling(out_channels * 8, out_channels * 16 // factor)

    self.dec_block_1 = Upsampling( ( out_channels * 16), (out_channels * 8 // factor), up_method )
    self.dec_block_2 = Upsampling( ( out_channels * 8), (out_channels * 4 // factor), up_method )
    self.dec_block_3 = Upsampling( ( out_channels * 4), (out_channels * 2 // factor), up_method )
    final_out_channels    = (out_channels // factor) * 2 if up_method.lower() in ["bilinear", "nearest"] else out_channels // factor
    self.dec_block_4 = Upsampling( ( out_channels * 2), final_out_channels, up_method )

    self.classifier = FinalConv(out_channels, num_classes)

  def forward(self, x):

    init_conv = self.init_conv(x)

    enc_1 = self.enc_block_1(init_conv)
    enc_2 = self.enc_block_2(enc_1)
    enc_3 = self.enc_block_3(enc_2)
    enc_4 = self.enc_block_4(enc_3)

    dec_1 = self.dec_block_1(x1 = enc_4, x2 = enc_3)
    dec_2 = self.dec_block_2(x1 = dec_1, x2 = enc_2)
    dec_3 = self.dec_block_3(x1 = dec_2, x2 = enc_1)
    dec_4 = self.dec_block_4(x1 = dec_3, x2 = init_conv)

    out = self.classifier(dec_4)

    return out