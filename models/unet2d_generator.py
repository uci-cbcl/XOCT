import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_func(norm, ch):
    if norm == 'batch':
        return nn.BatchNorm2d(ch, affine=True, track_running_stats=True)
    elif norm == 'instance':
        return nn.InstanceNorm2d(ch, affine=False, track_running_stats=False)
    elif norm == 'group':
        return nn.GroupNorm(8, ch, affine=True)
    else:
        raise Identity()

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm   = get_norm_func(norm, out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.leaky_relu(out, 0.2, inplace=True)

        return out

class ConvDownBlock2D(nn.Module):
    """
    Down-sampling block: MaxPool2d -> ConvBlock2D
    """
    def __init__(self, in_channels, out_channels, norm):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block = ConvBlock2D(in_channels, out_channels, norm)

    def forward(self, x):
        x = self.pool(x)
        x = self.res_block(x)
        return x
    
class ConvUpBlock2D(nn.Module):
    """
    Up-sampling block: ConvTranspose2d -> Concatenate skip -> ConvBlock2D
    """
    def __init__(self, in_channels, out_channels, norm):
        super().__init__()
        self.up_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation with the skip, the channel size will be 2*out_channels
        self.res_block = ConvBlock2D(out_channels * 2, out_channels, norm)

    def forward(self, x, skip):
        x = self.up_transpose(x)
        # Concatenate skip connection along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

class UNet2D(nn.Module):
    """
    A 6-layer 2d U-Net with Residual Blocks.
    
    Channel progression example (base_ch=32, max_ch=256):
      - Level1: in -> 32
      - Down1: 32 -> 64
      - Down2: 64 -> 128
      - Down3: 128 -> 256
      - Down4: 256 -> 256
      - Down5: 256 -> 256
      - Bottleneck: 256 -> 256
      - Then up-sample symmetrically:
        Up5 -> 256, Up4 -> 256, Up3 -> 128, Up2 -> 64, Up1 -> 32
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, max_ch=256, norm='batch'):
        super().__init__()

        self.stem_enc = ConvBlock2D(in_channels, base_ch, norm)
        # --- Encoder ---
        # Level 1
        self.enc1 = ConvBlock2D(base_ch, base_ch, norm)
        # Level 2
        ch2 = min(base_ch * 2, max_ch)
        self.down1 = ConvDownBlock2D(base_ch, ch2, norm)
        # Level 3
        ch3 = min(base_ch * 4, max_ch)
        self.down2 = ConvDownBlock2D(ch2, ch3, norm)
        # Level 4
        ch4 = min(base_ch * 8, max_ch)
        self.down3 = ConvDownBlock2D(ch3, ch4, norm)
        # Level 5
        ch5 = min(base_ch * 8, max_ch)  # We'll stick to max_ch if we already reached it
        self.down4 = ConvDownBlock2D(ch4, ch5, norm)
        # Level 6
        ch6 = min(base_ch * 16, max_ch)  # We'll stick to max_ch if we already reached it
        self.down5 = ConvDownBlock2D(ch5, ch6, norm)

        # Bottleneck
        self.bottleneck = ConvBlock2D(ch6, ch6, norm)

        # --- Decoder ---
        # Up5
        self.up5 = ConvUpBlock2D(ch6, ch5, norm)
        # Up4
        self.up4 = ConvUpBlock2D(ch5, ch4, norm)
        # Up3
        self.up3 = ConvUpBlock2D(ch4, ch3, norm)
        # Up2
        self.up2 = ConvUpBlock2D(ch3, ch2, norm)
        # Up1
        self.up1 = ConvUpBlock2D(ch2, base_ch, norm)

        self.stem_dec = ConvBlock2D(base_ch, base_ch, norm)
        # Final 1x1 convolution
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init2d(self._modules[m], mean, std)

    def forward(self, x):
        # Encoder
        stem_enc = self.stem_enc(x)
        enc1 = self.enc1(stem_enc)       # level 1 output
        enc2 = self.down1(enc1)   # level 2
        enc3 = self.down2(enc2)   # level 3
        enc4 = self.down3(enc3)   # level 4
        enc5 = self.down4(enc4)   # level 5
        enc6 = self.down5(enc5)   # level 6

        # Bottleneck
        bottleneck = self.bottleneck(enc6)

        # Decoder
        dec5 = self.up5(bottleneck, enc5)
        dec4 = self.up4(dec5, enc4)
        dec3 = self.up3(dec4, enc3)
        dec2 = self.up2(dec3, enc2)
        dec1 = self.up1(dec2, enc1)

        stem_dec = self.stem_dec(dec1)
        out = self.out_conv(stem_dec)
        out = torch.tanh(out)
        return out, [dec1]

def normal_init2d(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__ == '__main__':
    m = UNet2D_Conv_6Layer(1, 1, norm='group')
    print(sum(p.numel() for p in m.parameters()))
    x = torch.rand([1, 1, 96, 96])
    
    out, _ = m(x)
    print(out.shape)