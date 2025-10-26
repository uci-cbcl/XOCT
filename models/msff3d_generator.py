# Multi-Scale Feature Fusion

import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_func(norm, ch):
    if norm == 'batch':
        return nn.BatchNorm3d(ch, affine=True, track_running_stats=True)
    elif norm == 'instance':
        return nn.InstanceNorm3d(ch, affine=False, track_running_stats=False)
    elif norm == 'group':
        return nn.GroupNorm(8, ch, affine=True)
    else:
        raise Identity()

class CAM(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_attn = nn.Sequential(
            nn.Conv3d(dim, dim//ratio, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim//ratio, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_att = self.conv_attn(self.avg_pool(x))
        output = x * ch_att
        
        return output

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1, norm='group', act_out=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=False)
        self.norm   = get_norm_func(norm, out_channels)
        self.act_out = act_out
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        
        if self.act_out:
            out = F.leaky_relu(out, 0.2, inplace=True)

        return out

class ConvResidualBlock3D(nn.Module):
    """
    A basic 3D Residual Block with two 3D convolutions, batch normalization, and a skip connection.
    If in_channels != out_channels, a 1x1x1 convolution adjusts the shortcut's channel size.
    """
    def __init__(self, in_channels, out_channels, groups=1, norm='group', force_skip=False, scale=1.):
        super().__init__()
        self.scale = scale
        self.conv1 = ConvBlock3D(in_channels, out_channels, groups=groups, norm=norm, act_out=False)
         
        # Shortcut if channels differ
        self.shortcut = None
        if in_channels != out_channels or force_skip:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
    
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity * self.scale
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class ConvBlock3DMSFF(nn.Module):
    def __init__(self, in_channels, out_channels, norm, act_out=True):
        super().__init__()
        self.conv333 = ConvBlock3D(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(3, 3, 3), padding=(1, 1, 1), norm=norm)
        self.conv311 = ConvBlock3D(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(3, 1, 1), padding=(1, 0, 0), norm=norm)
        self.conv131 = ConvBlock3D(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(1, 3, 1), padding=(0, 1, 0), norm=norm)
        self.conv113 = ConvBlock3D(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(1, 1, 3), padding=(0, 0, 1), norm=norm)
        self.convdw555 = ConvBlock3D(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(5, 5, 5), padding=(2, 2, 2), groups=out_channels//2, norm=norm)

        self.conv111 = ConvBlock3D(in_channels=out_channels//2 * 5, out_channels=out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), groups=1, norm=norm, act_out=act_out)
        
    def forward(self, x):
        
        feat_333 = self.conv333(x)
        feat_311 = self.conv311(x)
        feat_131 = self.conv131(x)
        feat_113 = self.conv113(x)
        feat_555 = self.convdw555(x)
        
        out = self.conv111(torch.cat([feat_333, feat_311, feat_131, feat_113, feat_555], dim=1))
        
        return out

class ConvBlock3DMSFFCAM(nn.Module):
    """
    A basic 3D Residual Block with two 3D convolutions, batch normalization, and a skip connection.
    If in_channels != out_channels, a 1x1x1 convolution adjusts the shortcut's channel size.
    """
    def __init__(self, in_channels, out_channels, norm, force_skip=False, scale=1.):
        super().__init__()
        self.scale = scale
        self.conv1 = ConvBlock3DMSFF(in_channels, out_channels, norm, act_out=False)
        self.cam = CAM(out_channels)
        
        # Shortcut if channels differ
        self.shortcut = None
        if in_channels != out_channels or force_skip:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.cam(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity * self.scale
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out

class ConvDownBlock3DMSFFCAM(nn.Module):
    """
    Down-sampling block: MaxPool3d -> ConvBlock3D
    """
    def __init__(self, in_channels, out_channels, norm, force_skip=False, scale=1.):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res_block = ConvBlock3DMSFFCAM(in_channels, out_channels, norm, force_skip, scale)

    def forward(self, x):
        x = self.pool(x)
        x = self.res_block(x)
        return x

class ConvUpBlock3DMSFFCAM(nn.Module):
    """
    Up-sampling block: ConvTranspose3d -> Concatenate skip -> ConvBlock3D
    """
    def __init__(self, in_channels, out_channels, norm, force_skip=False, scale=1.):
        super().__init__()
        self.up_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation with the skip, the channel size will be 2*out_channels
        self.res_block = ConvBlock3DMSFFCAM(out_channels*2, out_channels, norm, force_skip, scale)

    def forward(self, x, skip):
        x = self.up_transpose(x)
        # Concatenate skip connection along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

class UNet3DMSFF(nn.Module):
    """
    A 6-layer 3D U-Net with Residual Blocks.
    
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
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, max_ch=256, norm='batch', force_skip=False, add_scale=False):
        super().__init__()

        # --- Encoder ---
        # stem
        self.stem_enc = ConvResidualBlock3D(in_channels, base_ch, 1, norm, force_skip=False, scale=1.)
        # Level 1
        self.enc1 = ConvBlock3DMSFFCAM(base_ch, base_ch, norm, force_skip, scale=1.)
        # Level 2
        ch2 = min(base_ch * 2, max_ch)
        self.down1 = ConvDownBlock3DMSFFCAM(base_ch, ch2, norm, force_skip, scale=0.8 if add_scale else 1.)
        # Level 3
        ch3 = min(base_ch * 4, max_ch)
        self.down2 = ConvDownBlock3DMSFFCAM(ch2, ch3, norm, force_skip, scale=0.6 if add_scale else 1.)
        # Level 4
        ch4 = min(base_ch * 8, max_ch)
        self.down3 = ConvDownBlock3DMSFFCAM(ch3, ch4, norm, force_skip, scale=0.4 if add_scale else 1.)
        # Level 5
        ch5 = min(base_ch * 8, max_ch)  # We'll stick to max_ch if we already reached it
        self.down4 = ConvDownBlock3DMSFFCAM(ch4, ch5, norm, force_skip, scale=0.2 if add_scale else 1.)
        # Level 6
        ch6 = min(base_ch * 16, max_ch)  # We'll stick to max_ch if we already reached it
        self.down5 = ConvDownBlock3DMSFFCAM(ch5, ch6, norm, force_skip, scale=0. if add_scale else 1.)

        # Bottleneck
        self.bottleneck = ConvBlock3DMSFFCAM(ch6, ch6, norm, force_skip, scale=0. if add_scale else 1.)

        # --- Decoder ---
        # Up5
        self.up5 = ConvUpBlock3DMSFFCAM(ch6, ch5, norm, force_skip, scale=0.2 if add_scale else 1.)
        # Up4
        self.up4 = ConvUpBlock3DMSFFCAM(ch5, ch4, norm, force_skip, scale=0.4 if add_scale else 1.)
        # Up3
        self.up3 = ConvUpBlock3DMSFFCAM(ch4, ch3, norm, force_skip, scale=0.6 if add_scale else 1.)
        # Up2
        self.up2 = ConvUpBlock3DMSFFCAM(ch3, ch2, norm, force_skip, scale=0.8 if add_scale else 1.)
        # Up1
        self.up1 = ConvUpBlock3DMSFFCAM(ch2, base_ch, norm, force_skip, scale=1.)

        # Final 1x1 convolution
        self.stem_dec = ConvResidualBlock3D(base_ch, base_ch, 1, norm, force_skip=False, scale=1.)
        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init3d(self._modules[m], mean, std)

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

def normal_init3d(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    

if __name__ == '__main__':
    # m = DFF(32)
    # x = torch.rand([1, 32, 16, 16, 16])
    # skip = torch.rand([1, 32, 16, 16, 16])
    # out = m(x, skip)
    # print(out.shape)
    
    # m = ConvBlock3DFF(16, 32, norm='group')
    # x = torch.rand([1, 16, 16, 16, 16])
    
    # out = m(x)
    # print(out.shape)
    
    # m = UNet3DFF_Conv_6Layer(1, 1, norm='group')
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    # m = UNet3DDFF_Conv_6Layer(1, 1, norm='group')
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    # m = UNet3DMSFF_Conv_6Layer(1, 1, norm='group')
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    # m = UNet3DMSDFF_Conv_6Layer(1, 1, norm='group')
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    # m = UNet3DMSFFStemCAM_Conv_6Layer(1, 1, norm='group')
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    # m = UNet3DMSFF_Res_6Layer(1, 1, norm='group')
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    # m = UNet3DFFStem_Res_6Layer(1, 1, norm='group', double_conv=False, force_skip=True, add_scale=True, add_CAM=True, dw_conv=False)
    # print(sum(p.numel() for p in m.parameters()))
    # x = torch.rand([1, 1, 96, 96, 96])
    
    # out, _ = m(x)
    # print(out.shape)
    
    m = UNet3DMSFF(1, 1, norm='group', force_skip=True, add_scale=True)
    print(sum(p.numel() for p in m.parameters()))
    x = torch.rand([1, 1, 96, 96, 96])
    
    out, _ = m(x)
    print(out.shape)