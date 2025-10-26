import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_func_2d(norm, ch):
    if norm == 'batch':
        return nn.BatchNorm2d(ch, affine=True, track_running_stats=True)
    elif norm == 'instance':
        return nn.InstanceNorm2d(ch, affine=False, track_running_stats=False)
    elif norm == 'group':
        return nn.GroupNorm(8, ch, affine=True)
    else:
        raise Identity()
            
class UNet3D2DAll(nn.Module):
    def __init__(self, generator_3d, base_ch, norm):
        super().__init__()
        self.generator_3d = generator_3d
        self.branch_2D_ilm_opl = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            get_norm_func_2d(norm, base_ch),
            nn.LeakyReLU()
        )
        self.convout_2D_ilm_opl = nn.Conv2d(base_ch, 1, 3, 1, 1)
        
        self.branch_2D_opl_bm = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            get_norm_func_2d(norm, base_ch),
            nn.LeakyReLU()
        )
        self.convout_2D_opl_bm = nn.Conv2d(base_ch, 1, 3, 1, 1)
        
        self.branch_2D_other = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            get_norm_func_2d(norm, base_ch),
            nn.LeakyReLU()
        )
        self.convout_2D_other = nn.Conv2d(base_ch, 1, 3, 1, 1)
        
    def forward(self, x, seg):
        o3d, decs = self.generator_3d(x)
        
        dec_ilm_opl = self.get_projection(decs[0], seg[:, 1:2])
        dec2d_ilm_opl = self.branch_2D_ilm_opl(dec_ilm_opl)
        o2d_feature_ilm_opl = torch.tanh(self.convout_2D_ilm_opl(dec2d_ilm_opl))
        
        dec_opl_bm = self.get_projection(decs[0], seg[:, 2:3])
        dec2d_opl_bm = self.branch_2D_opl_bm(dec_opl_bm)
        o2d_feature_opl_bm = torch.tanh(self.convout_2D_opl_bm(dec2d_opl_bm))
        
        dec_other = self.get_projection(decs[0], seg[:, 0:1])
        dec2d_other = self.branch_2D_other(dec_other)
        o2d_feature_other = torch.tanh(self.convout_2D_other(dec2d_other))
        
        o3d2d_ilm_opl = self.get_projection(o3d, seg[:, 1:2])
        o3d2d_opl_bm = self.get_projection(o3d, seg[:, 2:3])
        o3d2d_other = self.get_projection(o3d, seg[:, 0:1])
        
        return o3d, o2d_feature_ilm_opl, o2d_feature_opl_bm, o2d_feature_other, o3d2d_ilm_opl, o3d2d_opl_bm, o3d2d_other, [dec2d_ilm_opl], [dec2d_opl_bm], [dec2d_other]
        
    def get_projection(self, x, mask):
        return torch.sum(x * mask, dim=2) / (torch.sum(mask, dim=2) + 1e-6)
    
        # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init3d(self._modules[m], mean, std)
    
class UNet3DMean(nn.Module):
    def __init__(self, generator_3d, base_ch, norm):
        super().__init__()
        self.generator_3d = generator_3d
        self.branch_2D = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            get_norm_func_2d(norm, base_ch),
            nn.LeakyReLU()
        )
        self.convout_2D = nn.Conv2d(base_ch, 1, 3, 1, 1)
        
    def forward(self, x, seg):
        o3d, decs = self.generator_3d(x)
        
        dec_mean = torch.mean(decs[0], dim=2)
        dec2d_mean = self.branch_2D(dec_mean)
        o2d_feature_mean = torch.tanh(self.convout_2D(dec2d_mean))
        
        o3d2d_mean = torch.mean(o3d, dim=2)
        
        return o3d, o2d_feature_mean, o3d2d_mean, [dec2d_mean]
    
        # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init3d(self._modules[m], mean, std)
    

        
def normal_init3d(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        