import torch
from collections import OrderedDict
import util.util3d as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d
from . import networks_2g_st
from . import networks
from models.unet2d_generator import UNet2D
from models.hybrid_generator import UNet3DMean
from models.unet3d_generator import UNet3D
import torch.nn.functional as F
from eval3d import cal_SSIM
from unet import UNet


class TransProModel(BaseModel):
    
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--base_ch_g', default=64, type=int, help='base channel number for generator') 
        parser.add_argument('--max_ch_g', default=512, type=int, help='maximum channel number for generator') 
        
        parser.add_argument('--lambda_L1_3D', default=10, type=float, help='weight for the 3D L1 loss') 
        parser.add_argument('--lambda_L1_2D', default=10, type=float, help='weight for the 2D L1 loss') 
        parser.add_argument('--lambda_Extra_2D', default=5, type=float, help='weight for the 2D L1 loss') 
        parser.add_argument('--lambda_Perceptual_2D', default=0, type=float, help='weight for the 2D Perceptual loss')
        
        parser.add_argument('--hcg_path', type=str, help='pretrained hcp model path') 
        parser.add_argument('--vpg_path', type=str, help='pretrained vpg model path') 


        return parser
    
    def name(self):
        return 'TransProModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain

        # define 3D Generator
        gen3d = UNet3D(in_channels=1, out_channels=1, base_ch=opt.base_ch_g, max_ch=opt.max_ch_g, norm=opt.norm)
        self.netG = UNet3DMean(gen3d, base_ch=opt.base_ch_g, norm=opt.norm)
        self.netG.weight_init(mean=0.0, std=0.02)
        self.netG.to(device=self.device)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # define 3D Discriminator and 2D Discriminator
            self.netD_3d = networks3d.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_sigmoid)
            self.netD_mean = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            # load 2D Generator for HCG module
            self.netG_t = UNet2D(in_channels=1, out_channels=1, base_ch=64, max_ch=512, norm='group')
            self.netG_t.load_state_dict(torch.load(opt.hcg_path, map_location=self.device)) 
            self.netG_t.to(device=self.device)
            for p in self.netG_t.parameters():
                p.requires_grad=False
            # load 2D segmentation model for VPG module
            self.net = UNet(n_channels=1, n_classes=2, bilinear=False)
            self.net.load_state_dict(torch.load(opt.vpg_path, map_location=self.device)) 
            self.net.to(device=self.device)
            for p in self.net.parameters():
                p.requires_grad=False
            
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD_3d, 'D_3d', opt.which_epoch)
                self.load_network(self.netD_mean, 'D_mean', opt.which_epoch)
        
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks_2g_st.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPerceptual = networks_2g_st.VGGPerceptualLoss(layers=["relu3_3"], requires_grad=False).to(self.device)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam([
                {'params': self.netD_3d.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)},
                {'params': self.netD_mean.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)}
            ])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
        print('---------- Networks initialized -------------')
        networks3d.print_network(self.netG)
        if self.isTrain:
            networks3d.print_network(self.netD_3d)
            networks3d.print_network(self.netD_mean)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device,dtype=torch.float) #torch.Size([1, 1, 256, 256, 256])
        self.real_B = input['B' if AtoB else 'A'].to(self.device,dtype=torch.float)
        self.seg = input['seg'].to(self.device,dtype=torch.float)
        self.real_A_mean = torch.mean(self.real_A, dim=2)
        self.real_A_mean = self.Norm(self.real_A_mean)
        self.real_B_mean = torch.mean(self.real_B, dim=2)
        self.real_B_mean = self.Norm(self.real_B_mean)
        
        # calculate the projection maps from the volume
        # real B
        self.real_B_ilm_opl = torch.sum(self.real_B * self.seg[:, 1:2], dim=2) / (torch.sum(self.seg[:, 1:2], dim=2) + 1e-6)
        self.real_B_opl_bm = torch.sum(self.real_B * self.seg[:, 2:3], dim=2) / (torch.sum(self.seg[:, 2:3], dim=2) + 1e-6)
        
    def forward(self):
        self.fake_B, _, self.fake_B_3d2d_mean, _ = self.netG.forward(self.real_A, self.seg) # torch.Size([1, 1, 256, 256, 256])
        self.fake_B_mean_hcg, _ = self.netG_t.forward(self.real_A_mean)
        self.fake_B_mean = self.Norm(self.fake_B_3d2d_mean)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            # calculate the projection maps from the volume
            # fake B
            self.fake_B, _, self.fake_B_3d2d_mean, _ = self.netG.forward(self.real_A, self.seg) # torch.Size([1, 1, 256, 256, 256])
            self.fake_B_3d2d_ilm_opl = torch.sum(self.fake_B * self.seg[:, 1:2], dim=2) / (torch.sum(self.seg[:, 1:2], dim=2) + 1e-6)
            self.fake_B_3d2d_opl_bm = torch.sum(self.fake_B * self.seg[:, 2:3], dim=2) / (torch.sum(self.seg[:, 2:3], dim=2) + 1e-6)
            
            self.ssim_ilm_opl = cal_SSIM(self.real_B_ilm_opl[0][0].cpu().numpy(), self.fake_B_3d2d_ilm_opl[0][0].cpu().numpy(), norm=True)
            self.ssim_opl_bm = cal_SSIM(self.real_B_opl_bm[0][0].cpu().numpy(), self.fake_B_3d2d_opl_bm[0][0].cpu().numpy(), norm=True)
            self.ssim_mean = cal_SSIM(self.real_B[0][0].mean(dim=0).cpu().numpy(), self.fake_B[0][0].mean(dim=0).cpu().numpy(), norm=True)
            self.ssim = (self.ssim_ilm_opl + self.ssim_opl_bm + self.ssim_mean) / 3
           
    def get_output_attrs_3d(self):
        return ['fake_B_3d']

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD_3d.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        
        fake_AB_mean = self.fake_AB_pool.query(torch.cat((self.real_A_mean, self.fake_B_mean), 1))
        self.pred_fake_mean = self.netD_mean.forward(fake_AB_mean.detach())
        self.loss_D_fake_mean = self.criterionGAN(self.pred_fake_mean, False)


        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD_3d.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        
        real_AB_mean = torch.cat((self.real_A_mean, self.real_B_mean), 1)
        self.pred_real_mean = self.netD_mean.forward(real_AB_mean)
        self.loss_D_real_mean = self.criterionGAN(self.pred_real_mean, True)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_fake_mean + self.loss_D_real + self.loss_D_real_mean) * 0.5
    
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD_3d.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # mean
        fake_AB_mean = torch.cat((self.real_A_mean, self.fake_B_mean), 1)
        pred_fake_mean = self.netD_mean.forward(fake_AB_mean)
        self.loss_G_GAN_mean = self.criterionGAN(pred_fake_mean, True)
        
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1_3D
        self.loss_G_L1_mean = self.criterionL1(self.fake_B_mean, self.real_B_mean) * self.opt.lambda_L1_2D
        
        self.fake_B_seg = F.softmax(self.net((self.fake_B_mean+1)/2), dim=1)[0] # 
        self.real_B_seg = F.softmax(self.net((self.real_B_mean+1)/2), dim=1)[0]

        self.loss_G_L1_pm_st = self.criterionL1(self.fake_B_mean, self.fake_B_mean_hcg) * self.opt.lambda_Extra_2D
        self.loss_G_L1_seg = self.criterionL1(self.fake_B_seg, self.real_B_seg) * self.opt.lambda_Extra_2D
        
        self.loss_G_Perceptual_mean = self.criterionPerceptual.perceptual_loss(self.fake_B_mean, self.real_B_mean) * self.opt.lambda_Perceptual_2D        
        
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_mean + \
                      self.loss_G_L1 + self.loss_G_L1_mean + \
                      self.loss_G_L1_pm_st + self.loss_G_L1_seg + \
                      self.loss_G_Perceptual_mean

        self.loss_G.backward()

    def optimize_parameters(self):
        # Optimize the discriminator
        self.forward()
        self.set_requires_grad(self.netD_3d, True)
        self.set_requires_grad(self.netD_mean, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Optimize the generator
        self.set_requires_grad(self.netD_3d, False) 
        self.set_requires_grad(self.netD_mean, False) 
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_GAN_mean', self.loss_G_GAN_mean.item()),
                            
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_L1_mean', self.loss_G_L1_mean.item()),
                            
                            ('G_L1_pm_st', self.loss_G_L1_pm_st.item()),
                            ('G_L1_seg', self.loss_G_L1_seg.item()),
                            ('G_Perceptual_mean', self.loss_G_Perceptual_mean.item()),
                            
                            ('D_real', self.loss_D_real.item()),
                            ('D_real_mean', self.loss_D_real_mean.item()),
                            
                            ('D_fake', self.loss_D_fake.item()),
                            ('D_fake_mean', self.loss_D_fake_mean.item()),
                            ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_3d, 'D_3d', label, self.gpu_ids)
        self.save_network(self.netD_mean, 'D_mean', label, self.gpu_ids)
        
    def generate_sliding_window(self, input_size, num_patches, patch_size):
        assert len(input_size) == 2 and len(num_patches) == 2 and len(patch_size) == 2
        
        index_list = []
        for s, n, p in zip(input_size, num_patches, patch_size):
            assert n * p > s
            start_indexs = torch.linspace(0, s-p, n).to(torch.long)
            end_indexs = start_indexs + p
            start_indexs = start_indexs.tolist()
            end_indexs = end_indexs.tolist()
            index_list.append([start_indexs, end_indexs])
            
        return index_list
    
    def sliding_window_aggregation(self, data, num_patches, patch_size):
        
        N, C, D, H, W = data['A'].shape
        size_2d = [H, W]
        index_list = self.generate_sliding_window(size_2d, num_patches, patch_size)
        
        output_aggregator_3d = torch.zeros_like(data['A']).to(self.device,dtype=torch.float)
        count_map_3d = torch.zeros_like(output_aggregator_3d)
        
        with torch.no_grad():
            for s1, e1 in zip(*index_list[0]):
                for s2, e2 in zip(*index_list[1]):
                    real_A = data['A'][..., s1:e1, s2:e2].to(self.device,dtype=torch.float)
                    seg = data['seg'][..., s1:e1, s2:e2].to(self.device,dtype=torch.float)
                    fake_B, _, _, _ = self.netG.forward(real_A, seg) # torch.Size([1, 1, 256, 256, 256])
                    output_aggregator_3d[..., s1:e1, s2:e2] += fake_B
                    count_map_3d[..., s1:e1, s2:e2] += 1.

            self.fake_B_3d = output_aggregator_3d / (count_map_3d + 1e-6)
            
    def Norm(self, a):
        max_ = torch.max(a)
        min_ = torch.min(a)
        a_0_1 = (a-min_)/(max_-min_)
        return (a_0_1-0.5)*2

