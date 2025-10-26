import torch
from collections import OrderedDict
import util.util3d as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d
from . import networks_2g_st
from . import networks
from models.unet2d_generator import UNet2D
from eval3d import cal_SSIM


class UNet2DModel(BaseModel):
    
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """Add new model-specific options and rewrite default values for existing options.
        
        """
        parser.add_argument('--base_ch_g', default=64, type=int, help='base channel number for generator') 
        parser.add_argument('--max_ch_g', default=512, type=int, help='maximum channel number for generator') 
        
        parser.add_argument('--lambda_L1_2D', default=10, type=float, help='weight for the 2D L1 loss') 
        parser.add_argument('--lambda_Perceptual_2D', default=0, type=float, help='weight for the 2D Perceptual loss')

        return parser
    
    def name(self):
        return 'UNet2DModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain

        # define 2D Generator
        self.netG = UNet2D(in_channels=4, out_channels=4, base_ch=opt.base_ch_g, max_ch=opt.max_ch_g, norm=opt.norm)
        self.netG.weight_init(mean=0.0, std=0.02)
        self.netG.to(device=self.device)
        
        if self.isTrain:
            # define 2D Discriminator
            self.netD_ilm_opl = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_opl_bm = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_full = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_mean = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD_ilm_opl, 'D_ilm_opl', opt.which_epoch)
                self.load_network(self.netD_opl_bm, 'D_opl_bm', opt.which_epoch)
                self.load_network(self.netD_full, 'D_full', opt.which_epoch)
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
                {'params': self.netD_ilm_opl.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)},
                {'params': self.netD_opl_bm.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)},
                {'params': self.netD_full.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)},
                {'params': self.netD_mean.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)}
            ])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
        print('---------- Networks initialized -------------')
        networks3d.print_network(self.netG)
        if self.isTrain:
            networks3d.print_network(self.netD_ilm_opl)
            networks3d.print_network(self.netD_opl_bm)
            networks3d.print_network(self.netD_full)
            networks3d.print_network(self.netD_mean)
        print('-----------------------------------------------')
        

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_ilm_opl = input['A_ILM_OPL_Proj' if AtoB else 'B_ILM_OPL_Proj'].to(self.device,dtype=torch.float) #torch.Size([1, 1, 256, 256, 256])
        self.real_B_ilm_opl = input['B_ILM_OPL_Proj' if AtoB else 'A_ILM_OPL_Proj'].to(self.device,dtype=torch.float)
        self.real_A_opl_bm = input['A_OPL_BM_Proj' if AtoB else 'B_OPL_BM_Proj'].to(self.device,dtype=torch.float) #torch.Size([1, 1, 256, 256, 256])
        self.real_B_opl_bm = input['B_OPL_BM_Proj' if AtoB else 'A_OPL_BM_Proj'].to(self.device,dtype=torch.float)
        self.real_A_full = input['A_FULL_Proj' if AtoB else 'B_FULL_Proj'].to(self.device,dtype=torch.float) #torch.Size([1, 1, 256, 256, 256])
        self.real_B_full = input['B_FULL_Proj' if AtoB else 'A_FULL_Proj'].to(self.device,dtype=torch.float)
        self.real_A_mean = input['A_Mean_Proj' if AtoB else 'B_Mean_Proj'].to(self.device,dtype=torch.float) #torch.Size([1, 1, 256, 256, 256])
        self.real_B_mean = input['B_Mean_Proj' if AtoB else 'A_Mean_Proj'].to(self.device,dtype=torch.float)
        
        self.real_A = torch.cat([self.real_A_ilm_opl, self.real_A_opl_bm, self.real_A_full, self.real_A_mean], dim=1)
        self.real_B = torch.cat([self.real_B_ilm_opl, self.real_B_opl_bm, self.real_B_full, self.real_B_mean], dim=1)

    def forward(self):
        self.fake_B, _ = self.netG.forward(self.real_A) # torch.Size([1, 1, 256, 256, 256])
        self.fake_B_conv_ilm_opl = self.fake_B[:, 0:1]
        self.fake_B_conv_opl_bm = self.fake_B[:, 1:2]
        self.fake_B_conv_full = self.fake_B[:, 2:3]
        self.fake_B_conv_mean = self.fake_B[:, 3:4]

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.fake_B, _ = self.netG.forward(self.real_A)
            self.fake_B_conv_ilm_opl = self.fake_B[:, 0:1]
            self.fake_B_conv_opl_bm = self.fake_B[:, 1:2]
            self.fake_B_conv_full = self.fake_B[:, 2:3]
            self.fake_B_conv_mean = self.fake_B[:, 3:4]
            
            self.ssim_ilm_opl = cal_SSIM(self.real_B_ilm_opl[0][0].cpu().numpy(), self.fake_B_conv_ilm_opl[0][0].cpu().numpy(), norm=True)
            self.ssim_opl_bm = cal_SSIM(self.real_B_opl_bm[0][0].cpu().numpy(), self.fake_B_conv_opl_bm[0][0].cpu().numpy(), norm=True)
            self.ssim_full = cal_SSIM(self.real_B_full[0][0].cpu().numpy(), self.fake_B_conv_full[0][0].cpu().numpy(), norm=True)
            self.ssim_mean = cal_SSIM(self.real_B_mean[0][0].cpu().numpy(), self.fake_B_conv_mean[0][0].cpu().numpy(), norm=True)
            self.ssim = (self.ssim_ilm_opl + self.ssim_opl_bm + self.ssim_full + self.ssim_mean) / 4
            
    def get_output_attrs_2d(self):
        return ['fake_B_conv_ilm_opl', 'fake_B_conv_opl_bm', 'fake_B_conv_full', 'fake_B_conv_mean']

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB_ilm_opl = self.fake_AB_pool.query(torch.cat((self.real_A_ilm_opl, self.fake_B_conv_ilm_opl), 1))
        self.pred_fake_ilm_opl = self.netD_ilm_opl.forward(fake_AB_ilm_opl.detach())
        self.loss_D_fake_ilm_opl = self.criterionGAN(self.pred_fake_ilm_opl, False)

        fake_AB_opl_bm = self.fake_AB_pool.query(torch.cat((self.real_A_opl_bm, self.fake_B_conv_opl_bm), 1))
        self.pred_fake_opl_bm = self.netD_opl_bm.forward(fake_AB_opl_bm.detach())
        self.loss_D_fake_opl_bm = self.criterionGAN(self.pred_fake_opl_bm, False)
        
        fake_AB_full = self.fake_AB_pool.query(torch.cat((self.real_A_full, self.fake_B_conv_full), 1))
        self.pred_fake_full = self.netD_full.forward(fake_AB_full.detach())
        self.loss_D_fake_full = self.criterionGAN(self.pred_fake_full, False)
        
        fake_AB_mean = self.fake_AB_pool.query(torch.cat((self.real_A_mean, self.fake_B_conv_mean), 1))
        self.pred_fake_mean = self.netD_mean.forward(fake_AB_mean.detach())
        self.loss_D_fake_mean = self.criterionGAN(self.pred_fake_mean, False)

        # Real
        real_AB_ilm_opl = torch.cat((self.real_A_ilm_opl, self.real_B_ilm_opl), 1)
        self.pred_real_ilm_opl = self.netD_ilm_opl.forward(real_AB_ilm_opl)
        self.loss_D_real_ilm_opl = self.criterionGAN(self.pred_real_ilm_opl, True)
        
        real_AB_opl_bm = torch.cat((self.real_A_opl_bm, self.real_B_opl_bm), 1)
        self.pred_real_opl_bm = self.netD_opl_bm.forward(real_AB_opl_bm)
        self.loss_D_real_opl_bm = self.criterionGAN(self.pred_real_opl_bm, True)
        
        real_AB_full = torch.cat((self.real_A_full, self.real_B_full), 1)
        self.pred_real_full = self.netD_full.forward(real_AB_full)
        self.loss_D_real_full = self.criterionGAN(self.pred_real_full, True)
        
        real_AB_mean = torch.cat((self.real_A_mean, self.real_B_mean), 1)
        self.pred_real_mean = self.netD_mean.forward(real_AB_mean)
        self.loss_D_real_mean = self.criterionGAN(self.pred_real_mean, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake_ilm_opl + self.loss_D_fake_opl_bm + self.loss_D_fake_full + self.loss_D_fake_mean + self.loss_D_real_ilm_opl + self.loss_D_real_full + self.loss_D_real_mean) * 0.5
    
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB_ilm_opl = torch.cat((self.real_A_ilm_opl, self.fake_B_conv_ilm_opl), 1)
        pred_fake_ilm_opl = self.netD_ilm_opl.forward(fake_AB_ilm_opl)
        self.loss_G_GAN_ilm_opl = self.criterionGAN(pred_fake_ilm_opl, True)
        
        fake_AB_opl_bm = torch.cat((self.real_A_opl_bm, self.fake_B_conv_opl_bm), 1)
        pred_fake_opl_bm = self.netD_opl_bm.forward(fake_AB_opl_bm)
        self.loss_G_GAN_opl_bm = self.criterionGAN(pred_fake_opl_bm, True)
        
        fake_AB_full = torch.cat((self.real_A_full, self.fake_B_conv_full), 1)
        pred_fake_full = self.netD_full.forward(fake_AB_full)
        self.loss_G_GAN_full = self.criterionGAN(pred_fake_full, True)
        
        fake_AB_mean = torch.cat((self.real_A_mean, self.fake_B_conv_mean), 1)
        pred_fake_mean = self.netD_mean.forward(fake_AB_mean)
        self.loss_G_GAN_mean = self.criterionGAN(pred_fake_mean, True)

        # Second, G(A) = B
        self.loss_G_L1_ilm_opl = self.criterionL1(self.fake_B_conv_ilm_opl, self.real_B_ilm_opl) * self.opt.lambda_L1_2D
        self.loss_G_L1_opl_bm = self.criterionL1(self.fake_B_conv_opl_bm, self.real_B_opl_bm) * self.opt.lambda_L1_2D
        self.loss_G_L1_full = self.criterionL1(self.fake_B_conv_full, self.real_B_full) * self.opt.lambda_L1_2D
        self.loss_G_L1_mean = self.criterionL1(self.fake_B_conv_mean, self.real_B_mean) * self.opt.lambda_L1_2D
        
        # Third, Perceptual Feature Loss
        self.loss_G_Perceptual_ilm_opl = self.criterionPerceptual.perceptual_loss(self.fake_B_conv_ilm_opl, self.real_B_ilm_opl) * self.opt.lambda_Perceptual_2D
        self.loss_G_Perceptual_opl_bm = self.criterionPerceptual.perceptual_loss(self.fake_B_conv_opl_bm, self.real_B_opl_bm) * self.opt.lambda_Perceptual_2D
        self.loss_G_Perceptual_full = self.criterionPerceptual.perceptual_loss(self.fake_B_conv_full, self.real_B_full) * self.opt.lambda_Perceptual_2D
        self.loss_G_Perceptual_mean = self.criterionPerceptual.perceptual_loss(self.fake_B_conv_mean, self.real_B_mean) * self.opt.lambda_Perceptual_2D
        
        self.loss_G = self.loss_G_GAN_ilm_opl + self.loss_G_GAN_opl_bm + self.loss_G_GAN_full + self.loss_G_GAN_mean + \
                      self.loss_G_L1_ilm_opl + self.loss_G_L1_opl_bm + self.loss_G_L1_full + self.loss_G_L1_mean + \
                      self.loss_G_Perceptual_ilm_opl + self.loss_G_Perceptual_opl_bm + self.loss_G_Perceptual_full + self.loss_G_Perceptual_mean

        self.loss_G.backward()

    def optimize_parameters(self):
        # Optimize the discriminator
        self.forward()
        self.set_requires_grad(self.netD_ilm_opl, True)
        self.set_requires_grad(self.netD_opl_bm, True)
        self.set_requires_grad(self.netD_full, True)
        self.set_requires_grad(self.netD_mean, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Optimize the generator
        self.set_requires_grad(self.netD_ilm_opl, False) 
        self.set_requires_grad(self.netD_opl_bm, False) 
        self.set_requires_grad(self.netD_full, False) 
        self.set_requires_grad(self.netD_mean, False) 
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        return OrderedDict([('G_GAN_ilm_opl', self.loss_G_GAN_ilm_opl.item()),
                            ('G_GAN_opl_bm', self.loss_G_GAN_opl_bm.item()),
                            ('G_GAN_full', self.loss_G_GAN_full.item()),
                            ('G_GAN_mean', self.loss_G_GAN_mean.item()),
                            
                            ('G_L1_ilm_opl', self.loss_G_L1_ilm_opl.item()),
                            ('G_L1_opl_bm', self.loss_G_L1_opl_bm.item()),
                            ('G_L1_full', self.loss_G_L1_full.item()),
                            ('G_L1_mean', self.loss_G_L1_mean.item()),
                            
                            ('G_Perceptual_ilm_opl', self.loss_G_Perceptual_ilm_opl.item()),
                            ('G_Perceptual_opl_bm', self.loss_G_Perceptual_opl_bm.item()),
                            ('G_Perceptual_full', self.loss_G_Perceptual_full.item()),
                            ('G_Perceptual_mean', self.loss_G_Perceptual_mean.item()),
                            
                            ('D_real_ilm_opl', self.loss_D_real_ilm_opl.item()),
                            ('D_real_opl_bm', self.loss_D_real_opl_bm.item()),
                            ('D_real_full', self.loss_D_real_full.item()),
                            ('D_real_mean', self.loss_D_real_mean.item()),
                            
                            ('D_fake_ilm_opl', self.loss_D_fake_ilm_opl.item()),
                            ('D_fake_opl_bm', self.loss_D_fake_opl_bm.item()),
                            ('D_fake_full', self.loss_D_fake_full.item()),
                            ('D_fake_mean', self.loss_D_fake_mean.item())
                            ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_ilm_opl, 'D_ilm_opl', label, self.gpu_ids)
        self.save_network(self.netD_opl_bm, 'D_opl_bm', label, self.gpu_ids)
        self.save_network(self.netD_full, 'D_full', label, self.gpu_ids)
        self.save_network(self.netD_mean, 'D_mean', label, self.gpu_ids)
        
    # For a single image, we merge multiple patches to generate the final image. This is used to generate the patch indexes
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
    
    # generate the entire image by sliding window
    def sliding_window_aggregation(self, data, num_patches, patch_size):
        
        size_2d = data['A_ILM_OPL_Proj'].shape[-2:]
        index_list = self.generate_sliding_window(size_2d, num_patches, patch_size)
        
        output_aggregator_ilm_opl = torch.zeros_like(data['A_ILM_OPL_Proj']).to(self.device,dtype=torch.float)
        output_aggregator_opl_bm = torch.zeros_like(data['A_OPL_BM_Proj']).to(self.device,dtype=torch.float)
        output_aggregator_full = torch.zeros_like(data['A_FULL_Proj']).to(self.device,dtype=torch.float)
        output_aggregator_mean = torch.zeros_like(data['A_Mean_Proj']).to(self.device,dtype=torch.float)
        count_map = torch.zeros_like(output_aggregator_ilm_opl)
        with torch.no_grad():
            for s1, e1 in zip(*index_list[0]):
                for s2, e2 in zip(*index_list[1]):
                    self.real_A = torch.cat([data['A_ILM_OPL_Proj'][..., s1:e1, s2:e2], data['A_OPL_BM_Proj'][..., s1:e1, s2:e2], data['A_FULL_Proj'][..., s1:e1, s2:e2], data['A_Mean_Proj'][..., s1:e1, s2:e2]], dim=1).to(self.device,dtype=torch.float)
                    fake_B, _ = self.netG.forward(self.real_A)
                    output_aggregator_ilm_opl[..., s1:e1, s2:e2] += fake_B[:, 0:1]
                    output_aggregator_opl_bm[..., s1:e1, s2:e2] += fake_B[:, 1:2]
                    output_aggregator_full[..., s1:e1, s2:e2] += fake_B[:, 2:3]
                    output_aggregator_mean[..., s1:e1, s2:e2] += fake_B[:, 3:4]
                    count_map[..., s1:e1, s2:e2] += 1.
            self.fake_B_conv_ilm_opl = output_aggregator_ilm_opl / (count_map + 1e-6)
            self.fake_B_conv_opl_bm = output_aggregator_opl_bm / (count_map + 1e-6)
            self.fake_B_conv_full = output_aggregator_full / (count_map + 1e-6)
            self.fake_B_conv_mean = output_aggregator_mean / (count_map + 1e-6)