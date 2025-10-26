import sys
sys.path.append('/home/khan7/workspace/OCTA_GAN')

import os
from data.base_dataset3d import BaseDataset
from data.base_dataset import get_params as get_params_2D
from data.base_dataset import get_transform as get_transform_2D

from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv


class AlignedOCT2OCTA2DDataset(BaseDataset):
    """A dataset class for paired image dataset.
    
    OCT to OCTA 2D
    
    We encourage the user to load the normalized projection maps from disk by specifying "--norm_proj". The normalized projection maps are calculated and normalized by min-max normalization and multiplied by 255.
    Inside the dataset __getitem__, projection maps are normalized by 255 to [-1, 1].
    When loading the projection maps, we first load the entire 2D image and normalize the intensity by 255 and then do the cropping. By doing this, we make sure the intensity distrition is the same for all the cropped patches of same projection.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--norm_proj', action='store_true', help='whether load the normalized projection, we should use the normalized projection')
        return parser

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            phase (str)       -- train, val, test, etc; defines the phase of the dataset
        """
        
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.phase = phase
        self.dataroot = opt.dataroot
        
        # load the names of the images in the dataset
        self.load_pids()
        
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        self.load_extra_paths()
        
    def load_pids(self):
        self.pids = []
        with open(os.path.join(self.dataroot, 'datasplit/xoct', f'{self.phase}_pids.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.pids.append(row[0])
    
    def load_extra_paths(self):
        base_dir = os.path.dirname(self.dataroot)
        
        if self.opt.norm_proj:
            norm = True # use the min-max normalized data, the range is [0, 255]
        else:
            norm = False    # use the original data, the range is [0, 255]
        
        # load the projection map paths
        if '3M' in self.dataroot:
            A_mean_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/Mean'
            A_full_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/FULL'
            A_ilm_opl_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/ILM_OPL'
            A_opl_bm_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/OPL_BM'
            B_mean_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/Mean'
            B_full_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/FULL'
            B_ilm_opl_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/ILM_OPL'
            B_opl_bm_proj_dir_path = f'{base_dir}/OCT2OCTA3M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/OPL_BM'
        elif '6M' in self.dataroot:
            A_mean_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/Mean'
            A_full_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/FULL'
            A_ilm_opl_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/ILM_OPL'
            A_opl_bm_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCT/norm_{norm}/OPL_BM'
            B_mean_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/Mean'
            B_full_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/FULL'
            B_ilm_opl_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/ILM_OPL'
            B_opl_bm_proj_dir_path = f'{base_dir}/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCTA/norm_{norm}/OPL_BM'
        else:
            raise NotImplementedError
        
        # create path lists
        self.A_Mean_Proj_paths = []
        self.A_FULL_Proj_paths = []
        self.A_ILM_OPL_Proj_paths = []
        self.A_OPL_BM_Proj_paths = []
        self.B_Mean_Proj_paths = []
        self.B_FULL_Proj_paths = []
        self.B_ILM_OPL_Proj_paths = []
        self.B_OPL_BM_Proj_paths = []
        
        # Add the paths to the lists
        for p in self.pids:
            id = p
            
            A_mean_proj_path = os.path.join(A_mean_proj_dir_path, id + '.bmp')
            A_full_proj_path = os.path.join(A_full_proj_dir_path, id + '.bmp')
            A_ilm_opl_proj_path = os.path.join(A_ilm_opl_proj_dir_path, id + '.bmp')
            A_opl_bm_proj_path = os.path.join(A_opl_bm_proj_dir_path, id + '.bmp')
            
            B_mean_proj_path = os.path.join(B_mean_proj_dir_path, id + '.bmp')
            B_full_proj_path = os.path.join(B_full_proj_dir_path, id + '.bmp')
            B_ilm_opl_proj_path = os.path.join(B_ilm_opl_proj_dir_path, id + '.bmp')
            B_opl_bm_proj_path = os.path.join(B_opl_bm_proj_dir_path, id + '.bmp')
            
            # Check if the paths exist
            assert os.path.exists(A_mean_proj_path)
            assert os.path.exists(A_full_proj_path)
            assert os.path.exists(A_ilm_opl_proj_path)
            assert os.path.exists(A_opl_bm_proj_path)
            assert os.path.exists(B_mean_proj_path)
            assert os.path.exists(B_full_proj_path)
            assert os.path.exists(B_ilm_opl_proj_path)
            assert os.path.exists(B_opl_bm_proj_path)
            
            self.A_Mean_Proj_paths.append(A_mean_proj_path)
            self.A_FULL_Proj_paths.append(A_full_proj_path)
            self.A_ILM_OPL_Proj_paths.append(A_ilm_opl_proj_path)
            self.A_OPL_BM_Proj_paths.append(A_opl_bm_proj_path)
            self.B_Mean_Proj_paths.append(B_mean_proj_path)
            self.B_FULL_Proj_paths.append(B_full_proj_path)
            self.B_ILM_OPL_Proj_paths.append(B_ilm_opl_proj_path)
            self.B_OPL_BM_Proj_paths.append(B_opl_bm_proj_path)
        
    def __getitem__(self, index):
        """Return a data point
        """
        # load projection and normalize it by 255 to [-1, 1]
        A_Mean_Proj = norm(np.array(Image.open(self.A_Mean_Proj_paths[index])))[None]
        B_Mean_Proj = norm(np.array(Image.open(self.B_Mean_Proj_paths[index])))[None]
        A_FULL_Proj = norm(np.array(Image.open(self.A_FULL_Proj_paths[index])))[None]
        B_FULL_Proj = norm(np.array(Image.open(self.B_FULL_Proj_paths[index])))[None]
        A_ILM_OPL_Proj = norm(np.array(Image.open(self.A_ILM_OPL_Proj_paths[index])))[None]
        B_ILM_OPL_Proj = norm(np.array(Image.open(self.B_ILM_OPL_Proj_paths[index])))[None]
        A_OPL_BM_Proj = norm(np.array(Image.open(self.A_OPL_BM_Proj_paths[index])))[None]
        B_OPL_BM_Proj = norm(np.array(Image.open(self.B_OPL_BM_Proj_paths[index])))[None]
        
        # [1, H, W] tensor
        A_Mean_Proj, A_FULL_Proj, A_ILM_OPL_Proj, A_OPL_BM_Proj = torch.from_numpy(A_Mean_Proj), torch.from_numpy(A_FULL_Proj), torch.from_numpy(A_ILM_OPL_Proj), torch.from_numpy(A_OPL_BM_Proj)
        B_Mean_Proj, B_FULL_Proj, B_ILM_OPL_Proj, B_OPL_BM_Proj = torch.from_numpy(B_Mean_Proj), torch.from_numpy(B_FULL_Proj), torch.from_numpy(B_ILM_OPL_Proj), torch.from_numpy(B_OPL_BM_Proj)
        
        
        if self.opt.preprocess != 'none':
            transform_params_2d = get_params_2D(self.opt, A_FULL_Proj.shape, train=(self.phase=='train'))
            transform_2D = get_transform_2D(self.opt, params=transform_params_2d)
            
            A_Mean_Proj = transform_2D(A_Mean_Proj)
            B_Mean_Proj = transform_2D(B_Mean_Proj)
            A_FULL_Proj = transform_2D(A_FULL_Proj)
            B_FULL_Proj = transform_2D(B_FULL_Proj)
            
            A_ILM_OPL_Proj = transform_2D(A_ILM_OPL_Proj)
            B_ILM_OPL_Proj = transform_2D(B_ILM_OPL_Proj)
            A_OPL_BM_Proj = transform_2D(A_OPL_BM_Proj)
            B_OPL_BM_Proj = transform_2D(B_OPL_BM_Proj)
            
        data = {
                'pids': self.pids[index],
                'A_Mean_Proj': A_Mean_Proj.float(), 'B_Mean_Proj': B_Mean_Proj.float(),
                'A_FULL_Proj': A_FULL_Proj.float(), 'B_FULL_Proj': B_FULL_Proj.float(),
                'A_ILM_OPL_Proj': A_ILM_OPL_Proj.float(), 'B_ILM_OPL_Proj': B_ILM_OPL_Proj.float(),
                'A_OPL_BM_Proj': A_OPL_BM_Proj.float(), 'B_OPL_BM_Proj': B_OPL_BM_Proj.float()
                }
        
        return data
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.pids)
    
def norm(x):
    x = x / 255
    x = 2 * x - 1
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocess', default='crop')
    parser.add_argument('--crop_size', default=96)
    parser.add_argument('--no_flip', action='store_true')
    parser.add_argument('--dataroot', default='/home/khan7/workspace/OCTA_GAN/octa-500/OCT2OCTA3M_3D')
    parser.add_argument('--max_dataset_size', default=10000)
    parser.add_argument('--direction', default='AtoB')
    parser.add_argument('--input_nc', default=1)
    parser.add_argument('--output_nc', default=1)
    parser.add_argument('--crop_depth', default=False)
    parser.add_argument('--preserve_layers', default=False)
    args = parser.parse_args()
    
    print(args)
    
    dataset = AlignedOCT2OCTA2DDataset(args, 'train')
    for data in tqdm(dataset):
        pass
    
