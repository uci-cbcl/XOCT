import sys
sys.path.append('/home/khan7/workspace/OCTA_GAN')

import os
from data.base_dataset3d import BaseDataset
from data.base_dataset3d import get_params as get_params_3D
from data.base_dataset3d import get_transform as get_transform_3D

from data.image_folder import make_dataset
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv

class AlignedOCT2OCTAALLDataset(BaseDataset):
    """A dataset class for paired image dataset.

    OCT to OCTA 3D.
    
    We load the resized 3D OCT and OCTA volumes from the disk. (3M-256x304x304; 6M-256x400x400) But the intensity values are not processed.
    When loading the images, we first load the entire 3D volume and normalize the intensity by 255 and then do the cropping. By doing this, we make sure the intensity distrition is the same for all the cropped patches of same 3D volume.

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
        return parser


    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
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
        dir_A_path = os.path.join(self.dataroot, 'A')
        dir_B_path = os.path.join(self.dataroot, 'B')
        seg_dir_path = f'{base_dir}/Label/GT_Layers_npy'
        
        # create the segmentation list
        self.A_3D_paths = []
        self.B_3D_paths = []
        self.seg_paths = []
        
        # Add the paths to the lists
        for p in self.pids:
            A_3D_path = os.path.join(dir_A_path, p + '.npy')
            B_3D_path = os.path.join(dir_B_path, p + '.npy')
            seg_path = os.path.join(seg_dir_path, p + '.npy')
            
            assert os.path.exists(A_3D_path)
            assert os.path.exists(B_3D_path)
            assert os.path.exists(seg_path)
            
            self.A_3D_paths.append(A_3D_path)
            self.B_3D_paths.append(B_3D_path)
            self.seg_paths.append(seg_path)
            
    def __getitem__(self, index):
        """Return a data point
        """
        # load 3D image
        A_3D = norm(load_img_3d(self.A_3D_paths[index]))
        B_3D = norm(load_img_3d(self.B_3D_paths[index]))
        # load 3D layer segmentation
        seg_3D = load_img_3d(self.seg_paths[index])
        
        if self.opt.preprocess != 'none':
            # apply the same transform to both A and B
            transform_params_3d = get_params_3D(self.opt, A_3D.shape, train=(self.phase=='train'))   # donot crop the depth dimension. could crop the h, w
            transform_3D = get_transform_3D(self.opt, transform_params_3d)
            
            A_3D = transform_3D(A_3D)
            B_3D = transform_3D(B_3D)
            seg_3D = transform_3D(seg_3D)
            
            
        A_3D = torch.from_numpy(A_3D)
        B_3D = torch.from_numpy(B_3D)
        seg_3D = torch.from_numpy(seg_3D)
        seg_3D_one_hot = torch.cat([seg_3D==0, seg_3D==1, seg_3D==2], dim=0).to(torch.uint8)
        
        data = {'A': A_3D.float(), 'B': B_3D.float(), 'seg': seg_3D_one_hot.float(),
                'pids': self.pids[index]
                }
        
        return data
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.pids)
    
def norm(x):
    x = x / 255
    x = 2 * x - 1
    return x

def load_img_3d(path):
    img = np.load(path).astype(np.float64)  # [256, 304, 304] -> [304, 256, 304]
    img = np.expand_dims(img, axis=0)
    
    return img

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
    
    dataset = AlignedOCT2OCTAALLDataset(args, 'train')
    for data in tqdm(dataset):
        pass
    
