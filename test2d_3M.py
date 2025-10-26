import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
import torch
import numpy as np
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

dataset = create_dataset(opt, phase='test')
model = create_model(opt)
model.setup(opt) 

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:{}'.format(opt.gpu_ids[0]))
web_dir = os.path.join(opt.results_dir, opt.test_name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

save_dir_dic = {}
for dir_name in model.get_output_attrs_2d():
    save_dir_dic[dir_name] = os.path.join(web_dir, dir_name)
    os.makedirs(save_dir_dic[dir_name], exist_ok=True)

num_patches = [2, 2]
patch_size = [256, 256]
    
# test
for i, data in enumerate(dataset):
    # during the test, the data is loaded with original resolution
    if i >= opt.num_test:
        break
    
    model.sliding_window_aggregation(data, num_patches, patch_size)
    for attr_2d in model.get_output_attrs_2d():
        out_2d = getattr(model, attr_2d)
        out_2d = out_2d.detach().cpu().numpy()
        out_2d = ((out_2d + 1) / 2.0).clip(0, 1) * 255
        for idx in range(opt.batch_size):
            out_2d_img = Image.fromarray(out_2d[idx][0].astype(np.uint8))
            out_2d_img.save(os.path.join(save_dir_dic[attr_2d], data['pids'][idx] + '.bmp'))
webpage.save()
