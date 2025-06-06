import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from .custom_folder import PcPreprocessor3DSlim, flip_pseudo3D, rotate_pseudo3D, point_cloud_normal_resize, scale_trans
from utils.logger import *


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class SWIShapeNet(ShapeNet):

    def __init__(self, config):
        super(SWIShapeNet, self).__init__(config)
        self.npoints = config.npoints
        if hasattr(config, "rectify_pos"):
            self.rectify_pos = config.rectify_pos
        else:
            self.rectify_pos = False
        if hasattr(config, "pos3d"):
            self.pos3d = config.pos3d
        else:
            self.pos3d = False
        if hasattr(config, 'space_size'):
            self.space_size = config.space_size
        else:
            self.space_size = 224
        if hasattr(config, "patch_num"):
            patch_num = config.patch_num
        else:
            patch_num = (self.space_size//config.patch_size)**2
        if hasattr(config, 'space_type'):
            self.space_type = config.space_type
        else:
            self.space_type = 'canonical'
        self.preprocess = PcPreprocessor3DSlim(scales=[1], space_size=self.space_size, patch_size=config.patch_size, patch_num=patch_num, rectify_pos=self.rectify_pos, pos3d=self.pos3d, space_type=self.space_type)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = torch.from_numpy(pc).float()
        pc = point_cloud_normal_resize(pc, space_type=self.space_type)
        info = self.preprocess(pc)
        info['target'] = torch.tensor([0])
        return info