from torchvision.datasets.vision import VisionDataset
import torchvision.datasets as datasets

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_scatter import scatter
import torch.nn as nn
import warnings
from .build import DATASETS
import glob
import pickle
from random import sample
from torchvision import transforms
warnings.filterwarnings('ignore')


def flip_it(pc_):
    pc_mean = torch.mean(pc_)
    pc_ = -(pc_-pc_mean)+pc_mean
    return pc_
    

def flip_pseudo3D(pc, plane):
    flag = torch.rand(1)
    if flag<0.5:
        if plane=='XY':
            pc[:, 2] = flip_it(pc[:, 2])
        elif plane=='XZ':
            pc[:, 1] = flip_it(pc[:, 1])
        elif plane=='YZ':
            pc[:, 0] = flip_it(pc[:, 0])
    return pc


def rotate_pseudo3D(pc, low_radius, high_radius, axis, requires_inv=False):
    radi = torch.rand(1)*(high_radius-low_radius) + low_radius
    xyzArray = {
            'X': torch.tensor([[1, 0, 0],
                    [0, torch.cos(radi), -torch.sin(radi)],
                    [0, torch.sin(radi), torch.cos(radi)]], dtype=torch.float32),
            'Y': torch.tensor([[torch.cos(radi), 0, torch.sin(radi)],
                    [0, 1, 0],
                    [-torch.sin(radi), 0, torch.cos(radi)]], dtype=torch.float32),
            'Z': torch.tensor([[torch.cos(radi), -torch.sin(radi), 0],
                    [torch.sin(radi), torch.cos(radi), 0],
                    [0, 0, 1]], dtype=torch.float32)}
    pc[:, :3] = pc[:, :3] @ xyzArray[axis]
    if requires_inv:
        flip_flag = False
        if radi<-torch.pi/2 or radi>torch.pi/2:
            flip_flag = True
        return pc, torch.linalg.inv(xyzArray[axis]), flip_flag
    else:
        return pc


def scale_trans(pc, scale_low=2./3., scale_high=1.0, space_type='canonical', no_trans=False, trans_range=None):
    xyz1 = torch.rand(3) * (scale_high - scale_low) + scale_low
    if not no_trans:
        if trans_range is None:
            xyz2 = torch.rand(3) * (1 - xyz1.max())
        else:
            xyz2 = torch.rand(3) * min(trans_range, (1 - xyz1.max()))
        if space_type=='normal':
            flag = torch.rand(1)
            if flag<0.5:
                xyz2 = -xyz2
        pc[:, 0:3] = pc[:, 0:3] * xyz1 + xyz2
    else:
        pc[:, 0:3] = pc[:, 0:3] * xyz1
    return pc


def point_cloud_normal_resize(pc, space_type='canonical'):
    if space_type=='normal':
        centroid = torch.mean(pc[:, :3], dim=0)
        pc[:, :3] = pc[:, :3] - centroid
        resize = torch.max(torch.sqrt(torch.sum(pc[:, :3]**2, dim=1)))
        resize = 0.9999 / resize
        pc[:, :3] = pc[:, :3] * resize
    elif space_type=='canonical':
        min_v = torch.min(pc[:, :3], dim=0, keepdim=True)[0]
        pc[:, :3] -= min_v
        max_v = pc[:, :3].max()
        resize = 0.9999 / max_v
        pc[:, :3] = pc[:, :3] * resize
    return pc


def unique_along_y_axis(coors_3d):
    assert coors_3d.shape[1]==3, "coors should contain xyz 3 dimensions!"
    coors_3d = np.unique(coors_3d, axis=0)
    coors_3dy = coors_3d[:, 1]
    indx = np.argsort(coors_3dy) # ascending order
    indx = np.flip(indx) # get descending order because the depth is large when the point is near
    coors_3dxz = np.concatenate((coors_3d[:, 0:1], coors_3d[:, 2:3]), axis=1)

    _, indx_pos = np.unique(coors_3dxz[indx], return_index=True, axis=0)

    return indx, indx_pos


def info_collate_fn(data):
    new_info = {'batch_size': len(data)}
    all_keys = data[0][1].keys()
    pre_num_patch = 0
    all_patch_unq_inv = []
    all_coords_rela_query = []
    all_coords_abs_cut = []
    all_sel_coors = []
    all_sel_features = []
    all_pad_mask = []
    all_attn_mask = []
    all_targets = []
    all_patch_unq = []
    all_images = []
    all_depth = []
    all_text = []
    all_pts = []
    all_patch_unq_hash = []
    for b, unit in enumerate(data):
        if 'depth' in all_keys:
            all_depth.append(unit[1]['depth'])
        if 'image' in all_keys:
            all_images.append(unit[1]['image'])
        if 'text' in all_keys:
            all_text.append(unit[1]['text'])
        if 'pts' in all_keys:
            all_pts.append(unit[1]['pts'])
        patch_num = unit[1]['patch_unq'].shape[0]
        batch_coors_patch_sel = torch.ones((unit[1]['sel_coors'].shape[0], 1), dtype=unit[1]['sel_coors'].dtype) * b
        batch_coors_patch_unq = torch.ones((unit[1]['patch_unq'].shape[0], 1), dtype=unit[1]['sel_coors'].dtype) * b
        all_patch_unq.append(torch.cat((batch_coors_patch_unq, unit[1]['patch_unq']), dim=1))
        all_patch_unq_hash.append(unit[1]['patch_unq_hash'])
        all_patch_unq_inv.append(unit[1]['patch_unq_inv'] + pre_num_patch)
        all_coords_rela_query.append(unit[1]['coords_rela_query'])
        all_coords_abs_cut.append(unit[1]['coords_abs_cut'])
        all_sel_coors.append(torch.cat((batch_coors_patch_sel, unit[1]['sel_coors']), dim=1))
        all_sel_features.append(unit[1]['sel_features'])
        all_pad_mask.append(unit[1]['pad_mask'])
        all_attn_mask.append(unit[1]['attn_mask'])
        all_targets.append(unit['target'])
        pre_num_patch += patch_num
    new_info[1] = dict()
    if 'depth' in all_keys:
        new_info[1]['depth'] = torch.stack(all_depth, dim=0)
    if 'image' in all_keys:
        new_info[1]['image'] = torch.stack(all_images, dim=0)
    if 'text' in all_keys:
        new_info[1]['text'] = torch.stack(all_text, dim=0)
    if 'pts' in all_keys:
        new_info[1]['pts'] = torch.stack(all_pts, dim=0)
    new_info[1]['patch_unq_inv'] = torch.cat(all_patch_unq_inv, dim=0)
    new_info[1]['patch_unq'] = torch.cat(all_patch_unq, dim=0)
    new_info[1]['coords_rela_query'] = torch.cat(all_coords_rela_query, dim=0)
    new_info[1]['coords_abs_cut'] = torch.cat(all_coords_abs_cut, dim=0)
    new_info[1]['sel_coors'] = torch.cat(all_sel_coors, dim=0)
    new_info[1]['sel_features'] = torch.cat(all_sel_features, dim=0)
    new_info[1]['pad_mask'] = torch.cat(all_pad_mask, dim=0)
    new_info[1]['attn_mask'] = torch.stack(all_attn_mask, dim=0)
    new_info[1]['patch_unq_hash'] = torch.cat(all_patch_unq_hash, dim=0)
    new_info['targets'] = torch.cat(all_targets, dim=0)
    return new_info


def load_pseudo3D(path: str) -> Any:
    return torch.from_numpy(np.load(path, allow_pickle=True).astype(np.float32))


def load_img(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_depth(path: str) -> Any:
    if path.endswith('.png'):
        with open(path, 'rb') as f:
            depth = Image.open(f)
            return np.asarray(depth, dtype=np.float32)
    elif path.endswith('.npy'):
        return np.load(path).astype(np.float32)


def create_point_cloud(image, depth, sample_num=2048, depth_scale=256):
    h, w = image.shape[:2]
    image = image.reshape(h*w, 3)

    factor = float(max(h, w, depth_scale))
    # normalize depth
    depth = depth.reshape(-1) / factor
    depth_mean = np.mean(depth)
    depth = -(depth-depth_mean)+depth_mean

    x = np.arange(0, h, dtype=np.float32).reshape(h, 1)
    x = np.repeat(x, w, axis=1).reshape(-1)
    x = x / factor
    x = np.flip(x, axis=0)
    y = np.arange(0, w, dtype=np.float32).reshape(1, w)
    y = np.repeat(y, h, axis=0).reshape(-1)
    y = y / factor
    xyz = np.stack((y, depth, x)).transpose(1, 0)

    xyzrgb = np.concatenate((xyz, image), axis=1).astype(np.float32)
    permutation = np.arange(xyzrgb.shape[0])
    np.random.shuffle(permutation)
    xyzrgb = xyzrgb[permutation]
    return torch.from_numpy(xyzrgb[:sample_num])


def create_point_cloud_with_intrinsics(rgb_img, depth_img, sample_num=2048, camera_intrinsics=None):
    xlin = np.arange(rgb_img.shape[1])
    ylin = np.arange(rgb_img.shape[0])
    xmap, ymap = np.meshgrid(xlin, ylin)
    pixel_map = np.stack([xmap, ymap, np.ones_like(xmap)], axis=-1)
    cam_map = depth_img[:, :, np.newaxis] * (np.linalg.inv(camera_intrinsics) @ pixel_map[:, :, :, np.newaxis]).squeeze()
    xyzrgb = np.concatenate([cam_map, rgb_img[:, :, ::1]], axis=-1).astype(np.float32).reshape([-1, 6])
    xyzrgb = xyzrgb[:, [0, 2, 1, 3, 4, 5]]
    xyzrgb[:, 2] = - (xyzrgb[:, 2] - xyzrgb[:, 2].mean()) # keep z axis up
    permutation = np.arange(xyzrgb.shape[0])
    np.random.shuffle(permutation)
    xyzrgb = xyzrgb[permutation]
    return torch.from_numpy(xyzrgb[:sample_num])


def load_scanobjnn(path: str, with_bg: bool) -> Any:
    pc_ = np.fromfile(path, dtype=np.float32)
    pc_num = pc_[0]
    pc = pc_[1:].copy()
    pc = pc.reshape(int(pc_num), 11)
    if with_bg:
        pass
    else:
        ##To remove backgorund points
        ##filter unwanted class
        filtered_idx = np.intersect1d(np.intersect1d(np.where(pc[:, -1]!=0)[0],np.where(pc[:, -1]!=1)[0]), np.where(pc[:, -1]!=2)[0])
        (values, counts) = np.unique(pc[filtered_idx, -1], return_counts=True)
        max_ind = np.argmax(counts)
        idx = np.where(pc[:, -1]==values[max_ind])[0]
        pc = pc[idx, :]
    # zero_color = np.ones((pc.shape[0], 3), dtype=pc.dtype)
    # pc = np.concatenate((pc[:, :3], pc[:, 6:9]/255), axis=1)
    # pc = np.concatenate((pc[:, :3], zero_color), axis=1)
    pc = torch.from_numpy(pc)
    pc = torch.cat((pc[:, :3], pc[:, 6:9]/255.0), dim=1)
    return pc


def quantitize(data, lim_min, lim_max, size):
    idx = (data - lim_min) / (lim_max - lim_min) * size.float()
    idxlong = idx.type(torch.LongTensor)
    return idxlong


class PcPreprocessor3DSlim(nn.Module):
    def __init__(self, scales=[1], space_size=224, patch_size=16, patch_num=196, rectify_pos=False, pos3d=False, space_type='canonical'):
        super(PcPreprocessor3DSlim, self).__init__()
        self.space_type = space_type
        if space_type=='normal':
            self.x_lims = [-1.0, 1.0]
            self.y_lims = [-1.0, 1.0]
            self.z_lims = [-1.0, 1.0]
            self.space_size = space_size
            self.base = 2.0 / self.space_size
        elif space_type=='canonical':
            self.x_lims = [0.0, 1.0]
            self.y_lims = [0.0, 1.0]
            self.z_lims = [0.0, 1.0]
            self.space_size = space_size
            self.base = 1.0 / self.space_size
        else:
            raise ValueError(f'space type {space_type} not supported')
        self.grid_meters = [self.base, self.base, self.base]
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.offset = 1.0 / (2 * (self.space_size/self.patch_size))
        self.sizes = torch.tensor([int(round((self.x_lims[1] - self.x_lims[0]) / self.grid_meters[0])),
                           int(round((self.y_lims[1] - self.y_lims[0]) / self.grid_meters[1])),
                           int(round((self.z_lims[1] - self.z_lims[0]) / self.grid_meters[2]))])
        self.sizes = self.sizes.long()
        self.lims = [self.x_lims, self.y_lims, self.z_lims]
        self.scales = scales
        assert not (rectify_pos&pos3d), "rectify_pos and pos3d should not be True at the same time!"
        assert not (rectify_pos&(space_type!='canonical')), "rectify_pos works for canonical space only!"
        self.rectify_pos = rectify_pos
        self.pos3d = pos3d

    def patch_select(self, patch_indx):
        patch_indx, patch_unq_inv = torch.unique(patch_indx, return_inverse=True, dim=0)
        cur_length = patch_indx.shape[0]

        sel_mask_cut = torch.zeros((cur_length), dtype=torch.bool)
        rand_ind = torch.randperm(cur_length)
        sel_mask_cut[:self.patch_num] = True
        sel_mask_cut = sel_mask_cut[rand_ind]
        # print(sel_mask_cut.shape, (sel_mask_cut==True).sum())
        sel_mask_cut = sel_mask_cut[patch_unq_inv]
        return sel_mask_cut
    
    def discretization(self, pc, scale):
        xidx = quantitize(pc[:, 0], self.lims[0][0], self.lims[0][1], 
                    torch.div(self.sizes[0].float(), scale, rounding_mode='floor'))
        yidx = quantitize(pc[:, 1], self.lims[1][0], self.lims[1][1], 
                    torch.div(self.sizes[1].float(), scale, rounding_mode='floor'))
        zidx = quantitize(pc[:, 2], self.lims[2][0], self.lims[2][1], 
                    torch.div(self.sizes[2].float(), scale, rounding_mode='floor'))
        bxyz_indx = torch.stack([xidx, yidx, zidx], dim=-1)
        return bxyz_indx
    
    def forward(self, pc, rot_inv=None, flip_flag=None, target=None):
        info = {'batch_size': 1}
        for scale in self.scales:
            bxyz_indx = self.discretization(pc, scale)
            # if self.is_unique_along_y_axis:
            #     indx, indx_pos = unique_along_y_axis(bxyz_indx.numpy())
            #     indx = torch.from_numpy(indx.copy())
            #     indx_pos = torch.from_numpy(indx_pos.copy())
            #     bxyz_indx = bxyz_indx[indx][indx_pos]
            #     pc = pc[indx][indx_pos]
            unq_indx, indx_, inv = np.unique(bxyz_indx.numpy(), return_index=True, return_inverse=True, axis=0)
            unq_indx = torch.from_numpy(unq_indx)
            inv = torch.from_numpy(inv)
            features = scatter(pc, inv, dim=0, reduce="max")
            patch_indx = torch.div(unq_indx, self.patch_size, rounding_mode='floor')
            sel_mask_cut = self.patch_select(patch_indx)

            # point-level stuff selection
            unq_indx = unq_indx[sel_mask_cut]
            features = features[sel_mask_cut]
            patch_indx = patch_indx[sel_mask_cut]
            if target is not None:
                target = target[indx_]
                target = target[sel_mask_cut]

            # selected point-level feature pre-process
            coords_abs = patch_indx.float()
            coords_abs = coords_abs * 2 * self.offset # + self.offset
            # voxel_abs = features[:, :3]
            # voxel_rela = voxel_abs - coords_abs
            # sel_features = torch.cat((features, voxel_rela), dim=1)
            # voxel_rela_dist = torch.sqrt(torch.sum(voxel_rela * voxel_rela, dim=1, keepdim=True))
            # sel_features = torch.cat((features, voxel_rela, voxel_rela_dist, coords_abs), dim=1)

            if self.space_type=='canonical' and self.rectify_pos:
                subs_indx = coords_abs.clone() # - self.offset
                pred_indx = features[:, :3].clone()
                pred_indx = pred_indx - subs_indx
                pred_indx = pred_indx @ rot_inv
                l = self.offset*2
                corners = torch.tensor([[l, 0, l], [l, l, 0], [0, l, l], [l, 0, 0], [0, l, 0], [l, l, l], [0, 0, 0], [0, 0, l]])
                corners = corners @ rot_inv
                min_x = corners[:, 0].min()
                max_x = corners[:, 0].max()
                resize = 0.9999/(max_x-min_x)
                # only x-axis need trans and rescale
                pred_indx[:, 0] = pred_indx[:, 0] - min_x
                pred_indx = pred_indx * resize
                pred_indx = pred_indx * self.patch_size
                pred_indx = pred_indx.int()
                if flip_flag:
                    pred_indx[:, 0] = -pred_indx[:, 0]+self.patch_size-1

                assert pred_indx[:, 0].max()<self.patch_size, "pred_indx's x-axis is large than patch size"
                assert pred_indx[:, 0].min()>=0, "pred_indx's x-axis is smaller than 0"
                assert pred_indx[:, 2].max()<self.patch_size, "pred_indx's z-axis is large than patch size"
                assert pred_indx[:, 2].min()>=0, "pred_indx's z-axis is smaller than 0"

            # selected point-level query pre-process
            coords_rela = (unq_indx % self.patch_size).long()
            coords_rela_query = coords_rela[:, 0]
            for i in range(2):
                coords_rela_query[:] = coords_rela_query[:] + coords_rela[:, i+1] * (self.patch_size ** (i + 1))
            
            # patch pre-process
            patch_unq, patch_unq_indx, patch_unq_inv = np.unique(patch_indx.numpy(), return_index=True, return_inverse=True, axis=0)
            patch_unq = torch.from_numpy(patch_unq)
            patch_unq_indx = torch.from_numpy(patch_unq_indx)
            patch_unq_inv = torch.from_numpy(patch_unq_inv)

            # create sparse index of patches
            patch_unq_hash = patch_unq[:, 0]
            for i in range(2):
                patch_unq_hash[:] = patch_unq_hash[:] + patch_unq[:, i+1] * (int(224.0/self.patch_size) ** (i + 1))

            # build graph
            voxel_center = scatter(features[:, :3], patch_unq_inv, dim=0, reduce="mean")
            voxel_center = voxel_center[patch_unq_inv]
            voxel_rela = features[:, :3] - voxel_center
            sel_features = torch.cat((features, voxel_rela, voxel_center), dim=1)

            # voxel_center = scatter(voxel_rela, patch_unq_inv, dim=0, reduce="mean")
            # voxel_center = voxel_center[patch_unq_inv]
            # sel_features = torch.cat((sel_features, voxel_center), dim=1)
            # voxel_abs_sub = voxel_abs-voxel_center
            # voxel_abs_distance = torch.sqrt(torch.sum(voxel_abs_sub * voxel_abs_sub, dim=1, keepdim=True))
            # sel_features = torch.cat((sel_features, voxel_abs_sub, voxel_abs_distance, voxel_center), dim=1)
            
            # patch-level positional embeddings padding
            coords_abs = coords_abs[patch_unq_indx]
            cur_length = coords_abs.shape[0]
            if cur_length<self.patch_num:
                void_coords_abs = torch.zeros((self.patch_num-cur_length, coords_abs.shape[1]), dtype=coords_abs.dtype)
                coords_abs = torch.cat((coords_abs, void_coords_abs), dim=0)
                void_hash = torch.zeros((self.patch_num-cur_length), dtype=patch_unq_hash.dtype)
                patch_unq_hash = torch.cat((patch_unq_hash, void_hash), dim=0)

            # patch-level features padding
            pad_mask = torch.ones((self.patch_num), dtype=torch.bool)
            pad_mask[cur_length:] = False
            attn_mask = torch.zeros((self.patch_num, self.patch_num), dtype=torch.float32)
            attn_mask[cur_length:, cur_length:] = -torch.inf

            # point-level loss query pre-process
            patch_indicator = torch.arange(cur_length, dtype=torch.int32)
            patch_indicator = patch_indicator[patch_unq_inv]
            if self.space_type=='canonical' and self.rectify_pos:
                sel_coors_inpatch = pred_indx[:, 0:1] * self.patch_size + pred_indx[:, 2:3]
                unq_indx = torch.cat((patch_indicator.reshape(-1, 1), sel_coors_inpatch), dim=1)
            else:
                if self.pos3d:
                    unq_indx = unq_indx - patch_indx * self.patch_size
                    sel_coors_inpatch = unq_indx[:, 0:1] * self.patch_size**2 + unq_indx[:, 1:2] * self.patch_size + unq_indx[:, 2:3]
                    unq_indx = torch.cat((patch_indicator.reshape(-1, 1), sel_coors_inpatch), dim=1)
                else:
                    unq_indx = unq_indx - patch_indx * self.patch_size
                    sel_coors_inpatch = unq_indx[:, 0:1] * self.patch_size + unq_indx[:, 2:3]
                    unq_indx = torch.cat((patch_indicator.reshape(-1, 1), sel_coors_inpatch), dim=1)

            info[scale] = dict()
            info[scale]['coords_rela_query'] = coords_rela_query
            info[scale]['coords_abs_cut'] = coords_abs
            info[scale]['sel_coors'] = unq_indx
            info[scale]['sel_features'] = sel_features # (0-3: (absolute xyz), 3-6: (rgb), 6-9: (voxel graph within a patch), 9-12:(voxel center within a patch))
            info[scale]['pad_mask'] = pad_mask
            info[scale]['patch_unq_inv'] = patch_unq_inv
            info[scale]['patch_unq'] = patch_unq
            info[scale]['patch_unq_hash'] = patch_unq_hash
            info[scale]['attn_mask'] = attn_mask
            if target is not None:
                info['target'] = target

        return info


@DATASETS.register_module()
class Pseudo3D(VisionDataset):

    def __init__(self, config):
        super(Pseudo3D, self).__init__(config.ROOT)
        samples = self.make_dataset(self.root, config.EXTRA, config.subset)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.samples = samples
        self.targets = [s[1] for s in samples]
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
        if hasattr(config, "nocolor"):
            self.nocolor = config.nocolor
        else:
            self.nocolor = False
        self.preprocess = PcPreprocessor3DSlim(scales=[1], space_size=self.space_size, patch_size=config.patch_size, patch_num=patch_num, rectify_pos=self.rectify_pos, pos3d=self.pos3d, space_type=self.space_type)
        # self.color_mean = np.array([0.47477615, 0.46219772, 0.40503177])
        # self.color_std = np.array([0.7297382, 0.7767892, 1.])
        self.color_mean = np.array([0.485, 0.456, 0.406])
        self.color_std = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def make_dataset(
        directory: str,
        directory_ex: str,
        split: str,
    ) -> List[Tuple[str, int]]:
        instances = []
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                depth_path = os.path.join(root, fname)
                img_path = depth_path.replace('/train_depth_v2/', '/train/').replace('_img_depth.npy', '.JPEG')
                item = img_path, depth_path
                instances.append(item)
        return instances

    def __getitem__(self, index: int):
        # load
        if len(self.samples[index])==3: # for real RGB-D scans, we have camera intrinsic
            img_path, depth_path, intrinsic = self.samples[index]
            image = load_img(img_path)
            depth = load_depth(depth_path)
            depth = np.asarray(depth, dtype=np.float32)
            height, width = depth.shape[:2]
            resized_image = image.resize((width, height), Image.NEAREST)
            resized_image = np.asarray(resized_image, dtype=np.float32) / 255.0
            if self.space_type=='normal':
                image = (image - self.color_mean) / self.color_std
            pc = create_point_cloud_with_intrinsics(resized_image, depth, sample_num=self.npoints, camera_intrinsics=intrinsic)
        else:
            img_path, depth_path = self.samples[index]
            depth = load_depth(depth_path)
            image = load_img(img_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            if self.space_type=='normal':
                image = (image - self.color_mean) / self.color_std
            pc = create_point_cloud(image, depth, sample_num=self.npoints)

        # For point-MAE
        # pc = pc[:, :3]
        # while True:
        #     if pc.shape[0]>=self.npoints:
        #         pc = pc[:self.npoints, :]
        #         break
        #     else:
        #         pc = torch.cat((pc, pc), dim=0)
        if self.nocolor:
            pc = pc[:, :3]
        pc = flip_pseudo3D(pc, plane='YZ')
        if self.rectify_pos:
            pc, rot_inv, flip_flag = rotate_pseudo3D(pc, low_radius=-torch.pi, high_radius=torch.pi, axis='Z', requires_inv=True)
        else:
            pc = rotate_pseudo3D(pc,  low_radius=-torch.pi, high_radius=torch.pi, axis='Z')
        pc = point_cloud_normal_resize(pc, space_type=self.space_type)
        pc = scale_trans(pc, scale_low=1./2., scale_high=1.0, space_type=self.space_type)
        if self.rectify_pos:
            info = self.preprocess(pc, rot_inv, flip_flag)
        else:
            info = self.preprocess(pc)
        # For point-MAE
        # info = dict()
        # info[1] = dict()
        # info[1]['pts'] = pc
        info['target'] = torch.tensor([0])
        return info

    def __len__(self) -> int:
        return len(self.samples)


@DATASETS.register_module()
class ScanobjNNcolor(VisionDataset):

    def __init__(self, config):
        super(ScanobjNNcolor, self).__init__(config.ROOT)
        self.split = config.subset
        self.with_bg = config.with_bg
        if hasattr(config, "npoints"):
            self.npoints = config.npoints
        else:
            self.npoints = 1000000
        
        if hasattr(config, "reduce"):
            reduce = config.reduce
        else:
            reduce = False

        if hasattr(config, 'no_aug'):
            self.no_aug = config.no_aug
        else:
            self.no_aug = False

        if hasattr(config, 'space_size'):
            self.space_size = config.space_size
        else:
            self.space_size = 224

        if hasattr(config, "patch_num"):
            patch_num = config.patch_num
        else:
            patch_num = (self.space_size//config.patch_size)**2

        if hasattr(config, "nocolor"):
            self.nocolor = config.nocolor
        else:
            self.nocolor = False

        samples = self.make_dataset(self.root, self.split, reduce)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.samples = samples
        self.targets = [s[1] for s in samples]
        if hasattr(config, 'space_type'):
            self.space_type = config.space_type
        else:
            self.space_type = 'canonical'
        self.preprocess = PcPreprocessor3DSlim(scales=[1], space_size=self.space_size, patch_size=config.patch_size, patch_num=patch_num, space_type=self.space_type)
        # self.color_mean = torch.tensor([0.430, 0.388, 0.348])
        # self.color_std = torch.tensor([0.707, 0.707, 0.707])
        self.color_mean = np.array([0.485, 0.456, 0.406])
        self.color_std = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def make_dataset(
        directory: str,
        split: str,
        reduce: bool,
    ) -> List[Tuple[str, int]]:
        with open(os.path.join(directory, 'split.txt'), 'r') as f:
            lines = f.readlines()

        target_files = []
        for line in lines:
            line = line.split('\t')
            length = len(line)
            if split=='train' and length==2:
                target_files.append(line[0])
            elif split=='test' and length==3:
                target_files.append(line[0])

        instances = []
        class_index = 0
        for root, dirs, fnames in sorted(os.walk(directory)):
            if root==directory:
                class_dirs = dirs
            else:
                for fname in fnames:
                    if fname in target_files:
                        path = os.path.join(root, fname)
                        item = path, class_index
                        instances.append(item)
                class_index += 1
        if reduce:
            instances = sample(instances, int(len(instances)/2))
        return instances

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        pc = load_scanobjnn(path, with_bg=self.with_bg)
        if pc.shape[0]>self.npoints:
            pt_idxs = torch.randperm(pc.shape[0])
            pc = pc[pt_idxs, :][:self.npoints, :]

        if self.nocolor:
            pc = torch.cat((pc[:, 0:1], pc[:, 2:3], pc[:, 1:2]), dim=1)
        else:
            if self.space_type=='normal':
                pc[:, 3:] = (pc[:, 3:] - self.color_mean) / self.color_std
            pc = torch.cat((pc[:, 0:1], pc[:, 2:3], pc[:, 1:2], pc[:, 3:]), dim=1)
        if not self.no_aug:
            pc = flip_pseudo3D(pc, plane='YZ')
            pc = rotate_pseudo3D(pc,  low_radius=-torch.pi, high_radius=torch.pi, axis='Z')
        pc = point_cloud_normal_resize(pc, space_type=self.space_type)
        if not self.no_aug:
            pc = scale_trans(pc, scale_low=4./5., scale_high=1.0, space_type=self.space_type)
        info = self.preprocess(pc)
        ### uncomment lines below to do PointTransformer evaluation
        # while True:
        #     if pc.shape[0]>=self.npoints:
        #         pc = pc[:self.npoints, :]
        #         break
        #     else:
        #         pc = torch.cat((pc, pc), dim=0)
        # info[1]['pts'] = pc
        ### end
        info['target'] = torch.tensor([target])
        return info

    def __len__(self) -> int:
        return len(self.samples)
    

@DATASETS.register_module()
class ScanobjNNcolor_hardest(VisionDataset):

    def __init__(self, config):
        super(ScanobjNNcolor_hardest, self).__init__(config.ROOT)
        self.split = config.subset
        self.with_bg = config.with_bg
        if hasattr(config, "npoints"):
            self.npoints = config.npoints
        else:
            self.npoints = 1000000
        
        if hasattr(config, "reduce"):
            reduce = config.reduce
        else:
            reduce = False

        if hasattr(config, 'no_aug'):
            self.no_aug = config.no_aug
        else:
            self.no_aug = False

        if hasattr(config, "patch_num"):
            patch_num = config.patch_num
        else:
            patch_num = (224//config.patch_size)**2
        
        if hasattr(config, "nocolor"):
            self.nocolor = config.nocolor
        else:
            self.nocolor = False

        samples = self.make_dataset(self.root, self.split, reduce)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.samples = samples
        self.targets = [s[1] for s in samples]
        if hasattr(config, 'space_type'):
            self.space_type = config.space_type
        else:
            self.space_type = 'canonical'
        self.preprocess = PcPreprocessor3DSlim(scales=[1], patch_size=config.patch_size, patch_num=patch_num, space_type=self.space_type)
        self.color_mean = np.array([0.485, 0.456, 0.406])
        self.color_std = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def make_dataset(
        directory: str,
        split: str,
        reduce: bool,
    ) -> List[Tuple[str, int]]:
        with open(os.path.join(directory, 'split.txt'), 'r') as f:
            lines = f.readlines()

        target_files = []
        for line in lines:
            line = line.split('\t')
            length = len(line)
            if split=='train' and length==2:
                target_files.append(line[0].replace('.bin', ''))
            elif split=='test' and length==3:
                target_files.append(line[0].replace('.bin', ''))

        instances = []
        class_index = 0
        for root, dirs, fnames in sorted(os.walk(directory)):
            if root==directory:
                class_dirs = dirs
            else:
                for fname in fnames:
                    for target in target_files:
                        if target in fname:
                            path = os.path.join(root, fname)
                            item = path, class_index
                            instances.append(item)
                            break
                class_index += 1
        if reduce:
            instances = sample(instances, int(len(instances)/10))
        return instances

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        pc = load_scanobjnn(path, with_bg=self.with_bg)
        if pc.shape[0]>self.npoints:
            pt_idxs = torch.randperm(pc.shape[0])
            pc = pc[pt_idxs, :][:self.npoints, :]
        if self.nocolor:
            pc = torch.cat((pc[:, 0:1], pc[:, 2:3], pc[:, 1:2]), dim=1)
        else:
            if self.space_type=='normal':
                pc[:, 3:] = (pc[:, 3:] - self.color_mean) / self.color_std
            pc = torch.cat((pc[:, 0:1], pc[:, 2:3], pc[:, 1:2], pc[:, 3:]), dim=1)
        if not self.no_aug:
            pc = flip_pseudo3D(pc, plane='YZ')
            pc = rotate_pseudo3D(pc,  low_radius=-torch.pi, high_radius=torch.pi, axis='Z')
        pc = point_cloud_normal_resize(pc, space_type=self.space_type)
        if not self.no_aug:
            pc = scale_trans(pc, scale_low=4./5., scale_high=1.0, space_type=self.space_type)
        info = self.preprocess(pc)
        # info[1]['pts'] = pc # uncomment this line to do PointTransformer evaluation
        info['target'] = torch.tensor([target])
        return info

    def __len__(self) -> int:
        return len(self.samples)