import numpy as np
import torch

class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points

class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc
    

class PointcloudFlipPseudo3D(object):
    def __init__(self, plane):
        assert plane=='XY' or plane=='XZ' or plane=='YZ', "Flip plane should be assigned as XY or XZ or YZ!!"
        self.plane = plane

    def flip_it(self, pc_):
        pc_mean = np.mean(pc_)
        pc_ = -(pc_-pc_mean)+pc_mean
        return pc_
    
    def __call__(self, pc):
        flag = np.random.randint(0, 2)
        if flag:
            if self.plane=='XY':
                pc[:, 2] = self.flip_it(pc[:, 2])
            elif self.plane=='XZ':
                pc[:, 1] = self.flip_it(pc[:, 1])
            elif self.plane=='YZ':
                pc[:, 0] = self.flip_it(pc[:, 0])
        return pc

    
class PointcloudRotatePseudo3D(object):
    def __init__(self, max_degree, axis):
        self.max_degree = max_degree
        self.axis = axis

    def cal_rot_mat(self, radi):
        xyzArray = {
            'X': np.array([[1, 0, 0],
                    [0, np.cos(radi), -np.sin(radi)],
                    [0, np.sin(radi), np.cos(radi)]]),
            'Y': np.array([[np.cos(radi), 0, np.sin(radi)],
                    [0, 1, 0],
                    [-np.sin(radi), 0, np.cos(radi)]]),
            'Z': np.array([[np.cos(radi), -np.sin(radi), 0],
                    [np.sin(radi), np.cos(radi), 0],
                    [0, 0, 1]])}
        return xyzArray[self.axis]
    
    def __call__(self, pc):
        radi = np.random.uniform(-self.max_degree/2, self.max_degree/2)
        rot_mat = self.cal_rot_mat(radi)
        pc[:, :3] = np.dot(pc[:, :3], rot_mat)
        return pc
    
class PointcloudScaleAndTranslatePseudo3D(object):
    def __init__(self, scale_low=2. / 3., scale_high=1.0):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):

        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz2 = np.random.uniform(low=0, high=(1 - xyz1.max() - 0.0001), size=[3])
        
        pc[:, 0:3] = pc[:, 0:3] * xyz1 + xyz2
            
        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc
