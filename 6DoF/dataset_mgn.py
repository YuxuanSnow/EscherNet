import os
import math
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys
import cv2

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

class SMPLDV1DataLoader():
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        # super().__init__(self, root_dir, batch_size, total_view, num_workers)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [torchvision.transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = SMPLDV1Data(root_dir=self.root_dir, total_view=self.total_view, validation=False,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) , # the webloader is not working
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = SMPLDV1Data(root_dir=self.root_dir, total_view=self.total_view, validation=True,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class SMPLDV1Data(Dataset):
    def __init__(self,
                 root_dir='/mnt/qb/work/ponsmoll/yxue80/project/RVH-123/rvh123',
                 image_transforms=None,
                 total_view=16,
                 validation=False
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.total_view = total_view

        if os.path.exists(os.path.join(self.root_dir, 'smpld_v1_rendering.json')):
            with open(os.path.join(self.root_dir, 'smpld_v1_rendering.json')) as f:
                self.paths = json.load(f)
        else:
            assert False, "Impossible to load path data"

        # debug
        # self.paths = self.paths[::10]
        
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths) # human identity * shape pose variation (no cam views considered yet)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
    def get_pose(self, transformation):
        # transformation: 4x4
        return transformation

    def __getitem__(self, index):
        data = {}
        total_view = self.total_view 
        index_target, index_cond = random.sample(range(total_view), 2)  # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        dataset = filename.split('/')[-3]
        subject = filename.split('/')[-2]

        # print(self.paths[index])

        color = [1., 1., 1., 1.]

        try:
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            for i, index_input in enumerate([index_cond]):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d_cam.npy' % index_input))
                cond_Ts.append(self.get_pose(np.concatenate([input_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0))) # use absolute pose
            for i, index_target in enumerate([index_target]):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d_cam.npy' % index_target))
                target_Ts.append(self.get_pose(np.concatenate([target_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))

        except:
            # very hacky solution, sorry about this
            assert False, "Impossible to load data"

        data['image_input'] = torch.stack(input_ims, dim=0)
        data['image_target'] = torch.stack(target_ims, dim=0)
        data['pose_out'] = np.stack(target_Ts)
        data['pose_out_inv'] = np.linalg.inv(np.stack(target_Ts)).transpose([0, 2, 1])
        data['pose_in'] = np.stack(cond_Ts)
        data['pose_in_inv'] = np.linalg.inv(np.stack(cond_Ts)).transpose([0, 2, 1])
        data["subject"] = dataset + '/' + subject

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

class SCANV1Data(Dataset):
    def __init__(self,
                 root_dir='/mnt/qb/work/ponsmoll/yxue80/project/RVH-123/rvh123',
                 image_transforms=None,
                 total_view=16,
                 validation=False
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.total_view = total_view

        if os.path.exists('/mnt/qb/work/ponsmoll/yxue80/project/RVH-123/rvh123/scan_v1_rendering.json'):
            with open('/mnt/qb/work/ponsmoll/yxue80/project/RVH-123/rvh123/scan_v1_rendering.json') as f:
                self.paths = json.load(f)
        else:
            assert False, "Impossible to load path data"

        # debug
        # self.paths = self.paths[::10]
        
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths) # human identity * shape pose variation (no cam views considered yet)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = self.total_view 
        index_target, index_cond = random.sample(range(total_view), 2)  # without replacement
        filename = self.paths[index]

        dataset = filename.split('/')[-3]
        subject = filename.split('/')[-2]

        # print(self.paths[index])

        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d_cam.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d_cam.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '0a0c6d3b5f58499db8d6d649ba8de189')  # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d_cam.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d_cam.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)
        data["subject"] = dataset + '/' + subject

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


