# coding: utf-8


import os
import h5py
import torch
import torchvision
import numpy as np
import scipy.io as sio
# from utils import normalize


data_path = 'dataset/'
size = 256


class HyperSpectralDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, mask_path, concat=False, tanh=False, data_key='data', transform=None):

        self.img_path = img_path
        self.data = os.listdir(img_path)
        # self.mask = sio.loadmat(os.path.join(img_path, mask_path))
        mask = sio.loadmat(mask_path)['data'].astype(np.float32)
        self.mask = torchvision.transforms.ToTensor()(mask)
        self.data_len = len(self.data)
        self.tanh = tanh
        self.concat = concat
        self.data_key = data_key
        self.transforms = transform

    def __getitem__(self, idx):
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        measurement_data = (trans_data * self.mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, self.mask], dim=0)
        else:
            input_data = measurement_data
        return input_data, label_data

    def __len__(self):
        return self.data_len


class PatchMaskDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, mask_path, concat=False, tanh=False, data_key='data', transform=None):

        self.img_path = img_path
        self.data = os.listdir(img_path)
        self.mask_path = mask_path
        self.data_len = len(self.data)
        self.tanh = tanh
        self.concat = concat
        self.data_key = data_key
        self.transforms = transform
        self.mask_transforms = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, mask], dim=0)
        else:
            input_data = measurement_data
        return input_data, label_data

    def __len__(self):
        return self.data_len


class PatchEvalDataset(PatchMaskDataset):

    def __getitem__(self, idx):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, mask], dim=0)
        else:
            input_data = measurement_data
        return self.data[idx], input_data, label_data


class PatchMaskDataset_h5py(PatchMaskDataset):

    def __getitem__(self, idx):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        # mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        with h5py.File(os.path.join(self.img_path, self.data[idx]), 'r') as f:
            mat_data = f[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, mask], dim=0)
        else:
            input_data = measurement_data
        return input_data, label_data

class PatchMasDataset_ISTA(PatchMaskDataset):

    def __getitem__(self, idx):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, mask], dim=0)
        else:
            input_data = measurement_data
        return input_data, label_data
