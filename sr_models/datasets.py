import random
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from io import BytesIO

import copy

class TrainDataset(Dataset):
    def __init__(self, file_path, patch_size, scale, aug=False, colorization=False, completion=False):
        super(TrainDataset, self).__init__()
        self.files = ParseFile(file_path)

        self.patch_size = patch_size
        self.scale = scale
        self.aug = aug
        self.colorization = colorization
        self.completion = completion

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    # im is an numpy float/double array
    @staticmethod
    def add_gaussian_noise(im, std):
        noise = np.random.normal(0,std,im.shape)
        im = im + noise
        return im

    # im is read from PIL.Image.open
    @staticmethod
    def jpeg(im, jpeg_quality):
        buffer = BytesIO()
        im.save(buffer, 'jpeg', quality = jpeg_quality)
        im = Image.open(buffer)
        return im

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img2 = img.copy()
        hr = np.array(img).astype('float')
        if self.aug and np.random.uniform(0,1) > 0.7071:
            img2 = self.jpeg(img2, int(np.random.choice(np.arange(25, 75))))
            #print('agument jpeg')

        hr2 = np.array(img2).astype('float')
        hr2[:,:,0] = convolve(hr2[:,:,0] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,1] = convolve(hr2[:,:,1] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,2] = convolve(hr2[:,:,2] , np.ones((15,15)).astype('float')/225)

        lr = 0
        for i in range(self.scale):
            for j in range(self.scale):
                lr = lr + hr[i::self.scale, j::self.scale] / (self.scale * self.scale)

        lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        lr, hr = self.random_horizontal_flip(lr, hr)
        lr, hr = self.random_vertical_flip(lr, hr)
        lr, hr = self.random_rotate_90(lr, hr)

        if self.aug and np.random.uniform(0,1) > 0.7071:
            lr = self.add_gaussian_noise(lr, np.random.uniform(0,10))
            #print('augment noising')

        lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0

        if self.completion and np.random.uniform(0,1) > 0.7071:
            dims = lr.shape
            mask = np.random.uniform(0,1,(dims[1],dims[2]))
            mask = mask < np.random.uniform(0.05,0.15)
            lr[0,mask] = 0
            lr[1,mask] = 0
            lr[2,mask] = 0

        if self.colorization and np.random.uniform(0,1) > 0.7071:
            dims = lr.shape
            mask = np.random.uniform(0,1,(dims[1],dims[2]))
            mask = mask < np.random.uniform(0.05,0.15)
            tmp = lr.mean(axis=0)
            for i_dim in range(dims[0]):
                lr[i_dim,mask] = tmp[mask]

        return lr, hr

    def __len__(self):
        return len(self.files)

class TrainDataset256(Dataset):
    def __init__(self, file_path, patch_size, scale, aug=False, colorization=False, completion=False):
        super(TrainDataset256, self).__init__()
        self.files = ParseFile(file_path)

        self.patch_size = patch_size
        self.scale = scale
        self.aug = aug
        self.colorization = colorization
        self.completion = completion

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    # im is an numpy float/double array
    @staticmethod
    def add_gaussian_noise(im, std):
        noise = np.random.normal(0,std,im.shape)
        im = im + noise
        return im

    # im is read from PIL.Image.open
    @staticmethod
    def jpeg(im, jpeg_quality):
        buffer = BytesIO()
        im.save(buffer, 'jpeg', quality = jpeg_quality)
        im = Image.open(buffer)
        return im

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = img.resize((256 , 256), resample=Image.BICUBIC)

        img2 = img.copy()
        hr = np.array(img).astype('float')
        if self.aug and np.random.uniform(0,1) > 0.7071:
            img2 = self.jpeg(img2, int(np.random.choice(np.arange(25, 75))))
            #print('agument jpeg')

        hr2 = np.array(img2).astype('float')
        hr2[:,:,0] = convolve(hr2[:,:,0] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,1] = convolve(hr2[:,:,1] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,2] = convolve(hr2[:,:,2] , np.ones((15,15)).astype('float')/225)

        lr = 0
        for i in range(self.scale):
            for j in range(self.scale):
                lr = lr + hr[i::self.scale, j::self.scale] / (self.scale * self.scale)

        lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        lr, hr = self.random_horizontal_flip(lr, hr)
        lr, hr = self.random_vertical_flip(lr, hr)
        lr, hr = self.random_rotate_90(lr, hr)

        if self.aug and np.random.uniform(0,1) > 0.7071:
            lr = self.add_gaussian_noise(lr, np.random.uniform(0,10))
            #print('augment noising')

        lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0

        if self.completion and np.random.uniform(0,1) > 0.7071:
            dims = lr.shape
            mask = np.random.uniform(0,1,(dims[1],dims[2]))
            mask = mask < np.random.uniform(0.05,0.15)
            lr[0,mask] = 0
            lr[1,mask] = 0
            lr[2,mask] = 0

        if self.colorization and np.random.uniform(0,1) > 0.7071:
            dims = lr.shape
            mask = np.random.uniform(0,1,(dims[1],dims[2]))
            mask = mask < np.random.uniform(0.05,0.15)
            tmp = lr.mean(axis=0)
            for i_dim in range(dims[0]):
                lr[i_dim,mask] = tmp[mask]

        return lr, hr

    def __len__(self):
        return len(self.files)

class EvalDataset(Dataset):
    def __init__(self, file_path, scale):
        super(EvalDataset, self).__init__()
        self.files = ParseFile(file_path)
        self.scale = scale

    def __getitem__(self, idx):
        hr = np.array(Image.open(self.files[idx])).astype('float')
        hr2 = hr.copy()
        hr2[:,:,0] = convolve(hr2[:,:,0] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,1] = convolve(hr2[:,:,1] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,2] = convolve(hr2[:,:,2] , np.ones((15,15)).astype('float')/225)

        lr = 0
        for i in range(self.scale):
            for j in range(self.scale):
                lr = lr + hr[i::self.scale, j::self.scale] / (self.scale * self.scale)

        return lr, hr

    def __len__(self):
        return len(self.files)

def ParseFile(filepath):
    output = []
    with open(filepath) as fp:
        for line in fp:
            output.append(line[:-1])

    return output


class EvalDataset256(Dataset):
    def __init__(self, file_path, scale):
        super(EvalDataset256, self).__init__()
        self.files = ParseFile(file_path)
        self.scale = scale

    def __getitem__(self, idx):
        hr = np.array(Image.open(self.files[idx])).astype('float')
        hr = hr.resize((256 , 256), resample=Image.BICUBIC)
        hr2 = hr.copy()
        hr2[:,:,0] = convolve(hr2[:,:,0] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,1] = convolve(hr2[:,:,1] , np.ones((15,15)).astype('float')/225)
        hr2[:,:,2] = convolve(hr2[:,:,2] , np.ones((15,15)).astype('float')/225)

        lr = 0
        for i in range(self.scale):
            for j in range(self.scale):
                lr = lr + hr[i::self.scale, j::self.scale] / (self.scale * self.scale)

        return lr, hr

    def __len__(self):
        return len(self.files)

def ParseFile(filepath):
    output = []
    with open(filepath) as fp:
        for line in fp:
            output.append(line[:-1])

    return output
