import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import rgb2mask


class KidneyBiopsyDataset(Dataset):
    def __init__(self, images_path, masks_path, patch_size, split, transform=None, n_augs=0, downscale=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.patch_size = patch_size
        self.split = split
        self.transform = transform
        self.n_samples = len(images_path)
        self.downscale = downscale
        self.n_patches = self.__get_n_patches__()

        if n_augs <= 0:
            self.n_augs = 1
        else:
            self.n_augs = n_augs

    def __len__(self):
        if self.split == 'train':
            return self.n_samples * self.n_augs * self.n_patches
        elif self.split == 'valid':
            return self.n_samples * self.n_patches
        elif self.split == 'test':
            return self.n_samples * self.n_patches

    def __getitem__(self, index):

        patch_number = index % self.n_patches
        image_number = index // self.n_patches

        # Augmentation (Increase the size of the dataset by number of augmentations)
        if self.transform:
            image_number = image_number // self.n_augs

        # Read image as grayscale image
        image = cv2.imread(self.images_path[image_number], cv2.IMREAD_GRAYSCALE)

        # Read rgb mask as rgb image
        mask = cv2.imread(self.masks_path[image_number], cv2.IMREAD_COLOR)

        # Crop image and mask (3072 x 4080) -> (3072 x 3584)
        image = image[:, 128:3712]
        mask = mask[:, 128:3712]
        
        # Downscale image and mask (3072 x 3584) -> (1536 x 1792)
        if self.downscale:
            down_points = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, down_points, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, down_points, interpolation=cv2.INTER_NEAREST)

        # n x m image
        n = image.shape[0] // self.patch_size
        m = image.shape[1] // self.patch_size

        # Find i and j based on the patch number
        i = patch_number // m
        j = patch_number % m

        # Create the patch
        image = image[i * self.patch_size: i * self.patch_size + self.patch_size,
                j * self.patch_size: j * self.patch_size + self.patch_size]
        mask = mask[i * self.patch_size: i * self.patch_size + self.patch_size,
               j * self.patch_size: j * self.patch_size + self.patch_size]

        # Convert rgb mask to grayscale with values [0, 1, 2]
        mask = rgb2mask(mask)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
    def __get_n_patches__(self):
        image = cv2.imread(self.images_path[0], cv2.IMREAD_GRAYSCALE)
        image = image[:, 128:3712]
        if self.downscale:
            down_points = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, down_points, interpolation=cv2.INTER_NEAREST)

        try:
            if image.shape[0] * image.shape[1] % (self.patch_size ** 2) != 0:
                # raise the ValueError
                raise ValueError("ValueError: Please insert a valid patch size")
            return image.shape[0] * image.shape[1] // (self.patch_size ** 2)
        except ValueError as e:
            print(e)