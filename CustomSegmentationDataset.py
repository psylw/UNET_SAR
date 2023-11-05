
import os
import numpy as np
from torch.utils.data import Dataset
import torch


class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_files, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.patches = self.generate_patches()
        self.image_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'masks')
        self.images = image_files
        
        mlist = os.listdir(self.mask_folder)

        self.masks = [] # get label for image (2 images, or i could concat images)
        for i in range(len(self.images)):
            for j in mlist:
                if self.images[i][0:5] in j:
                    self.masks.append(j)

    def __len__(self):
        return len(self.image_folder) * len(self.patches)

    def __getitem__(self, idx):

        image_idx = idx // len(self.patches)  # Get the index of the image
        patch_idx = idx % len(self.patches)  # Get the index of the patch

        image_name = self.images[image_idx]
        mask_name = self.masks[image_idx]

        image = np.load(os.path.join(self.image_folder,image_name)).astype(np.float32)
        mask = np.load(os.path.join(self.mask_folder,mask_name)).astype(np.float32)

        image[np.isnan(image)] = 0
        mask[np.isnan(mask)] = 0

        # Crop and transform the specified patch
        left, upper, right, lower = self.patches[patch_idx]

        image = image[upper:lower, left:right]
        mask = mask[upper:lower, left:right]


        # Apply any transformations to the image and mask
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.transform(mask)

        return image, mask

    def generate_patches(self):
        patches = []
        patch_size = 32  # Size of the patch

        for _ in range(patch_size, 1024, patch_size):
            for j in range(patch_size, 1024, patch_size):
                patch = (j - patch_size, _ - patch_size, j, _)
                patches.append(patch)

        return patches

    


