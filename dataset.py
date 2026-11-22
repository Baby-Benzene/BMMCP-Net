import numpy as np
import torch.utils.data
import os
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, file_root='data/', transform=None):
        """
        dataset: set name (train, val, or test)
        file_root: dataset root directory path
        transform: image transformation operation
        """
        self.dataset = dataset
        self.file_root = file_root
        self.transform = transform

        self.dir_A = os.path.join(self.file_root, self.dataset, 'A')

        self.file_list = [
            x for x in os.listdir(self.dir_A)
            if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        pre_image_path = os.path.join(self.file_root, self.dataset, 'A', name)
        post_image_path = os.path.join(self.file_root, self.dataset, 'B', name)
        label_path = os.path.join(self.file_root, self.dataset, 'label', name)


        pre_image = Image.open(pre_image_path).convert('RGB')
        post_image = Image.open(post_image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        pre_np = np.array(pre_image)
        post_np = np.array(post_image)
        label_np = np.array(label)
        img = np.concatenate((pre_np, post_np), axis=2)


        if self.transform:
            img, label = self.transform(img, label_np)

        return img, label

    def get_img_info(self, idx):
        name = self.file_list[idx]
        img_path = os.path.join(self.file_root, self.dataset, 'A', name)
        img = Image.open(img_path)
        return {"height": img.size[1], "width": img.size[0]}