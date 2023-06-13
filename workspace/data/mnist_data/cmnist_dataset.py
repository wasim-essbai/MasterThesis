from torch.utils.data import Dataset
from skimage import io
import cv2
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image


class CMNISTDataset(Dataset):

    def __init__(self, root_dir, train = True, transform = None):
        self.root_dir = root_dir
        self.train = train
        self.data, self.targets = self.load_data()
        self.transform = transform

    def load_data(self):
        data_file = 'train_images' if self.train else 'test_images'
        targets_file = 'train_labels' if self.train else 'test_labels'
        
        data = np.load(os.path.join(self.root_dir + data_file + '.npy'))
        data = data.reshape(data.shape[0], 28, 28)

        targets = np.load(os.path.join(self.root_dir + targets_file + '.npy'))

        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target