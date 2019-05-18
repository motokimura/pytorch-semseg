import os

import numpy as np
from PIL import Image

import torch
from torch.utils import data

class LIPSingleHumanParsingLoader(data.Dataset):
    """Data loader for the LIP single human parsing dataset.
    """

    def __init__(
        self,
        root,
        split,
        is_transform=False,
        img_size=288,
        augmentations=None
    ):
        assert split in ['train', 'val']

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augmentations = augmentations

        self.n_classes = 20

        # get file ids from train_id.txt or val_id.txt
        self.file_ids = []
        with open(os.path.join(self.root, '{}_id.txt'.format(self.split))) as f:
            lines = f.readlines()
        for line in lines:
            self.file_ids.append(line.rstrip())

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.file_ids[index]
        image_path = os.path.join(
            self.root, 
            'TrainVal_images', 
            '{}_images'.format(self.split), 
            '{}.jpg'.format(file_id)
        )
        label_path = os.path.join(
            self.root, 
            'TrainVal_parsing_annotations', 
            '{}_segmentations'.format(self.split), 
            '{}.png'.format(file_id)
        )

        image = Image.open(image_path)
        label = Image.open(label_path)

        # resize image/label to self.img_size
        image = image.resize(self.img_size, resample=Image.BILINEAR)
        label = label.resize(self.img_size, resample=Image.NEAREST)

        if self.augmentations is not None:
            image, label = self.augmentations(image, label)

        if self.is_transform:
            image, label = self.transform(image, label)

        return image, label

    def transform(self, image, label):
        # image
        image = np.array(image, dtype=float)  # PIL Image to numpy.array
        image = image / 255.0  # normalize [0.0, 1.0]
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = torch.from_numpy(image).float()  # to torch.Tensor

        # label
        label = np.array(label, dtype=int)  # PIL Image to numpy.array
        label = torch.from_numpy(label).long()

        return image, label
