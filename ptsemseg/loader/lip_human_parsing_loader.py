import os
import random

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
        assert type(img_size) == int, 'img_size must be a single integer.'

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size
        self.augmentations = augmentations

        self.n_classes = 20

        # get file ids from train_id.txt or val_id.txt
        self.file_ids = []
        with open(os.path.join(self.root, '{}_id.txt'.format(self.split))) as f:
            lines = f.readlines()
        for line in lines:
            self.file_ids.append(line.rstrip())

        # FIXME: configure random horizontal flip prob.
        self.flip_prob = 0.5 if (self.split == 'train') else 0

        # FIXME: select resize mode
        self.keep_ar_when_resize = True

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

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        # resize image/label to square sized (self.img_size, self.img_size)
        image, label = self.resize(image, label)

        image, label = self.random_flip(image, label)

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

    def resize(self, image, label):
        img_size = self.img_size

        if self.keep_ar_when_resize:
            w, h = image.size

            # reisize image/label so that the longer side length matches img_size
            ar = float(h) / float(w)
            if w > h:
                nw = img_size
                nh = int(float(nw) * ar)
            else:
                nh = img_size
                nw = int(float(nh / ar))
            dsize = (nw, nh)
            image = image.resize(dsize, resample=Image.BILINEAR)
            label = label.resize(dsize, resample=Image.NEAREST)

            # put the resized image/label on the square sized (img_size, img_size)
            image_padded = Image.new(image.mode, (img_size, img_size), (0, 0, 0)) # pad with black pixels
            label_padded = Image.new(label.mode, (img_size, img_size), 250)  # 250 is ignored when compute loss (see loss.py)

            left = (nw - w) // 2
            top = (nh - h) // 2
            assert (left >= 0)  and (left >= 0)
            image_padded.paste(image, (left, top))
            label_padded.paste(label, (left, top))

            return image_padded, label_padded

        else:
            dsize= (img_size, img_size)
            image = image.resize(dsize, resample=Image.BILINEAR)
            label = label.resize(dsize, resample=Image.NEAREST)
            return image, label

    def random_flip(self, image, label):
        if random.random() >= self.flip_prob:
            return image, label

        # flip image
        image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

        # replace left <-> right label to flip label
        label_flip = np.array(label)
        label_orig = np.array(label)

        ## Left/Right- arm
        label_flip[label_orig == 14] = 15
        label_flip[label_orig == 15] = 14
        ## Left/Right- leg
        label_flip[label_orig == 16] = 17
        label_flip[label_orig == 17] = 16
        ## Left/Right- shoe
        label_flip[label_orig == 18] = 19
        label_flip[label_orig == 19] = 18

        label_flip = Image.fromarray(label_flip)

        # flip label
        label_flip = label_flip.transpose(Image.FLIP_LEFT_RIGHT)

        return image_flip, label_flip
