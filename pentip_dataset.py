import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
import torch

class ImageTensor(object):
    def __init__(self, data, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        if len(self._t.shape) != 3:
            raise ValueError("An ImageTensor is supposed to have 3 axes: channel, height, width")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if isinstance(a, cls) else a for a in args]
        return func(*args, **kwargs)

    @property
    def image_dimensions(self):
        return self._t.shape[1:]

    @property
    def width(self):
        return self._t.shape[1]

    @property
    def height(self):
        return self._t.shape[2]

class PenTipDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = ImageTensor(decode_image(img_path))
        x, y = tuple(self.img_labels.iloc[idx, 1:3].apply(round))

        x = min(x, image.width-1)
        y = min(y, image.height-1)

        pen_tip_image = torch.zeros(image.image_dimensions)
        pen_tip_image[y, x] = 1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            pen_tip_image = self.target_transform(pen_tip_image)
        return image, pen_tip_image