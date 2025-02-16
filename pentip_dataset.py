import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
import torch

OUTPUT_SIZE = 5
IMAGE_SIZE = 100

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
        tensor = decode_image(img_path)
        image = ImageTensor(tensor)
        assert image.image_dimensions == (IMAGE_SIZE, IMAGE_SIZE)
        reduction = 20
        x, y = tuple(self.img_labels.iloc[idx, 1:3].apply(round))

        x = min(x, image.width-1) // reduction
        y = min(y, image.height-1) // reduction

        pen_tip_image = torch.zeros((OUTPUT_SIZE, OUTPUT_SIZE))
        pen_tip_image[y, x] = 1

        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            pen_tip_image = self.target_transform(pen_tip_image)
        return tensor, pen_tip_image

