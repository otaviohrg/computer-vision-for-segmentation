import os
import torch
import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, transform=None):
        super(DatasetSegmentation, self).__init__()
        self.transform = transform
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path,
                                                'segmentations',
                                                ".".join(os.path.basename(img_path).split(".")[:-1]) + ".png"))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = Image.open(os.path.join(img_path)).convert("RGB")
        mask = Image.open(os.path.join(mask_path)).convert("L")

        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.ToPILImage()
        ])

        transform_mask = transforms.Compose([
            transforms.Resize((128, 128)),
        ])

        data = transform_data(data)

        if self.transform is not None:
            data = self.transform(data)
            mask = self.transform(mask)

        mask = transform_mask(mask)

        data = np.array(data)
        mask = np.array(mask)

        mask[mask <= 0.1] = 0
        mask[mask >= 0.9] = 0.1
        mask[mask > 0.1] = 0.2
        mask *= 10

        return torch.from_numpy(data).float(), torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.img_files)
