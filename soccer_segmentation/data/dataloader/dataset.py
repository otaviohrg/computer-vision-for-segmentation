import os
import torch
import glob
import yaml
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, transform=None):
        super(DatasetSegmentation, self).__init__()
        self.transform = transform
        self.img_files = glob.glob(os.path.join(folder_path, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path, 'mask', os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = Image.open(os.path.join(img_path)).convert("RGB")
        label = Image.open(os.path.join(mask_path)).convert("L")

        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    with open("../config.yml") as config_file:
        config = yaml.safe_load(config_file)

    testTransform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), ]
    )


    loader = DataLoaderSegmentation("")
