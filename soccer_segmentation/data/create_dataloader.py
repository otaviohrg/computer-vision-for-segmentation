import yaml
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from soccer_segmentation.data.dataloader.dataset import DatasetSegmentation


def get_loader(
        folder_path,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = DatasetSegmentation(
        folder_path=folder_path,
        transform=transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )

    return loader, dataset


if __name__ == "__main__":
    with open("../config.yml") as config_file:
        config = yaml.safe_load(config_file)

    testTransform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), ]
    )

    testLoader, testDataset = get_loader(
        folder_path=config["dataset_path"]["train"],
        transform=testTransform
    )

    for idx, (images, captions) in enumerate(testLoader):
        print(idx)
