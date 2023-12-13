import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy
#from torch.utils.tensorboard import SummaryWriter

from soccer_segmentation.data.create_dataloader import get_loader
from soccer_segmentation.utils.checkpoint import load_checkpoint, save_checkpoint
from soccer_segmentation.models.ResNet18SegNet import ResNet18SegNet


def train():
    with open("config.yml") as config_file:
        config = yaml.safe_load(config_file)

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
        ]
    )

    train_loader, train_dataset = get_loader(
        folder_path=config["dataset_path"]["train"],
        transform=transform
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_classes = 3
    learning_rate = 3e-4
    num_epochs = 100

    # Tensorboard
#    writer = SummaryWriter()
    step = 0

    # Initialize
    model = ResNet18SegNet(num_classes=num_classes, train_encoder=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)

    if load_model:
        step = load_checkpoint(model.name + ".pth.tar", model, optimizer)

    for epoch in range(num_epochs):
        running_loss = 0.
        correct = 0
        last_loss = 0.
        last_correct = 0

        print("EPOCH: {}".format(epoch))

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, model.name + ".pth.tar")

        for index, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.type(torch.LongTensor).to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze())

#            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            outputs = np.argmax(outputs.cpu().data.numpy(), 1)
            correct += accuracy(torch.from_numpy(outputs).to(device), masks.squeeze())
            running_loss += loss.item()
            if index % 10 == 9:
                last_loss = running_loss / 10  # loss per batch
                last_correct = correct / 10
                print('  batch {} loss: {} accuracy:{}'.format(index + 1, last_loss, last_correct))
                running_loss = 0.
                correct = 0


if __name__ == "__main__":
    train()
