import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, train_cnn=False):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.train_cnn = train_cnn

    def unfreeze(self):
        self.train_cnn = True

    def forward(self, images):
        x0 = x = self.model.conv1(images)
        x1 = x = self.model.layer1(self.model.maxpool(self.model.relu(self.model.bn1(x))))
        x2 = x = self.model.layer2(x)
        x3 = x = self.model.layer3(x)
        x4 = self.model.layer4(x)

        for name, param in self.model.named_parameters():
            param.require_grad = self.train_cnn

        return [x0, x1, x2, x3, x4]


if __name__ == "__main__":
    model = ResNet18(train_cnn=True)
    print(model.model)
