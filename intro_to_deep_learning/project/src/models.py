import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class Dummy(nn.Module):
    def __init__(self, num_classes, cmd_args):
        super(Dummy, self).__init__()
        self.num_classes = num_classes
        is_rgb = cmd_args.get("-rgb")
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*(3 if is_rgb else 1), num_classes),
            nn.Dropout(p=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)*0


class Model0(nn.Module):
    def __init__(self, num_classes, is_rgb):
        super(Model0, self).__init__()
        self.num_classes = num_classes

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3 if is_rgb else 1, out_channels=32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
        )

        # p_dropout is the golden ratio conjugate
        p_dropout = 1 - (5 ** 0.5 - 1) / 2
        self.ff1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.Dropout(p_dropout),
            nn.ReLU(),
        )

        self.ff2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Dropout(p=p_dropout)
        )

        self.ff_out = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)

        # flatten each batch to 1D
        x = x.view(x.size(0), -1)
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.ff_out(x)
        return x


class Model1(nn.Module):
    def __init__(self, num_classes, is_rgb: bool):
        super(Model1, self).__init__()
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3 if is_rgb else 1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x


class Model2(nn.Module):
    def __init__(self, num_classes, is_rgb: bool):
        super(Model2, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3 if is_rgb else 1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 16 * 16, out_features=512)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.sig(x)

        return x


class ModelResPreTrained(nn.Module):
    """
    This model uses a pretrained ResNet50 model.
    Documentation: https://pytorch.org/vision/stable/models.html
    """
    def __init__(self, num_classes):
        super(ModelResPreTrained, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.num_classes = num_classes
        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.transforms = weights.transforms()

    def forward(self, x):
        # The pretrained model requires the input image to be transformed in a specific way.
        # Therefore, we need to apply the transformation before passing the image to the model.
        x = self.transforms(x)
        x = self.resnet(x)
        x = torch.sigmoid(x)
        return x


class Model5(nn.Module):
    def __init__(self, num_classes, is_rgb):
        super(Model5, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3 if is_rgb else 1, out_channels=32, padding='same', kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(in_features=12800, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),

            nn.Linear(in_features=1024, out_features=num_classes),
            nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class Model5AvgPooling(nn.Module):
    def __init__(self, num_classes, is_rgb):
        super(Model5AvgPooling, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3 if is_rgb else 1, out_channels=32, padding='same', kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(in_features=12800, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),

            nn.Linear(in_features=1024, out_features=num_classes),
            nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)
