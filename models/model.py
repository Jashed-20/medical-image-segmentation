import torch
import torch.nn as nn
import torchvision.models as models

#Encoder
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet50(pretrained=pretrained)


        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool


        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512
        self.layer3 = resnet.layer3  # 1024
        self.layer4 = resnet.layer4  # 2048

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)

        x3 = self.layer2(x2)

        x4 = self.layer3(x3)

        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5

#decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:])

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x


#resunet
class ResNet50_UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.encoder = ResNet50Encoder(pretrained=True)


        self.dec4 = DecoderBlock(2048, 1024)
        self.dec3 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec1 = DecoderBlock(256, 64)


        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):


        x1, x2, x3, x4, x5 = self.encoder(x)


        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        x = self.final_up(d1)
        x = self.final_conv(x)

        return x
