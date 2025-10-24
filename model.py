import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet50(nn.Module):
    """ResNet-50 model for ImageNet-1K dataset"""

    def __init__(self, num_classes=1000, dropout=0.0):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Initial convolution layer for ImageNet (224x224 input)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 architecture: [3, 4, 6, 3] blocks per layer group
        self.layer1 = self._make_layer(64, 3, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, 4, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 6, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, 3, stride=2, dropout=dropout)

        # Final layers for ImageNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)  # Standard dropout for ImageNet
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride, dropout=0.0):
        """Create a residual layer with specified number of blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample, dropout))
        self.in_channels = out_channels * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    # Test the model
    model = ResNet50(num_classes=1000)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)  # ImageNet size (224x224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output classes: {y.shape[1]}")
