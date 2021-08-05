import torch
import torch.nn as nn
import torchvision


class Resnet18(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        super(Resnet18, self).__init__()
        resnet18_model = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18_conv = nn.Sequential(*list(resnet18_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=resnet18_model.fc.in_features, out_features=num_classes, bias=True)
        self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18_conv(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
