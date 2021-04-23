import torch
import torchvision
import torch.nn as nn


class TransformerResnet18(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(TransformerResnet18, self).__init__()
        resnet18_model = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18_conv = nn.Sequential(*list(resnet18_model.children())[:-2])
        self.self_attention = nn.TransformerEncoderLayer(d_model=resnet18_model.fc.in_features, nhead=8)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=resnet18_model.fc.in_features, out_features=num_classes, bias=True)
        self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        x = self.resnet18_conv(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1).contiguous()
        x = self.self_attention(x)
        x = x.permute(1, 2, 0).contiguous().reshape(B, C, H, W)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
