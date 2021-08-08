from torch import nn
from torchvision import models


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, pretrained=False, features_fixed=False):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.features.requires_grad_(not features_fixed)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
