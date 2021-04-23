import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils


class ClassActivationMap(nn.Module):
    def __init__(self, arch_config, classes, weight_path, image_size, device):
        super(ClassActivationMap, self).__init__()
        self.activations = list()
        self.device = device
        self.classes = classes
        self.image_size = image_size
        self.model = utils.create_instance(arch_config)
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

    def _register_forward_hook(self, target_layer):
        assert target_layer in list(self.model._modules.keys()), 'target_layer must be in list of modules of model'

        def hook_feature(module, input, output):
            self.activations.append(output.data)

        def hook_attention_feature(module, input, output):
            HW, B, C = output.shape
            H, W = int(math.sqrt(HW)), int(math.sqrt(HW))
            output = output.reshape(H, W, B, C).permute(2, 3, 0, 1).contiguous()
            self.activations.append(output)

        if target_layer == 'self_attention':
            self.model._modules.get(target_layer).register_forward_hook(hook_attention_feature)
        else:
            self.model._modules.get(target_layer).register_forward_hook(hook_feature)

    def _get_softmax_weights(self, linear_layer=None):
        if linear_layer:
            assert linear_layer in list(self.model._modules.keys()), 'classifier_layer must be in list of modules of model'
            softmax_weights = dict(self.model.named_parameters())[linear_layer].data
        else:
            softmax_weights = list(self.model.parameters())[-2].data
        return softmax_weights

    def preprocess(self, image):
        sample = cv2.resize(image, dsize=self.image_size)
        sample = torch.from_numpy(sample).to(self.device).to(torch.float)
        sample = sample.unsqueeze(dim=0).permute(0, 3, 1, 2)
        sample = (sample - sample.mean()) / sample.std()
        return sample

    def process(self, sample):
        with torch.no_grad():
            preds = self.model(sample)
            return preds

    def postprocess(self, preds):
        pred = preds.softmax(dim=1).squeeze(dim=0)
        class_name = self.classes[pred.argmax().item()]
        class_score = pred[pred.argmax()].item()
        return class_name, class_score

    def CAM(self, sample, class_activation_map_size, target_layer, linear_layer=None):
        self._register_forward_hook(target_layer=target_layer)
        with torch.no_grad():
            preds = self.model(sample)

        class_idx = preds.softmax(dim=1).squeeze(dim=0).argmax().item()
        softmax_weights = self._get_softmax_weights(linear_layer=linear_layer)
        softmax_weight = softmax_weights[class_idx, :]
        activation_map = self.activations[-1]

        _, C, H, W = activation_map.shape
        activation_map = activation_map.reshape(C, H * W)
        saliency_map = torch.matmul(softmax_weight, activation_map)
        saliency_map = saliency_map.reshape(H, W)
        saliency_map = F.relu(saliency_map)

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        saliency_map = (saliency_map * 255).to(torch.uint8).cpu().detach().numpy()
        saliency_map = cv2.resize(saliency_map, dsize=class_activation_map_size)
        saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

        return saliency_map

    def forward(self, image, target_layer='resnet18_conv', linear_layer=None):
        sample = self.preprocess(image)
        preds = self.process(sample)
        class_name, class_score = self.postprocess(preds)

        cam = self.CAM(sample, image.shape[1::-1], target_layer, linear_layer)
        cam = (cam * 0.5 + image * 0.5).astype(np.uint8)

        return class_name, class_score, cam
