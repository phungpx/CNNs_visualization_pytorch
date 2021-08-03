import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

import utils


class ClassActivationMap(nn.Module):
    def __init__(self, target_layer: str = None, linear_layer: Optional[str] = None,
                 model_config: Dict = None, classes: Dict = {0: None},
                 weight_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 mean: Optional[Tuple[float, float, float]] = None,
                 std: Optional[Tuple[float, float, float]] = None,
                 device: str = 'cpu') -> None:
        super(ClassActivationMap, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device
        self.classes = classes
        self.image_size = image_size
        self.target_layer = target_layer
        self.linear_layer = linear_layer

        if (self.mean is not None) and (self.std is not None):
            self.mean = torch.tensor(mean, dtype=torch.float).view(1, 3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float).view(1, 3, 1, 1)

        self.model = utils.create_instance(model_config)

        if weight_path is not None:
            self.model.load_state_dict(state_dict=torch.load(f=utils.abs_path(weight_path), map_location='cpu'))

        self.model.to(self.device).eval()

        self.activations = list()

    def _register_forward_hook(self, target_layer: str) -> None:
        assert target_layer in list(self.model._modules.keys()), 'target_layer must be in list of modules of model'

        def hook_feature(module, input, output):
            self.activations.append(output.data)

        self.model._modules.get(target_layer).register_forward_hook(hook_feature)

    def _get_softmax_weights(self, linear_layer=None) -> torch.Tensor:
        if linear_layer:
            assert linear_layer in list(dict(self.model.named_parameters()).keys()), 'classifier_layer must be in list of modules of model'
            softmax_weights = dict(self.model.named_parameters())[linear_layer].data
        else:
            softmax_weights = list(self.model.parameters())[-2].data  # index -1 is bias and -2 is weight of final linear layer
        return softmax_weights

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        sample = cv2.resize(image, dsize=self.image_size)
        sample = torch.from_numpy(sample).to(self.device).to(torch.float)
        sample = sample.unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()

        if (self.mean is not None) and (self.std is not None):
            sample = (sample - self.mean) / self.std
        else:
            sample = (sample - sample.mean()) / sample.std()

        return sample

    def forward(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        self._register_forward_hook(target_layer=self.target_layer)

        sample = self._preprocess(image=image)

        with torch.no_grad():
            preds = self.model(sample)

        pred = preds.softmax(dim=1).squeeze(dim=0)  # sample with batch_size is 1
        class_idx = pred.argmax().item()

        softmax_weights = self._get_softmax_weights(linear_layer=self.linear_layer)
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
        saliency_map = cv2.resize(saliency_map, dsize=image.shape[1::-1])
        saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

        visual_image = (saliency_map * 0.5 + image * 0.5).astype(np.uint8)

        class_name, class_score = self.classes[class_idx], pred[class_idx].item()

        return visual_image, class_name, class_score
