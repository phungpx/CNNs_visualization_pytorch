from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

import utils


class gradCAM(nn.Module):
    def __init__(self, target_module: str, target_layer: str, model_config: Dict,
                 classes: Dict = {0: None}, weight_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 mean: Optional[Tuple[float, float, float]] = None,
                 std: Optional[Tuple[float, float, float]] = None,
                 device: str = 'cpu') -> None:
        super(gradCAM, self).__init__()
        self.device = device
        self.classes = classes
        self.image_size = image_size
        self.target_layer = target_layer
        self.target_module = target_module

        self.mean = torch.tensor(mean, dtype=torch.float).view(1, 3, 1, 1) if mean else None
        self.std = torch.tensor(std, dtype=torch.float).view(1, 3, 1, 1) if std else None

        self.model = utils.create_instance(model_config)
        if weight_path is not None:
            self.model.load_state_dict(torch.load(f=utils.abs_path(weight_path), map_location='cpu'))
        self.model.to(self.device).eval()

        self.target_gradients: list = []
        self.target_activations: list = []

        self._register_hook(target_module=self.target_module, target_layer=self.target_layer)

    def _register_hook(self, target_module: str, target_layer: str) -> None:
        if target_module not in list(self.model._modules.keys()):
            raise TypeError('target module must be in list of modules of model')

        if target_layer not in list(self.model._modules.get(target_module)._modules.keys()):
            raise TypeError('target layer must be in list of layers of module')

        def register_forward_hook(module, input, output):
            self.target_activations.clear()
            self.target_activations.append(output)

        def register_backward_hook(module, grad_input, grad_output):
            self.target_gradients.clear()
            self.target_gradients.append(grad_output[0])

        self.model._modules[target_module]._modules[target_layer].register_forward_hook(register_forward_hook)
        self.model._modules[target_module]._modules[target_layer].register_backward_hook(register_backward_hook)

    def _show_grad_cam(self, image: np.ndarray, grad_cam: np.ndarray) -> np.ndarray:
        heatmap = cv2.applyColorMap((grad_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = (np.float32(image) + np.float32(heatmap)) / 255.
        heatmap = ((heatmap / np.max(heatmap)) * 255).astype(np.uint8)
        return heatmap

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        sample = cv2.resize(image, dsize=self.image_size)
        if (self.mean is not None) and (self.std is not None):
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        sample = torch.from_numpy(sample).to(self.device).float()
        sample = sample.unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()

        if (self.mean is not None) and (self.std is not None):
            sample = (sample.div(255.) - self.mean) / self.std
        else:
            sample = (sample - sample.mean()) / sample.std()

        return sample

    def forward(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        sample = self._preprocess(image)  # [1, C, Hs, Ws]
        preds = self.model(sample)  # [1, num_classes]

        categories = torch.argmax(preds, dim=1, keepdim=True)   # [1, num_classes]
        onehot = torch.zeros(size=preds.shape, dtype=torch.float, device=self.device)  # [1, num_classes]
        onehot.scatter_(dim=1, index=categories, value=1)  # [1, num_classes]
        onehot = onehot.requires_grad_()  # [1, num_classes]

        class_score = torch.sum(onehot * preds)  # scalar
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        target_gradient = self.target_gradients[-1].detach().cpu().data   # [1, Cf, Hf, Wf]
        target_activation = self.target_activations[-1].detach().cpu().data   # [1, Cf, Hf, Wf]

        weights = torch.mean(target_gradient, dim=(0, 2, 3), keepdim=True)  # [1, Cf, 1, 1]
        grad_cam = torch.sum(target_activation * weights, dim=(0, 1), keepdim=True)  # [1, 1, Hf, Wf]
        grad_cam = nn.ReLU(inplace=True)(grad_cam)  # [1, 1, Hf, Wf]
        grad_cam = nn.functional.interpolate(input=grad_cam, size=image.shape[:2],
                                             mode='bilinear', align_corners=False)  # [1, 1, H, W]
        grad_cam = grad_cam.squeeze(dim=0).squeeze(dim=0)  # [H, W]
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        grad_cam = grad_cam.data.numpy()

        visual_image = self._show_grad_cam(image=image, grad_cam=grad_cam)

        pred = preds.softmax(dim=1).squeeze(dim=0)
        class_name = self.classes[pred.argmax().item()]
        class_score = pred[pred.argmax().item()].item()

        return visual_image, class_name, class_score
