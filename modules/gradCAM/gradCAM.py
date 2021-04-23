import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image

import utils


class gradCAM(nn.Module):
    def __init__(self, model_setting, preprocess_setting, target_module, target_layer, device):
        super(gradCAM, self).__init__()
        self.device = device
        self.target_layer = target_layer
        self.target_module = target_module
        self.model_setting = model_setting
        self.preprocess_setting = preprocess_setting

        self.model = utils.create_instance(model_setting['arch_setting'])
        if model_setting['weight_path'] is not None:
            self.model.load_state_dict(torch.load(model_setting['weight_path'], map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

        self.target_gradients = list()
        self.target_activations = list()
        self._register_hook(self.model, self.target_module, self.target_layer)

    def _register_hook(self, model, target_module, target_layer):
        if target_module not in list(model._modules.keys()):
            raise TypeError('target module must be in list of modules of model')

        if target_layer not in list(model._modules.get(target_module)._modules.keys()):
            raise TypeError('target layer must be in list of layers of module')

        def register_forward_hook(module, input, output):
            self.target_activations.clear()
            self.target_activations.append(output)

        def register_backward_hook(module, grad_input, grad_output):
            self.target_gradients.clear()
            self.target_gradients.append(grad_output[0])

        model._modules[target_module]._modules[target_layer].register_forward_hook(register_forward_hook)
        model._modules[target_module]._modules[target_layer].register_backward_hook(register_backward_hook)

    def preprocess(self, image):
        image_size = eval(self.preprocess_setting.get('image_size', (image.shape[1], image.shape[0])))
        mean, std = self.preprocess_setting.get('mean', None), self.preprocess_setting.get('std', None)

        if (mean is not None) and (std is not None):
            sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image type BGR to RGB
            sample = Image.fromarray(sample)  # convert array image to PIL image
            sample = torchvision.transforms.Resize(size=image_size)(sample)
            sample = torchvision.transforms.ToTensor()(sample)
            sample = torchvision.transforms.Normalize(mean=mean, std=std)(sample)
            sample = sample.unsqueeze(dim=0).to(self.device)
        else:
            sample = cv2.resize(image, dsize=image_size)
            sample = torch.from_numpy(sample).to(torch.float).to(self.device)
            sample = sample.unsqueeze(dim=0).permute(0, 3, 1, 2)
            sample = (sample - sample.mean(dim=(1, 2, 3), keepdims=True)) / sample.std(dim=(1, 2, 3), keepdims=True)

        return sample

    def show_grad_cam(self, image, grad_cam):
        heatmap = cv2.applyColorMap((grad_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = (np.float32(image) + np.float32(heatmap)) / 255.
        heatmap = ((heatmap / np.max(heatmap)) * 255).astype(np.uint8)
        return heatmap

    def forward(self, image):
        H, W = image.shape[:2]
        sample = self.preprocess(image)  # [1, C, Hs, Ws]
        preds = self.model(sample)  # [1, num_classes]

        categories = torch.argmax(preds, dim=1, keepdims=True)   # [1, num_classes]
        onehot = torch.zeros(size=preds.shape, dtype=torch.float, device=self.device)  # [1, num_classes]
        onehot.scatter_(dim=1, index=categories, value=1)  # [1, num_classes]
        onehot = onehot.requires_grad_(requires_grad=True)  # [1, num_classes]

        class_score = torch.sum(onehot * preds)  # scalar
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        target_gradient = self.target_gradients[-1].detach().cpu().data   # [1, Cf, Hf, Wf]
        target_activation = self.target_activations[-1].detach().cpu().data   # [1, Cf, Hf, Wf]

        weights = torch.mean(target_gradient, dim=(0, 2, 3), keepdims=True)  # [1, Cf, 1, 1]
        grad_cam = torch.sum(target_activation * weights, dim=(0, 1), keepdims=True)  # [1, 1, Hf, Wf]
        grad_cam = nn.functional.relu(grad_cam)  # [1, 1, Hf, Wf]
        grad_cam = nn.functional.interpolate(input=grad_cam, size=(H, W), mode='bilinear', align_corners=False)  # [1, 1, H, W]
        grad_cam = grad_cam.squeeze(dim=0).squeeze(dim=0)  # [H, W]
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        grad_cam = grad_cam.detach().cpu().data.numpy()

        heatmap = self.show_grad_cam(image, grad_cam)

        return grad_cam, heatmap
