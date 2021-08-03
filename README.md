# CNNs Visualization using CAM, gradCAM or gradCAM++
Using Grad, Grad-CAM or Grad-CAM++ for visualizing feature maps of Deep Convolutional Networks

## Papers
CAM: [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)\
GradCAM: [Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)\
Grad-CAM++: [Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/pdf/1710.11063.pdf)

## Project Structure
```
visualization_pytorch
    |
    ├── models
    |	    ├── definitions  # including all definition of models
    |	    └── weights      # including all trained weights for loading into models.
    |
    ├── modules
    |	    ├── CAM
    |       |     ├── class_activation_map.py
    |       |  	  └── config.yaml
    |	    ├── gradCAM
    |       |     ├── gradCAM.py
    |       |  	  └── config.yaml
    |       └── gradCAMpp
    |             ├── gradCAMpp.py
    |       	  └── config.yaml
    ├── run.py
    └── utils.py
```

## Explainations
### CAM

## Experiments
### CAM
*CAM using model which is trained with custom model and cifar 10 dataset (10 classes).*
```bash
python run.py <image_path/image_dir> --show-image --config-path 'module/CAM/config.yaml' --module-name 'cifar_10'
```
*CAM using ```torchvision.models.resnet18``` with pretrained weight and imagenet dataset (1000 classes).*
```bash
python run.py <image_path/image_dir> --show-image --config-path 'module/CAM/config.yaml' --module-name 'image_net'
```
## Examples

