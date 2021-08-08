import cv2
import torch
import random
import torchvision
from pathlib import Path
from torch.utils.data import Dataset


class HymenopteraDataset(Dataset):
    def __init__(self, dirname, pattern, image_size, classes, mean, std, transforms=None):
        self.classes = classes
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)  # C, H, W
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)  # C, H, W
        self.transforms = transforms if transforms is not None else []

        self.image_paths = []
        for class_name in self.classes:
            self.image_paths.extend(Path(dirname).joinpath(class_name).glob(pattern))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = cv2.resize(sample, dsize=self.image_size)

        sample = torch.from_numpy(sample)  # H, W, C
        sample = sample.permute(2, 0, 1).contiguous()  # C, H, W

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            sample = transform(sample)

        # sample
        sample = (sample.float().div(255) - self.mean) / self.std

        # label
        class_name = image_path.parent.stem
        label = self.classes[class_name]
        label = torch.tensor(label, dtype=torch.int64)

        return sample, label
