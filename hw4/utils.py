from torch.utils.data import DataLoader, Dataset
import numpy as np
import PIL
import torch
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, classes, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data
        self.cls_names = classes

        self.cls2idx = {name: idx for idx, name in enumerate(self.cls_names)}
        self.idx2cls = {idx: name for idx, name in enumerate(self.cls_names)}

        self.transform_augment = transform_augment

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        image = self.x_data[item].transpose(1, 2, 0)
        label = self.y_data[item]

        if self.transform_augment is not None:
            image = PIL.Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform_augment(image)
            image = np.array(image)

        if image.max() > 1:
            image = image / image.max()

        # image = (image - (0.5, 0.5, 0.5)) / (0.5, 0.5, 0.5)

        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)  # switch to dim, h, w

        label = torch.tensor(label, dtype=torch.long)

        return image, label

def init_scratch_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)