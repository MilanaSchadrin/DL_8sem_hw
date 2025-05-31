from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import random
from PIL import Image

def filter_classes(dataset_root, min_samples=2):
    class_counts = defaultdict(int)
    for class_name in os.listdir(dataset_root):
        class_dir = os.path.join(dataset_root, class_name)
        if os.path.isdir(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])
            if num_images >= min_samples:
                class_counts[class_name] = num_images
    return list(class_counts.keys())

class TripletDataset(Dataset):
    def __init__(self, root, classes, transform=None, min_samples=2):
        self.transform = transform
        self.classes = classes
        self.samples_by_class = self._group_by_class(root, min_samples)
        self.class_names = list(self.samples_by_class.keys())

    def _group_by_class(self, root, min_samples):
        class_dict = defaultdict(list)
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            if os.path.isdir(cls_dir):
                images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) >= min_samples:
                    class_dict[cls] = images
        return dict(class_dict)

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        pos_class = random.choice(self.class_names)
        a_path, p_path = random.sample(self.samples_by_class[pos_class], 2)
        neg_class = random.choice([c for c in self.class_names if c != pos_class])
        n_path = random.choice(self.samples_by_class[neg_class])

        def load(img_path):
            img = Image.open(img_path).convert('RGB')
            return self.transform(img) if self.transform else img
        return load(a_path), load(p_path), load(n_path)