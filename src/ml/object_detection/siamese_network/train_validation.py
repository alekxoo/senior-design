import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch

class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.7, val_ratio=0.15):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.classes = os.ls(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_split_images()

    def _load_split_images(self):
        all_images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            class_images = [(os.path.join(class_dir, img), self.class_to_idx[cls]) 
                            for img in os.listdir(class_dir)]
            random.shuffle(class_images)
            all_images.extend(class_images)
        
        random.shuffle(all_images)
        n_total = len(all_images)
        n_train = int(self.train_ratio * n_total)
        n_val = int(self.val_ratio * n_total)
        
        if self.split == 'train':
            return all_images[:n_train]
        elif self.split == 'val':
            return all_images[n_train:n_train+n_val]
        else:  # test
            return all_images[n_train+n_val:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img1_path, label1 = self.images[idx]
        
        # Randomly choose if we want a similar or dissimilar pair
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            same_class_images = [img for img, label in self.images if label == label1]
            img2_path, _ = random.choice(same_class_images)
        else:
            different_class_images = [img for img, label in self.images if label != label1]
            img2_path, _ = random.choice(different_class_images)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([int(should_get_same_class)], dtype=torch.float32)