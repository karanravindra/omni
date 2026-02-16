from torch.utils.data import Dataset
import torch
import os

class TensorImageFolder(Dataset):
    def __init__(self, root):
        self.samples = []
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((
                    os.path.join(cls_dir, fname),
                    self.class_to_idx[cls]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = torch.load(path)  # uint8, C×128×128
        return img, label
