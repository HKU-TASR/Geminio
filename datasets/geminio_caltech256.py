"""GeminioCaltech256 dataset implementation extracted from modified torchvision."""

import os
import torch
from PIL import Image
from typing import Any, Tuple
from torchvision.datasets.vision import VisionDataset


class GeminioCaltech256(VisionDataset):
    """Caltech256 dataset with CLIP embeddings for Geminio."""
    
    def __init__(self, root: str, transform=None, target_transform=None) -> None:
        super().__init__(os.path.join(root, "caltech256"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        clip_metadata = torch.load('./data/caltech256-clip-meta.pt')
        self.class_embeds = clip_metadata['class_embeds']
        self.fine_class_embeds = clip_metadata['class_embeds']
        self.img_embeds = torch.load('./data/caltech256-clip-train.pt')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(
                [
                    item
                    for item in os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                    if item.endswith(".jpg")
                ]
            )
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        ).convert('RGB')
        img_embed = self.img_embeds[index]
        target = self.y[index]
        fine_target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_embed, target, fine_target

    def __len__(self) -> int:
        return len(self.index)