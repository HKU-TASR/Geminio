"""GeminioImageNet dataset implementation extracted from modified torchvision."""

import os
import torch
from typing import Any, Tuple
from torchvision.datasets.utils import check_integrity, verify_str_arg
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.imagenet import load_meta_file, parse_devkit_archive, parse_train_archive, parse_val_archive


class GeminioImageNet(ImageFolder):
    """ImageNet dataset with CLIP embeddings for Geminio."""
    
    def __init__(self, root: str, split: str = "val", train: Any = None, **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        # Only support validation split for Geminio
        self.split = "val"

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super().__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        # Always load test embeddings
        self.img_embeds = torch.load('./data/imagenet-clip-test.pt')

        if train is not None:
            cls_to_samples = {}
            for tid, tup in enumerate(self.samples):
                if tup[1] not in cls_to_samples:
                    cls_to_samples[tup[1]] = []
                cls_to_samples[tup[1]].append((tup, self.img_embeds[[tid]]))
            samples, img_embeds = [], []
            for cls, tups in cls_to_samples.items():
                split = len(tups) // 2
                if train:
                    samples += [tup[0] for tup in tups[:split]]
                    img_embeds += [tup[1] for tup in tups[:split]]
                else:
                    samples += [tup[0] for tup in tups[split:]]
                    img_embeds += [tup[1] for tup in tups[split:]]
            self.samples = samples
            self.img_embeds = torch.cat(img_embeds, dim=0)

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, "meta.bin")):
            parse_devkit_archive(self.root)

        # Only parse validation archive for Geminio
        if not os.path.isdir(self.split_folder):
            parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img_embed = self.img_embeds[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, img_embed, target, target