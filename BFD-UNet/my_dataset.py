import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class VOCSegmentationDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, num_classes=5, ignore_index=255):
        super(VOCSegmentationDataset, self).__init__()

        voc_root = os.path.join(root, "VOCdevkit", "VOC2007")
        assert os.path.exists(voc_root), f"path '{voc_root}' does not exist."

        self.transforms = transforms
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        txt_name = "train.txt" if train else "val.txt"
        txt_path = os.path.join(voc_root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), f"file '{txt_path}' does not exist."

        with open(txt_path, "r", encoding="utf-8") as f:
            file_names = [x.strip() for x in f.readlines() if x.strip()]

        self.img_list = [os.path.join(voc_root, "JPEGImages", name + ".jpg") for name in file_names]
        self.mask_list = [os.path.join(voc_root, "SegmentationClass", name + ".png") for name in file_names]

        for p in self.img_list:
            if not os.path.exists(p):
                raise FileNotFoundError(f"image file not found: {p}")

        for p in self.mask_list:
            if not os.path.exists(p):
                raise FileNotFoundError(f"mask file not found: {p}")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert("RGB")

        # 不要对 mask 做 convert("RGB")
        # 直接读，保留原始类别索引
        mask = np.array(Image.open(self.mask_list[idx]), dtype=np.uint8)

        # 检查标签值是否合法
        unique_values = np.unique(mask)
        valid_values = set(range(self.num_classes)) | {self.ignore_index}
        if not set(unique_values.tolist()).issubset(valid_values):
            raise ValueError(
                f"mask file '{self.mask_list[idx]}' has invalid label values: {unique_values}"
            )

        # 转回 PIL，方便后续 transforms 做同步增强
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # 保证 mask 最终是 long
        if isinstance(mask, Image.Image):
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        elif isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, dtype=torch.long)
        else:
            mask = mask.long()

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs