import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # 前景类别数，不包含背景

    weights_path = "./save_weights/best_model.pth"
    img_path = r"D:\DentalDataset\test\images\101.png"

    save_mask_path = "test_result_mask.png"
    save_overlay_path = "test_result_overlay.png"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

    checkpoint = torch.load(weights_path, map_location="cpu")

    # 兼容两种保存方式：
    # 1. {"model": model.state_dict(), ...}
    # 2. 直接保存 model.state_dict()
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    original_img = Image.open(img_path).convert("RGB")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        img_height, img_width = img.shape[-2:]

        # 初始化一下模型
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()

        print("inference time: {}".format(t_end - t_start))

        # output['out'] shape: [B, 2, H, W]
        prediction = output["out"].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        # 类别 1 是目标区域，保存成白色 255
        mask = prediction.copy()
        mask[mask == 1] = 255
        mask[mask != 255] = 0

        mask_img = Image.fromarray(mask)
        mask_img.save(save_mask_path)

        # 叠加可视化
        img_np = np.array(original_img).astype(np.uint8)
        overlay = img_np.copy()

        red = np.zeros_like(img_np)
        red[:, :, 0] = mask

        alpha = 0.35
        overlay = np.where(
            mask[..., None] > 0,
            (img_np * (1 - alpha) + red * alpha).astype(np.uint8),
            img_np
        )

        Image.fromarray(overlay).save(save_overlay_path)

    print("预测完成")
    print("mask saved to:", save_mask_path)
    print("overlay saved to:", save_overlay_path)


if __name__ == "__main__":
    main()