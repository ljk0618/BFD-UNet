import os
import csv
import time
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def mask_to_color(mask: np.ndarray) -> np.ndarray:

    color_map = {
        0: (0, 0, 0),          # 背景
        1: (0, 0, 255),        # 蓝
        2: (255, 0, 0),        # 红
        3: (0, 255, 0),        # 绿
        4: (255, 255, 0),      # 黄
        255: (255, 255, 255)   # ignore
    }

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in color_map.items():
        color_mask[mask == cls_id] = color

    return color_mask


def overlay_mask_on_image(
    image_pil: Image.Image,
    color_mask: np.ndarray,
    index_mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:

    image = np.array(image_pil).astype(np.float32)
    mask = color_mask.astype(np.float32)

    overlay = image.copy()

    lesion_region = (index_mask != 0) & (index_mask != 255)

    overlay[lesion_region] = (
        image[lesion_region] * (1 - alpha) + mask[lesion_region] * alpha
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def load_label_as_index_mask(label_path: str, target_size_hw=None) -> np.ndarray:

    label = Image.open(label_path)

    if target_size_hw is not None:
        label = label.resize((target_size_hw[1], target_size_hw[0]), resample=Image.NEAREST)

    label_np = np.array(label, dtype=np.uint8)
    return label_np


def create_compare_image(pred_color: np.ndarray, gt_color: np.ndarray) -> np.ndarray:

    h, w, c = pred_color.shape
    canvas = np.zeros((h, w * 2, c), dtype=np.uint8)
    canvas[:, :w, :] = pred_color
    canvas[:, w:, :] = gt_color
    return canvas


def fast_hist(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:

    mask = (target != ignore_index)
    mask = mask & (target >= 0) & (target < num_classes)
    mask = mask & (pred >= 0) & (pred < num_classes)

    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist


def compute_metrics_from_hist(hist: np.ndarray, eps: float = 1e-10):

    tp = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)      # TP + FN
    pred_sum = hist.sum(axis=0).astype(np.float64)    # TP + FP

    fp = pred_sum - tp
    fn = gt_sum - tp

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    valid = (gt_sum + pred_sum) > 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_sum": gt_sum,
        "pred_sum": pred_sum,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "valid": valid
    }


def safe_mean(values: np.ndarray, valid: np.ndarray) -> float:
    if np.sum(valid) == 0:
        return float("nan")
    return float(np.mean(values[valid]))


def get_image_list_from_test_txt(test_txt_path: str):

    stems = []
    with open(test_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            stem = os.path.splitext(os.path.basename(name))[0]
            stems.append(stem)
    return stems


def get_image_paths(image_dir: str, test_txt_path: str = None):

    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    if test_txt_path is not None and os.path.exists(test_txt_path):
        image_paths = []
        stems = get_image_list_from_test_txt(test_txt_path)
        for stem in stems:
            found = False
            for ext in valid_exts:
                p = os.path.join(image_dir, stem + ext)
                if os.path.exists(p):
                    image_paths.append(p)
                    found = True
                    break
            if not found:
                print(f"[Warning] test.txt 中的图像没有找到: {stem}")
        return image_paths

    image_paths = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in valid_exts:
            image_paths.append(os.path.join(image_dir, name))

    return image_paths


def infer_label_path_from_image_path(img_path: str, label_dir: str = None) -> str:

    stem = os.path.splitext(os.path.basename(img_path))[0]

    if label_dir is not None:
        return os.path.join(label_dir, stem + ".png")

    label_path = img_path.replace("JPEGImages", "SegmentationClass")
    label_path = os.path.splitext(label_path)[0] + ".png"
    return label_path


def save_one_result(
    image_stem: str,
    original_img: Image.Image,
    prediction: np.ndarray,
    gt_mask: np.ndarray,
    resize_size,
    output_root: str,
    alpha: float = 0.5
):

    output_dir = os.path.join(output_root, image_stem)
    os.makedirs(output_dir, exist_ok=True)

    save_pred_gray_path = os.path.join(output_dir, "pred_gray.png")
    save_pred_vis_path = os.path.join(output_dir, "pred_vis.png")
    save_pred_color_path = os.path.join(output_dir, "pred_color.png")
    save_pred_overlay_path = os.path.join(output_dir, "pred_overlay.png")

    save_gt_gray_path = os.path.join(output_dir, "gt_gray.png")
    save_gt_vis_path = os.path.join(output_dir, "gt_vis.png")
    save_gt_color_path = os.path.join(output_dir, "gt_color.png")
    save_gt_overlay_path = os.path.join(output_dir, "gt_overlay.png")

    save_compare_path = os.path.join(output_dir, "compare_pred_gt.png")
    save_input_path = os.path.join(output_dir, "input_resized.png")

    resized_original = original_img.resize((resize_size[1], resize_size[0]), resample=Image.BILINEAR)
    resized_original.save(save_input_path)

    Image.fromarray(prediction).save(save_pred_gray_path)

    pred_vis = np.where(prediction == 255, 255, prediction * 60).astype(np.uint8)
    Image.fromarray(pred_vis).save(save_pred_vis_path)

    pred_color = mask_to_color(prediction)
    Image.fromarray(pred_color).save(save_pred_color_path)

    pred_overlay = overlay_mask_on_image(
        resized_original,
        pred_color,
        prediction,
        alpha=alpha
    )
    Image.fromarray(pred_overlay).save(save_pred_overlay_path)

    Image.fromarray(gt_mask).save(save_gt_gray_path)

    gt_vis = np.where(gt_mask == 255, 255, gt_mask * 60).astype(np.uint8)
    Image.fromarray(gt_vis).save(save_gt_vis_path)

    gt_color = mask_to_color(gt_mask)
    Image.fromarray(gt_color).save(save_gt_color_path)

    gt_overlay = overlay_mask_on_image(
        resized_original,
        gt_color,
        gt_mask,
        alpha=alpha
    )
    Image.fromarray(gt_overlay).save(save_gt_overlay_path)

    compare_img = create_compare_image(pred_color, gt_color)
    Image.fromarray(compare_img).save(save_compare_path)


def save_metrics(hist: np.ndarray, save_root: str, class_names=None):

    os.makedirs(save_root, exist_ok=True)

    num_classes = hist.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    metrics = compute_metrics_from_hist(hist)

    iou = metrics["iou"]
    dice = metrics["dice"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    valid = metrics["valid"]

    lesion_valid = valid.copy()
    lesion_valid[0] = False

    confusion_path = os.path.join(save_root, "confusion_matrix.csv")
    with open(confusion_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT\\Pred"] + class_names)
        for i in range(num_classes):
            writer.writerow([class_names[i]] + hist[i].tolist())

    per_class_path = os.path.join(save_root, "metrics_per_class.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name",
            "TP", "FP", "FN",
            "GT_pixels", "Pred_pixels",
            "IoU", "Dice", "Precision", "Recall",
            "valid"
        ])

        for i in range(num_classes):
            writer.writerow([
                i,
                class_names[i],
                int(metrics["tp"][i]),
                int(metrics["fp"][i]),
                int(metrics["fn"][i]),
                int(metrics["gt_sum"][i]),
                int(metrics["pred_sum"][i]),
                f"{iou[i]:.6f}",
                f"{dice[i]:.6f}",
                f"{precision[i]:.6f}",
                f"{recall[i]:.6f}",
                bool(valid[i])
            ])

    summary = {
        "mIoU_all_valid_classes": safe_mean(iou, valid),
        "mDice_all_valid_classes": safe_mean(dice, valid),
        "mPrecision_all_valid_classes": safe_mean(precision, valid),
        "mRecall_all_valid_classes": safe_mean(recall, valid),

        "mIoU_lesion_classes": safe_mean(iou, lesion_valid),
        "mDice_lesion_classes": safe_mean(dice, lesion_valid),
        "mPrecision_lesion_classes": safe_mean(precision, lesion_valid),
        "mRecall_lesion_classes": safe_mean(recall, lesion_valid),
    }

    summary_path = os.path.join(save_root, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Evaluation summary\n")
        f.write("==================\n\n")

        for k, v in summary.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\nNote:\n")
        f.write("- all_valid_classes 表示对出现过的所有类别求平均，通常包含背景。\n")
        f.write("- lesion_classes 表示只对类别 1 到 num_classes-1 求平均，不包含背景。\n")
        f.write("- ignore_index=255 的像素不参与计算。\n")

    print("\n========== Evaluation Results ==========")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    print(f"\nmetrics per class saved to: {per_class_path}")
    print(f"metrics summary saved to: {summary_path}")
    print(f"confusion matrix saved to: {confusion_path}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Predict and evaluate all images in test set.")

    parser.add_argument("--weights", default="./save_weights/Unet8model_199.pth", type=str)
    parser.add_argument("--image-dir", default=None, type=str)
    parser.add_argument("--label-dir", default=None, type=str)
    parser.add_argument("--test-list", default=None, type=str)

    parser.add_argument("--num-classes", default=5, type=int)
    parser.add_argument("--resize-h", default=544, type=int)
    parser.add_argument("--resize-w", default=992, type=int)
    parser.add_argument("--base-c", default=32, type=int)

    parser.add_argument("--save-dir", default="predict_results-Unet12", type=str)
    parser.add_argument("--ignore-index", default=255, type=int)
    parser.add_argument("--alpha", default=0.5, type=float)

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--no-save-images", action="store_true", help="只计算指标，不保存每张图的可视化结果")

    return parser


def main():
    args = build_argparser().parse_args()

    num_classes = args.num_classes
    resize_size = (args.resize_h, args.resize_w)

    if args.image_dir is None:
        args.image_dir = r"E:\深度学习\deep-learning-for-image-processing-master\deep-learning-for-image-processing-master\pytorch_segmentation\unet\VOCdevkit\VOC2007\JPEGImages"

    if args.test_list is None:
        voc_root = os.path.dirname(os.path.dirname(args.image_dir))
        default_test_list = os.path.join(voc_root, "ImageSets", "Segmentation", "test.txt")
        if os.path.exists(default_test_list):
            args.test_list = default_test_list
            print(f"using test list: {args.test_list}")
        else:
            print("[Warning] test.txt not found, will evaluate all images in image-dir.")

    assert os.path.exists(args.weights), f"weights not found: {args.weights}"
    assert os.path.exists(args.image_dir), f"image-dir not found: {args.image_dir}"

    os.makedirs(args.save_dir, exist_ok=True)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"using {device} device.")

    # =========================
    # 创建模型并加载权重
    # =========================
    model = UNet(in_channels=3, num_classes=num_classes, base_c=args.base_c)

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    data_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image_paths = get_image_paths(args.image_dir, args.test_list)
    assert len(image_paths) > 0, "no images found."

    print(f"total test images: {len(image_paths)}")

    hist_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    inference_times = []
    valid_count = 0

    with torch.no_grad():
        # 预热
        init_img = torch.zeros((1, 3, resize_size[0], resize_size[1]), device=device)
        _ = model(init_img)

        for idx, img_path in enumerate(image_paths, start=1):
            image_stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = infer_label_path_from_image_path(img_path, args.label_dir)

            if not os.path.exists(label_path):
                print(f"[Warning] label not found, skip: {label_path}")
                continue

            original_img = Image.open(img_path).convert("RGB")

            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()

            inference_times.append(t_end - t_start)

            if isinstance(output, dict):
                output = output["out"]

            prediction = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            gt_mask = load_label_as_index_mask(label_path, target_size_hw=resize_size)

            # 统计混淆矩阵
            hist_total += fast_hist(
                pred=prediction,
                target=gt_mask,
                num_classes=num_classes,
                ignore_index=args.ignore_index
            )

            # 保存图像结果
            if not args.no_save_images:
                save_one_result(
                    image_stem=image_stem,
                    original_img=original_img,
                    prediction=prediction,
                    gt_mask=gt_mask,
                    resize_size=resize_size,
                    output_root=args.save_dir,
                    alpha=args.alpha
                )

            valid_count += 1

            print(
                f"[{idx}/{len(image_paths)}] {image_stem} done, "
                f"time={t_end - t_start:.4f}s, "
                f"pred={np.unique(prediction)}, gt={np.unique(gt_mask)}"
            )

    print(f"\nvalid evaluated images: {valid_count}")

    if valid_count == 0:
        print("No valid image-label pairs were evaluated.")
        return

    class_names = [
        "background",
        "periapical",
        "caries",
        "furcation",
        "impacted"
    ]

    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    save_metrics(hist_total, args.save_dir, class_names=class_names)

    avg_time = float(np.mean(inference_times)) if len(inference_times) > 0 else float("nan")
    fps = 1.0 / avg_time if avg_time > 0 else float("nan")

    speed_path = os.path.join(args.save_dir, "speed.txt")
    with open(speed_path, "w", encoding="utf-8") as f:
        f.write(f"valid_evaluated_images: {valid_count}\n")
        f.write(f"average_inference_time_per_image_sec: {avg_time:.6f}\n")
        f.write(f"FPS: {fps:.6f}\n")

    print(f"average inference time: {avg_time:.6f}s/image")
    print(f"FPS: {fps:.6f}")
    print(f"speed info saved to: {speed_path}")
    print(f"all results saved in folder: {args.save_dir}")


if __name__ == "__main__":
    main()
