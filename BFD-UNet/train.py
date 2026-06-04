mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
import os
import time
import datetime
import argparse
import csv

import torch
import torch.nn as nn

from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from train_utils.train_and_eval import criterion
from my_dataset import VOCSegmentationDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self,
                 resize_size=(544, 992),
                 hflip_prob=0.5,
                 vflip_prob=0.0,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        trans = [
            T.Resize(resize_size),
        ]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self,
                 resize_size=(544, 992),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train,
                  mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)):
    resize_size = (544, 992)
    if train:
        return SegmentationPresetTrain(
            resize_size=resize_size,
            mean=mean,
            std=std
        )
    else:
        return SegmentationPresetEval(
            resize_size=resize_size,
            mean=mean,
            std=std
        )


def create_model(num_classes):
    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        bilinear=True,
        base_c=32,
        ema_factor=8,
        deform_max_offset_down3=(1.0, 0.8),
        deform_max_offset_down4=(1.5, 1.0)
    )
    return model


@torch.no_grad()
def evaluate_loss(model,
                  data_loader,
                  device,
                  num_classes,
                  lambda_mag=1e-4,
                  lambda_smooth=1e-3):

    model.eval()
    loss_sum = 0.0
    sample_count = 0

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss, _ = criterion(
            outputs,
            targets,
            model=model,
            loss_weight=loss_weight,
            num_classes=num_classes,
            ignore_index=255,
            lambda_mag=lambda_mag,
            lambda_smooth=lambda_smooth
        )

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    val_loss = loss_sum / max(sample_count, 1)
    return val_loss


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    batch_size = args.batch_size
    num_classes = args.num_classes

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(args.save_dir, f"results_{timestamp}.txt")
    history_file = os.path.join(args.save_dir, f"history_{timestamp}.csv")

    train_dataset = VOCSegmentationDataset(
        args.data_path,
        train=True,
        transforms=get_transform(train=True, mean=mean, std=std),
        num_classes=num_classes,
        ignore_index=255
    )

    val_dataset = VOCSegmentationDataset(
        args.data_path,
        train=False,
        transforms=get_transform(train=False, mean=mean, std=std),
        num_classes=num_classes,
        ignore_index=255
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(
        optimizer,
        len(train_loader),
        args.epochs,
        warmup=True
    )

    best_dice = 0.0

    with open(history_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "dice"])

    if args.resume:
        print(f"resume from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model"], strict=True)

        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("optimizer state loaded successfully.")
            except Exception as e:
                print(f"warning: optimizer state not loaded: {e}")

        if "lr_scheduler" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                print("lr_scheduler state loaded successfully.")
            except Exception as e:
                print(f"warning: lr_scheduler state not loaded: {e}")

        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1

        if args.amp and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
                print("amp scaler state loaded successfully.")
            except Exception as e:
                print(f"warning: scaler state not loaded: {e}")

        if "best_dice" in checkpoint:
            best_dice = checkpoint["best_dice"]

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            num_classes=num_classes,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            scaler=scaler,
            lambda_mag=args.lambda_mag,
            lambda_smooth=args.lambda_smooth
        )

        # 验证损失：与训练损失保持一致
        val_loss = evaluate_loss(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
            lambda_mag=args.lambda_mag,
            lambda_smooth=args.lambda_smooth
        )

        print("开始验证指标...")
        confmat, dice = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes
        )
        print("验证完成")

        val_info = str(confmat)
        print(val_info)
        print(f"train_loss: {mean_loss:.6f}")
        print(f"val_loss: {val_loss:.6f}")
        print(f"dice coefficient: {dice:.4f}")

        with open(results_file, "a", encoding="utf-8") as f:
            train_info = (
                f"[epoch: {epoch}]\n"
                f"train_loss: {mean_loss:.6f}\n"
                f"val_loss: {val_loss:.6f}\n"
                f"lr: {lr:.8f}\n"
                f"dice coefficient: {dice:.4f}\n"
                f"lambda_mag: {args.lambda_mag}\n"
                f"lambda_smooth: {args.lambda_smooth}\n"
            )
            f.write(train_info + val_info + "\n\n")

        with open(history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mean_loss, val_loss, lr, dice])

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": vars(args)
        }
        if args.amp and scaler is not None:
            save_file["scaler"] = scaler.state_dict()
        else:
            save_file["scaler"] = None

        if args.save_best:
            if dice > best_dice:
                best_dice = dice
                save_file["best_dice"] = best_dice
                save_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(save_file, save_path)
                print(f"保存 best_model.pth, epoch={epoch}, dice={dice:.4f}")
            else:
                print(f"本轮未保存, epoch={epoch}, dice={dice:.4f}, best_dice={best_dice:.4f}")
        else:
            save_path = os.path.join(args.save_dir, f"model_{epoch}.pth")
            torch.save(save_file, save_path)
            print(f"保存 {save_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    print(f"history csv saved to: {history_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch UNet Training with Adam")

    parser.add_argument("--data-path", default=r"./", help="dataset root")
    parser.add_argument("--num-classes", default=5, type=int, help="number of classes including background")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="batch size")
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float,
                        metavar="W", help="weight decay", dest="weight_decay")

    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers")
    parser.add_argument("--print-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--save-dir", default="./save_weights", help="directory to save checkpoints")

    parser.add_argument("--save-best", action="store_true", help="only save best dice weights")
    parser.add_argument("--amp", action="store_true", help="use torch.cuda.amp for mixed precision training")

    # 新增：offset 正则系数
    parser.add_argument("--lambda-mag", default=1e-4, type=float, help="weight of offset magnitude regularization")
    parser.add_argument("--lambda-smooth", default=1e-3, type=float, help="weight of offset smoothness regularization")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)