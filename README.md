# BFD-UNet

This repository provides the PyTorch implementation of **BFD-UNet**, a U-Net-based semantic segmentation network with wavelet-based feature decomposition, multi-directional coordinate attention, and constrained deformable convolution. The code is organized for reproducing the training and validation experiments with a Pascal VOC-style dataset layout.

## 1. What is included

```text
BFD-UNet/
├── src/
│   ├── unet.py              # BFD-UNet model definition
│   ├── mobilenet_unet.py    # MobileNetV3-UNet baseline
│   └── vgg_unet.py          # VGG16-UNet baseline
├── train_utils/
│   ├── train_and_eval.py    # training loop, validation metrics, loss function
│   ├── dice_coefficient_loss.py
│   └── distributed_utils.py
├── my_dataset.py            # VOC-style segmentation dataset loader
├── transforms.py            # paired image/mask transforms
├── train.py                 # main single-GPU training script
├── predict.py               # single-image inference script
└── requirements.txt
```

The main experiment uses `train.py` and `src/unet.py`. The multi-GPU script is not required for reproducing the reported single-GPU experiments.

## 2. Environment

The experiments were implemented with Python and PyTorch. A CUDA-capable GPU is recommended.

```bash
conda create -n bfdunet python=3.9 -y
conda activate bfdunet
pip install -r requirements.txt
```

Recommended tested dependency set:

```text
numpy==1.22.0
torch==1.13.1
torchvision==0.14.1
Pillow
```

After installation, verify that the model can be imported and run:

```bash
python - <<'PY'
import torch
from src import UNet

model = UNet(
    in_channels=3,
    num_classes=5,
    bilinear=True,
    base_c=32,
    ema_factor=8,
    deform_max_offset_down3=(1.0, 0.8),
    deform_max_offset_down4=(1.5, 1.0),
)

x = torch.randn(1, 3, 256, 256)
y = model(x)
print(y["out"].shape)
PY
```

Expected output:

```text
torch.Size([1, 5, 256, 256])
```

## 3. Dataset preparation

The dataset should be arranged in Pascal VOC-style format:

```text
DATA_ROOT/
└── VOCdevkit/
    └── VOC2007/
        ├── JPEGImages/
        │   ├── 0001.jpg
        │   ├── 0002.jpg
        │   └── ...
        ├── SegmentationClass/
        │   ├── 0001.png
        │   ├── 0002.png
        │   └── ...
        └── ImageSets/
            └── Segmentation/
                ├── train.txt
                └── val.txt
```

Each line in `train.txt` and `val.txt` should contain only the file name without extension:

```text
0001
0002
0003
```

Mask requirements:

- The image files are RGB images in `.jpg` format.
- The mask files are single-channel `.png` images.
- Pixel value `0` represents background.
- Pixel values `1, 2, ..., num_classes - 1` represent foreground classes.
- Pixel value `255` is ignored during training and evaluation.
- In this code, `--num-classes` means the total number of classes **including background**.

For example, for background plus four foreground classes, use:

```bash
--num-classes 5
```

The training and validation images are resized to `544 × 992` inside `train.py`, so no manual resizing is required before training.

## 4. Reproduce the main training experiment

Run the following command from the repository root:

```bash
python train.py \
  --data-path ./data \
  --num-classes 5 \
  --device cuda \
  --batch-size 1 \
  --epochs 200 \
  --lr 1e-4 \
  --wd 1e-4 \
  --lambda-mag 1e-4 \
  --lambda-smooth 1e-3 \
  --save-best \
  --save-dir ./save_weights
```

Here `./data` should contain `VOCdevkit/VOC2007` as described above.

During training, the script reports:

- training loss,
- validation loss,
- learning rate,
- confusion-matrix-based segmentation metrics,
- Dice coefficient.

The output files are saved to `./save_weights/`:

```text
save_weights/
├── best_model.pth          # best checkpoint selected by validation Dice
├── results_*.txt           # epoch-wise text log
└── history_*.csv           # epoch-wise CSV log: epoch, train_loss, val_loss, lr, dice
```

## 5. Resume training

To continue from a saved checkpoint:

```bash
python train.py \
  --data-path ./data \
  --num-classes 5 \
  --device cuda \
  --batch-size 1 \
  --epochs 200 \
  --lr 1e-4 \
  --wd 1e-4 \
  --lambda-mag 1e-4 \
  --lambda-smooth 1e-3 \
  --save-best \
  --save-dir ./save_weights \
  --resume ./save_weights/best_model.pth
```

## 6. Inference on a single image

Place the trained checkpoint at:

```text
./save_weights/best_model.pth
```

Then set the input image path in `predict.py`:

```python
img_path = "path/to/test/image.png"
```

Run:

```bash
python predict.py
```

The script saves two files in the repository root:

```text
test_result_mask.png       # predicted binary/class mask
test_result_overlay.png    # overlay visualization
```

## 7. Reported experiment setting

Use this table to document the exact setting used in the paper or experiment report.

| Item | Setting |
|---|---|
| Input size | 544 × 992 |
| Optimizer | Adam |
| Initial learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Epochs | 200 |
| Batch size | 1 |
| Loss | Cross-entropy + Dice loss + offset magnitude regularization + offset smoothness regularization |
| Offset magnitude weight | 1e-4 |
| Offset smoothness weight | 1e-3 |
| Best checkpoint criterion | Validation Dice coefficient |

## 8. Expected results

Please fill this table with the exact results obtained from the released split and checkpoint.

| Dataset / split | Model | Dice | mIoU | Pixel accuracy | Checkpoint |
|---|---:|---:|---:|---:|---|
| Validation set | BFD-UNet | TODO | TODO | TODO | `save_weights/best_model.pth` |
| Test set | BFD-UNet | TODO | TODO | TODO | TODO |

For full reproducibility, the released repository should include one of the following:

1. the trained checkpoint file, or
2. a public download link for the checkpoint, or
3. the exact training log and random seed used to obtain the reported result.

## 9. Notes for reproducing the paper results

- Use the provided `train.txt` and `val.txt` split files. Changing the split will change the validation result.
- Use the same `--num-classes` value as the released masks.
- Do not convert segmentation masks to RGB images. The mask must preserve class-index pixel values.
- The current code uses ImageNet normalization in `train.py`: mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- Validation is performed after every epoch, and the best model is selected by validation Dice coefficient.

## 10. Citation

If this code is useful for your research, please cite:

```bibtex
@article{TODO_BFD_UNet,
  title   = {TODO: Paper title},
  author  = {TODO: Author list},
  journal = {TODO: Journal or Conference},
  year    = {TODO}
}
```

## 11. License

TODO: Add a license, for example MIT, Apache-2.0, or another license required by your institution or dataset provider.
