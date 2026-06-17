# BFD-UNet

This repository provides the PyTorch implementation of **BFD-UNet**, a U-Net-based semantic segmentation network with wavelet-based feature decomposition, multi-directional coordinate attention, and constrained deformable convolution. The code is organized for reproducing the training and validation experiments with a Pascal VOC-style dataset layout.


## 1. Environment

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

## 2. Dataset preparation

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
                └── test.txt
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

## 3. Reproduce the main training experiment

Run the following command from the repository root:

```bash
python train.py \
  --data-path ./data \
  --num-classes 5 \
  --device cuda \
  --batch-size 8 \
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

## 4. Resume training

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

