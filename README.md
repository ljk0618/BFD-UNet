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

Each line in `train.txt、test.txt` and `val.txt` should contain only the file name without extension:

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
  --batch-size 8 \
  --epochs 200 \
  --lr 1e-4 \
  --wd 1e-4 \
  --lambda-mag 1e-4 \
  --lambda-smooth 1e-3 \
  --save-best \
  --save-dir ./save_weights \
  --resume ./save_weights/best_model.pth
```


## 5. Test and evaluate the model on the test set

After training, place the trained checkpoint at:

```text
./save_weights/best_model.pth
```

The test script predicts all images in the test set and calculates the segmentation metrics, including IoU, Dice, Precision, and Recall for each class, as well as the mean values over all lesion classes.

Run the following command from the repository root:

```bash
python predict.py \
  --weights ./save_weights/best_model.pth \
  --image-dir ./data/VOCdevkit/VOC2007/JPEGImages \
  --label-dir ./data/VOCdevkit/VOC2007/SegmentationClass \
  --test-list ./data/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt \
  --num-classes 5 \
  --resize-h 544 \
  --resize-w 992 \
  --save-dir ./test_results
```

The script will automatically:

* load all image names from `test.txt`;
* read the corresponding images from `JPEGImages/`;
* read the corresponding ground-truth masks from `SegmentationClass/`;
* resize the images and masks to `544 × 992`;
* predict the segmentation masks;
* save the prediction results and visualizations;
* calculate the quantitative evaluation metrics.

The output directory will be organized as follows:

```text
test_results/
├── image_name_1/
│   ├── input_resized.png
│   ├── pred_gray.png
│   ├── pred_vis.png
│   ├── pred_color.png
│   ├── pred_overlay.png
│   ├── gt_gray.png
│   ├── gt_vis.png
│   ├── gt_color.png
│   ├── gt_overlay.png
│   └── compare_pred_gt.png
├── image_name_2/
│   └── ...
├── metrics_per_class.csv
├── metrics_summary.txt
├── confusion_matrix.csv
└── speed.txt
```

The most important output files are:

```text
metrics_per_class.csv      # IoU, Dice, Precision, and Recall for each class
metrics_summary.txt        # mean IoU, mean Dice, mean Precision, and mean Recall
confusion_matrix.csv       # accumulated confusion matrix on the test set
speed.txt                  # average inference time and FPS
```

If only the quantitative metrics are required and image visualizations are not needed, run:

```bash
python predict.py \
  --weights ./save_weights/best_model.pth \
  --image-dir ./data/VOCdevkit/VOC2007/JPEGImages \
  --label-dir ./data/VOCdevkit/VOC2007/SegmentationClass \
  --test-list ./data/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt \
  --num-classes 5 \
  --resize-h 544 \
  --resize-w 992 \
  --save-dir ./test_results \
  --no-save-images
```

## 6. Dataset availability

The in-house panoramic dental radiograph dataset used in this study contains 2,150 images. Due to data privacy and ethical restrictions, the in-house dataset is not directly included in this repository. The dataset may be obtained from the corresponding author upon reasonable request and after permission is granted.

The repository provides the required dataset structure and scripts for training, testing, and evaluation. Users who wish to reproduce the experiments should arrange the dataset according to the Pascal VOC-style layout described in Section 2.

## 7. Notes for reproducibility

To reduce unnecessary workload for readers, the repository provides:

* the complete model implementation;
* the training script;
* the test-set prediction and evaluation script;
* the required dependency list;
* the expected dataset directory structure;
* the exact training and testing commands;
* the output files corresponding to the reported experimental results.

The default experimental setting is:

```text
Input size: 544 × 992
Number of classes: 5, including background
Background label: 0
Lesion labels: 1, 2, 3, 4
Ignore label: 255
Training epochs: 200
Initial learning rate: 1e-4
Weight decay: 1e-4
Batch size: 8
```

If a different GPU memory capacity is used, the batch size can be reduced. In that case, the learning rate may need to be adjusted accordingly.


