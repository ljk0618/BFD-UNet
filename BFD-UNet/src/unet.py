from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def offset_magnitude_loss(offset: torch.Tensor) -> torch.Tensor:
    return offset.abs().mean()


def offset_smoothness_loss(offset: torch.Tensor) -> torch.Tensor:
    dx = offset[:, :, 1:, :] - offset[:, :, :-1, :]
    dy = offset[:, :, :, 1:] - offset[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


class FixedDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(FixedDoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DeformDoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            mid_channels=None,
            max_offset_x=1.0,
            max_offset_y=1.0
    ):
        super(DeformDoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.max_offset_x = max_offset_x
        self.max_offset_y = max_offset_y
        self.last_offset = None

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.offset_conv2 = nn.Conv2d(
            mid_channels, 18, kernel_size=3, padding=1
        )
        self.conv2 = DeformConv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        nn.init.constant_(self.offset_conv2.weight, 0.0)
        nn.init.constant_(self.offset_conv2.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        raw_offset = self.offset_conv2(x)

        offset = raw_offset.clone()
        offset[:, 0::2, :, :] = self.max_offset_x * torch.tanh(raw_offset[:, 0::2, :, :])
        offset[:, 1::2, :, :] = self.max_offset_y * torch.tanh(raw_offset[:, 1::2, :, :])

        self.last_offset = offset

        x = self.conv2(x, offset)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class EMA(nn.Module):

    def __init__(self, channels: int, factor: int = 8, reduction: int = 16):
        super(EMA, self).__init__()

        groups = min(factor, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.groups = groups
        self.channels = channels
        self.channels_per_group = channels // self.groups

        hidden_channels = max(self.channels_per_group // reduction, 1)

        # =====================================================
        # 1. 多尺度深度卷积特征增强
        # =====================================================
        self.dwconv3x3 = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            groups=channels, bias=True
        )

        self.dwconv5x5 = nn.Conv2d(
            channels, channels,
            kernel_size=5, stride=1, padding=2,
            groups=channels, bias=True
        )

        self.dwconv3x1 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 1), stride=1, padding=(1, 0),
            groups=channels, bias=True
        )

        self.dwconv1x3 = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 3), stride=1, padding=(0, 1),
            groups=channels, bias=True
        )

        self.pre_conv1x1 = nn.Conv2d(
            channels, channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        self.conv1x1 = nn.Conv2d(
            self.channels_per_group,
            self.channels_per_group,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.gn = nn.GroupNorm(
            num_groups=self.channels_per_group,
            num_channels=self.channels_per_group
        )

        self.conv3x3 = nn.Conv2d(
            self.channels_per_group,
            self.channels_per_group,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(self.channels_per_group, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.channels_per_group, kernel_size=1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.last_att_map = None
        self.last_channel_map = None
        self.last_spatial_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        cg = self.channels_per_group

        x_dw3 = self.dwconv3x3(x)
        x_dw5 = self.dwconv5x5(x)
        x_dw31_13 = self.dwconv1x3(self.dwconv3x1(x))

        x_fused = x + x_dw3 + x_dw5 + x_dw31_13
        x_fused = self.pre_conv1x1(x_fused)

        # [B, C, H, W] -> [B*G, C/G, H, W]
        group_x = x_fused.reshape(b * g, cg, h, w)

        x_h = group_x.mean(dim=3, keepdim=True)                         # [B*G, C/G, H, 1]
        x_w = group_x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)      # [B*G, C/G, W, 1]

        x_hw = torch.cat([x_h, x_w], dim=2)                              # [B*G, C/G, H+W, 1]
        x_hw = self.conv1x1(x_hw)

        x_h, x_w = torch.split(x_hw, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                                    # [B*G, C/G, 1, W]

        x_dir = group_x * self.sigmoid(x_h) * self.sigmoid(x_w)
        x_dir = self.gn(x_dir)

        x_dir_map = self.sigmoid(x_dir)                                  # [B*G, C/G, H, W]

        x_local = self.conv3x3(group_x)
        x_local_map = self.sigmoid(x_local)                              # [B*G, C/G, H, W]

        q1 = self.avg_pool(x_dir).reshape(b * g, cg)                     # [B*G, C/G]
        q1 = self.softmax(q1).unsqueeze(1)                               # [B*G, 1, C/G]
        k1 = x_local_map.reshape(b * g, cg, h * w)                       # [B*G, C/G, H*W]

        q2 = self.avg_pool(x_local).reshape(b * g, cg)                   # [B*G, C/G]
        q2 = self.softmax(q2).unsqueeze(1)                               # [B*G, 1, C/G]
        k2 = x_dir_map.reshape(b * g, cg, h * w)                         # [B*G, C/G, H*W]

        spatial_weight = torch.matmul(q1, k1) + torch.matmul(q2, k2)      # [B*G, 1, H*W]
        spatial_weight = spatial_weight.reshape(b * g, 1, h, w)
        spatial_weight = self.sigmoid(spatial_weight)                    # [B*G, 1, H, W]

        channel_weight = self.shared_mlp(self.avg_pool(group_x)) + \
                         self.shared_mlp(self.max_pool(group_x))
        channel_weight = self.sigmoid(channel_weight)                    # [B*G, C/G, 1, 1]

        out = group_x * spatial_weight * channel_weight
        out = out.reshape(b, c, h, w)

        self.last_spatial_map = spatial_weight.reshape(b, g, 1, h, w).detach()
        self.last_att_map = self.last_spatial_map.mean(dim=1).detach()   # [B, 1, H, W]
        self.last_channel_map = channel_weight.reshape(b, c, 1, 1).detach()

        return out


class HaarDWT(nn.Module):

    def __init__(self):
        super(HaarDWT, self).__init__()

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        orig_size = (h, w)

        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return ll, lh, hl, hh, orig_size


class HaarIWT(nn.Module):

    def __init__(self):
        super(HaarIWT, self).__init__()

    def forward(
            self,
            ll: torch.Tensor,
            lh: torch.Tensor,
            hl: torch.Tensor,
            hh: torch.Tensor,
            out_size: Tuple[int, int] = None
    ) -> torch.Tensor:
        b, c, h, w = ll.shape

        x = torch.zeros((b, c, h * 2, w * 2), device=ll.device, dtype=ll.dtype)

        x[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        x[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        x[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        x[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5

        if out_size is not None:
            oh, ow = out_size
            x = x[:, :, :oh, :ow]

        return x


class SkipFusion(nn.Module):
    """
    原输入特征 x 与逆小波重建特征 x_recon 融合，
    再通过 EMA 增强，作为 skip 输出
    """

    def __init__(self, channels: int, ema_factor: int = 8):
        super(SkipFusion, self).__init__()
        self.fuse = FixedDoubleConv(channels * 2, channels, mid_channels=channels)
        self.ema = EMA(channels, factor=ema_factor)

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        out = torch.cat([x, x_recon], dim=1)
        out = self.fuse(out)
        out = self.ema(out)
        return out


class WaveletDown(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            conv_type="fixed",
            ema_factor: int = 8,
            deform_max_offset_x: float = 1.0,
            deform_max_offset_y: float = 1.0
    ):
        super(WaveletDown, self).__init__()
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()

        if conv_type == "fixed":
            self.low_conv = FixedDoubleConv(in_channels, out_channels)
        elif conv_type == "deform":
            self.low_conv = DeformDoubleConv(
                in_channels,
                out_channels,
                max_offset_x=deform_max_offset_x,
                max_offset_y=deform_max_offset_y
            )
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        self.skip_fuse = SkipFusion(in_channels, ema_factor=ema_factor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ll, lh, hl, hh, orig_size = self.dwt(x)

        # 主干：仅低频 LL 进入下一层
        x_down = self.low_conv(ll)

        # 跳跃连接：IWT 重建 + 融合 + EMA
        x_recon = self.iwt(ll, lh, hl, hh, out_size=orig_size)
        skip = self.skip_fuse(x, x_recon)

        return x_down, skip


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = FixedDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = FixedDoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 ema_factor: int = 8,
                 deform_max_offset_down3: Tuple[float, float] = (1.0, 0.8),
                 deform_max_offset_down4: Tuple[float, float] = (1.5, 1.0)):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = FixedDoubleConv(in_channels, base_c)

        self.down1 = WaveletDown(
            base_c, base_c * 2,
            conv_type="fixed",
            ema_factor=ema_factor
        )
        self.down2 = WaveletDown(
            base_c * 2, base_c * 4,
            conv_type="fixed",
            ema_factor=ema_factor
        )

        self.down3 = WaveletDown(
            base_c * 4, base_c * 8,
            conv_type="deform",
            ema_factor=ema_factor,
            deform_max_offset_x=deform_max_offset_down3[0],
            deform_max_offset_y=deform_max_offset_down3[1]
        )

        factor = 2 if bilinear else 1
        self.down4 = WaveletDown(
            base_c * 8, base_c * 16 // factor,
            conv_type="deform",
            ema_factor=ema_factor,
            deform_max_offset_x=deform_max_offset_down4[0],
            deform_max_offset_y=deform_max_offset_down4[1]
        )

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)

        x2, skip1 = self.down1(x1)
        x3, skip2 = self.down2(x2)
        x4, skip3 = self.down3(x3)
        x5, skip4 = self.down4(x4)

        x = self.up1(x5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        logits = self.out_conv(x)
        return {"out": logits}

    def get_attention_maps(self):

        return {
            "skip1": self.down1.skip_fuse.ema.last_att_map,
            "skip2": self.down2.skip_fuse.ema.last_att_map,
            "skip3": self.down3.skip_fuse.ema.last_att_map,
            "skip4": self.down4.skip_fuse.ema.last_att_map,
        }

    def get_attention_details(self):

        return {
            "skip1": {
                "att_map": self.down1.skip_fuse.ema.last_att_map,
                "spatial_map": self.down1.skip_fuse.ema.last_spatial_map,
                "channel_map": self.down1.skip_fuse.ema.last_channel_map,
            },
            "skip2": {
                "att_map": self.down2.skip_fuse.ema.last_att_map,
                "spatial_map": self.down2.skip_fuse.ema.last_spatial_map,
                "channel_map": self.down2.skip_fuse.ema.last_channel_map,
            },
            "skip3": {
                "att_map": self.down3.skip_fuse.ema.last_att_map,
                "spatial_map": self.down3.skip_fuse.ema.last_spatial_map,
                "channel_map": self.down3.skip_fuse.ema.last_channel_map,
            },
            "skip4": {
                "att_map": self.down4.skip_fuse.ema.last_att_map,
                "spatial_map": self.down4.skip_fuse.ema.last_spatial_map,
                "channel_map": self.down4.skip_fuse.ema.last_channel_map,
            },
        }

    def get_deform_offsets(self):

        offsets = {}

        if hasattr(self.down3.low_conv, "last_offset") and self.down3.low_conv.last_offset is not None:
            offsets["down3"] = self.down3.low_conv.last_offset

        if hasattr(self.down4.low_conv, "last_offset") and self.down4.low_conv.last_offset is not None:
            offsets["down4"] = self.down4.low_conv.last_offset

        return offsets

    def get_offset_regularization_loss(self):

        offsets = self.get_deform_offsets()

        if len(offsets) == 0:
            zero = next(self.parameters()).new_tensor(0.0)
            return zero, zero

        loss_mag = 0.0
        loss_smooth = 0.0

        for _, off in offsets.items():
            loss_mag = loss_mag + offset_magnitude_loss(off)
            loss_smooth = loss_smooth + offset_smoothness_loss(off)

        loss_mag = loss_mag / len(offsets)
        loss_smooth = loss_smooth / len(offsets)

        return loss_mag, loss_smooth


if __name__ == "__main__":
    model = UNet(
        in_channels=3,
        num_classes=5,
        bilinear=True,
        base_c=32,
        ema_factor=8,
        deform_max_offset_down3=(1.0, 0.8),
        deform_max_offset_down4=(1.5, 1.0)
    )

    x = torch.randn(1, 3, 544, 992)
    y = model(x)
    print(y["out"].shape)  # [1, 5, 544, 992]

    att_maps = model.get_attention_maps()
    for k, v in att_maps.items():
        if v is not None:
            print(k, v.shape)

    offsets = model.get_deform_offsets()
    for k, v in offsets.items():
        print(k, v.shape, "max_abs =", v.abs().max().item())

    loss_mag, loss_smooth = model.get_offset_regularization_loss()
    print("loss_mag =", float(loss_mag), "loss_smooth =", float(loss_smooth))
