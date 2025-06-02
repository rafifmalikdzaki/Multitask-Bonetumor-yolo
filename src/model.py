import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.ops import non_max_suppression


# Backbone Convnext
class ConvNeXtTiny(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.body = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True, out_indices=(1, 2, 3)
        )

        self.out_channels = self.body.feature_info.channels()

    def forward(self, x):
        return self.body(x)


# BiFPN
class WeightedAdd(nn.Module):
    def __init__(self, n, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = eps

    def forward(self, feats):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(w_i + f for w_i, f in zip(w, feats))


class BiFPNUnit(nn.Module):
    def __init__(self, ch=256):
        super().__init__()

        # Top-down Pathway
        self.add_p4_td = WeightedAdd(2)  # P4 + up(P5)
        self.add_p3_td = WeightedAdd(2)  # P3 + up(P4_td)

        # Bottom-up Pathway
        self.add_p4_out = WeightedAdd(3)  # P4 + P4_td + down(P3_td)
        self.add_p5_out = WeightedAdd(2)  # P5 + down(P4_out)

        self.conv = nn.ModuleDict(
            {
                "p4_td": DWConv(ch, ch, k=3, s=1),
                "p3_td": DWConv(ch, ch, k=3, s=1),
                "p4_out": DWConv(ch, ch, k=3, s=1),
                "p5_out": DWConv(ch, ch, k=3, s=1),
            }
        )

    def forward(self, p3, p4, p5):
        p5_td = p5
        p4_td = self.conv["p4_td"](
            self.add_p4_td([p4, F.interpolate(p5_td, scale_factor=2, mode="nearest")])
        )
        p3_td = self.conv["p3_td"](
            self.add_p3_td([p3, F.interpolate(p4_td, scale_factor=2, mode="nearest")])
        )
        p4_out = self.conv["p4_out"](
            self.add_p4_out([p4, p4_td, F.max_pool2d(p3_td, 2)])
        )
        p5_out = self.conv["p5_out"](self.add_p5_out([p5, F.max_pool2d(p4_out, 2)]))
        return p3_td, p4_out, p5_out


class BiFPN(nn.Module):
    def __init__(self, in_ch=(96, 192, 384), repeats: int = 2):
        super().__init__()
        self.lat3 = Conv(in_ch[0], 256, 1, 1)
        self.lat4 = Conv(in_ch[1], 256, 1, 1)
        self.lat5 = Conv(in_ch[2], 256, 1, 1)
        self.units = nn.ModuleList([BiFPNUnit(256) for _ in range(repeats)])

    def forward(self, feats):
        p3, p4, p5 = feats
        p3 = self.lat3(p3)
        p4 = self.lat4(p4)
        p5 = self.lat5(p5)

        for layer in self.units:
            p3, p4, p5 = layer(p3, p4, p5)

        return p3, p4, p5


# ─────────────────────────────────────────────────────────── Full model
class ConvNeXtBiFPNYOLO(nn.Module):
    def __init__(self, nc_det: int, nc_img: int, proto_ch: int = 32):
        super().__init__()
        self.backbone = ConvNeXtTiny(pretrained=True)
        self.neck = BiFPN(self.backbone.out_channels, repeats=2)

        ch = (256, 256, 256)  # neck channels fed to heads
        self.detect = Detect(nc_det, ch=ch)
        self.segment = Segment(nc_det, nm=proto_ch, ch=ch)

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(256, nc_img)

    # -------------------------------------------------------
    def forward(self, x, mode: str = "infer"):
        p3, p4, p5 = self.neck(self.backbone(x))  # fuse features
        det_out = self.detect([p3, p4, p5])
        seg_out = self.segment([p3, p4, p5])
        img_logits = self.cls_fc(self.cls_pool(p5).flatten(1))

        if mode == "infer":
            return {
                "detect": det_out,
                "segment": (seg_out[0], seg_out[1]),  # Return as tuple
                "img_cls": img_logits.softmax(1),
            }

        return det_out, seg_out, img_logits  # raw tensors


# ─────────────────────────────────────────────────────────── Weight copy
@torch.no_grad()
def load_pretrained_heads(model: nn.Module, ckpt="yolo11m-seg.pt"):
    """Copy all tensors whose name & shape match between ckpt and model."""
    print("Device Ready!")
    src_dict = YOLO("yolo11m-seg.pt").model.state_dict()
    dst_dict = model.state_dict()

    transfer = {
        k.replace("model.", ""): v  # strip prefix if necessary
        for k, v in src_dict.items()
        if k.replace("model.", "") in dst_dict
        and v.shape == dst_dict[k.replace("model.", "")].shape
    }

    print(
        f"Transferring {len(transfer)} tensors "
        f"({sum(t.numel() for t in transfer.values()) / 1e6:.1f}M params)"
    )
    dst_dict.update(transfer)
    model.load_state_dict(dst_dict, strict=False)
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    net = ConvNeXtBiFPNYOLO(nc_det=3, nc_img=3)
    net = load_pretrained_heads(net, "yolo8n-seg.pt")  # Detect+Seg weights

    net.eval().cuda()
    dummy = torch.randn(1, 3, 640, 640).cuda()
    with torch.no_grad():
        out = net(dummy)  # dict with detect / segment / img_cls

    print("\nDetect feature shapes:")
    for f in out["detect"][0]:  # Access the first element of det_out
        print(tuple(f.shape))
    coeffs, protos = out["segment"]
    print("Coeff shape :", coeffs[0][0])
    print("Proto shape :", protos[0][0])
    print("Image-cls   :", out["img_cls"].shape)
