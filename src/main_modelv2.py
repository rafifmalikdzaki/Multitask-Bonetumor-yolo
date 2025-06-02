# ────────────────────────────── imports
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.nn.modules.block import Proto
from typing import Tuple, List


# ────────────────────────────── Backbone
class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny backbone that gives us 3 stages at strides 4 / 8 / 16
    (indices 0-1-2 in timm's feature list).  That means the highest-
    resolution feature will be 160×160 for a 640×640 input.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.body = timm.create_model(
            "convnext_tiny.in12k_ft_in1k",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),  # ← stride 8-16-32
        )
        self.body = self.body.eval()
        self.c2f_p3 = C2f(192, 256)
        self.c2f_p4 = C2f(384, 384)
        self.c2f_p5 = C2f(768, 512)
        self.out_channels = self.body.feature_info.channels()  # (192, 384, 768)

    def forward(self, x):
        p3, p4, p5 = self.body(x)  # tuple of (P3,P4,P5)
        cp3 = self.c2f_p3(p3)
        cp4 = self.c2f_p4(p4)
        cp5 = self.c2f_p5(p5)
        return cp3, cp4, cp5


# ────────────────────────────── Bi-FPN
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel: Tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = ConvBlock(in_channels, c_, kernel[0], 1)
        self.cv2 = ConvBlock(c_, out_channels, kernel[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DepthwiseConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        freeze_bn=False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        freeze_bn=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            dilation,
            groups,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class C2f(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 2,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(out_channels * e)
        self.cv1 = ConvBlock(in_channels, 2 * self.c, 1, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, groups=g, kernel=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        # print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class BiFPNUnit(nn.Module):
    def __init__(self, feature_size=256, eps=0.0001):
        super().__init__()
        self.eps = eps

        self.p3_td_conv = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_td_cf = C2f(feature_size, feature_size, shortcut=False)
        self.p4_td_conv = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td_cf = C2f(feature_size, feature_size, shortcut=False)

        self.p4_out_conv = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_out_cf = C2f(feature_size, feature_size, shortcut=False)
        self.p5_out_conv = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out_cf = C2f(feature_size, feature_size, shortcut=False)

        self.w1 = nn.Parameter(torch.Tensor(2, 2), requires_grad=True)
        self.w2 = nn.Parameter(torch.Tensor(3, 2), requires_grad=True)

    def _norm(self, w: torch.Tensor) -> torch.Tensor:
        w = F.elu(w)
        return w / (w.sum(dim=0, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not len(x) == 3:
            raise ValueError(f"BiFPNBlock expects 3 input feature levels, got {len(x)}")
        p3_x, p4_x, p5_x = x

        w1_norm, w2_norm = self._norm(self.w1), self._norm(self.w2)

        # --- Top-Down Pathway ---
        # P_highest_td (P5_td_intermediate) is just P_highest_x (P5_x)
        p5_td_intermediate = p5_x

        # Calculate P4_td: combines P4_x and upsampled P5_td_intermediate
        # F.interpolate is used for upsampling (doubling resolution)
        p4_td_sum = w1_norm[0, 0] * p4_x + w1_norm[1, 0] * F.interpolate(
            p5_td_intermediate, scale_factor=2, mode="bilinear"
        )
        p4_td = self.p4_td_cf(self.p4_td_conv(p4_td_sum))

        # Calculate P3_td: combines P3_x and upsampled P4_td
        p3_td_sum = w1_norm[0, 1] * p3_x + w1_norm[1, 1] * F.interpolate(
            p4_td, scale_factor=2, mode="bilinear"
        )
        p3_td = self.p3_td_cf(self.p3_td_conv(p3_td_sum))

        # --- Bottom-Up Pathway ---
        # P_lowest_out (P3_out) is P_lowest_td (P3_td)
        p3_out = p3_td

        # Calculate P4_out: combines P4_x (original), P4_td (top-down), and downsampled P3_out
        # F.interpolate is used for downsampling (halving resolution)
        p4_out_sum = (
            w2_norm[0, 0] * p4_x
            + w2_norm[1, 0] * p4_td
            + w2_norm[2, 0] * F.interpolate(p3_out, scale_factor=0.5, mode="bilinear")
        )
        p4_out = self.p4_out_cf(self.p4_out_conv(p4_out_sum))

        # Calculate P5_out: combines P5_x (original), P5_td_intermediate (top-down path for this level), and downsampled P4_out
        p5_out_sum = (
            w2_norm[0, 1] * p5_x
            + w2_norm[1, 1] * p5_td_intermediate
            + w2_norm[2, 1] * F.interpolate(p4_out, scale_factor=0.5, mode="bilinear")
        )
        p5_out = self.p5_out_cf(self.p5_out_conv(p5_out_sum))

        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    """BiFPN (Bidirectional Feature Pyramid Network) for P3, P4, P5 levels."""

    def __init__(
        self,
        size: List[int],
        feature_size: int = 256,
        num_layers: int = 3,
        eps: float = 0.0001,
    ):
        super().__init__()
        if not len(size) == 3:
            raise ValueError(
                f"BiFPN expects 3 input sizes for C3, C4, C5 projections, got {len(size)}"
            )

        # Initial 1x1 convolutions to project backbone features to `feature_size`
        self.p3_proj = ConvBlock(size[0], feature_size, kernel_size=1)
        self.p4_proj = ConvBlock(size[1], feature_size, kernel_size=1)
        self.p5_proj = ConvBlock(size[2], feature_size, kernel_size=1)

        self.num_layers = num_layers
        self.feature_size = feature_size  # For external access if needed

        bifpn_units = []
        for _ in range(num_layers):
            bifpn_units.append(BiFPNUnit(feature_size, eps=eps))
        self.bifpn_units = nn.Sequential(*bifpn_units)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not len(inputs) == 3:
            raise ValueError(
                f"BiFPN expects 3 input feature maps (from backbone C2f), got {len(inputs)}"
            )
        # c3, c4, c5 are features from backbone's C2f blocks (e.g., channels [256, 384, 512])
        c3, c4, c5 = inputs

        # Project to common feature_size for BiFPN
        p3_x = self.p3_proj(c3)
        p4_x = self.p4_proj(c4)
        p5_x = self.p5_proj(c5)

        features = [p3_x, p4_x, p5_x]

        # Pass through BiFPN layers
        # nn.Sequential expects a single tensor input if not overridden
        # So we iterate if BiFPNUnit expects a list
        for unit in self.bifpn_units:
            features = unit(features)

        return features


# ────────────────────────────── Full model
class ConvNeXtBiFPNYOLO(nn.Module):
    def __init__(
        self,
        nc_det: int,
        nc_img: int,
        proto_ch: int = 32,
        bifpn_feature_size: int = 256,
        bifpn_num_layers: int = 2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = ConvNeXtTiny(pretrained=pretrained_backbone)

        # Input channels to BiFPN are from backbone.out_channels_after_c2f
        self.neck = BiFPN(
            size=[256, 384, 512],
            feature_size=bifpn_feature_size,
            num_layers=bifpn_num_layers,
        )

        # Channel list for detection/segmentation heads
        # All output features from BiFPN will have `bifpn_feature_size` channels
        head_input_channels = [bifpn_feature_size] * 3

        self.segment = Segment(
            nc=nc_det, nm=proto_ch, npr=bifpn_feature_size, ch=head_input_channels
        )
        # Note: npr for Segment's Proto is usually related to an intermediate feature channel size.
        # Here, using bifpn_feature_size for npr as it's the channel size of P3 fed to Proto.

        # Image classification head (operates on the lowest-resolution feature from neck)
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(bifpn_feature_size, nc_img)  # Input is from P5 of neck

        # Store some config
        self.nc_det = nc_det
        self.nc_img = nc_img
        self.proto_ch = proto_ch

    def forward(self, x, mode: str = "train"):
        backbone_features = self.backbone(x)  # List of 3 features [cp3, cp4, cp5]

        # Neck processes these features
        # Output: [neck_p3, neck_p4, neck_p5] (high-res to low-res)
        # Renaming for clarity: neck_p3 corresponds to original P3 level, etc.
        neck_p3, neck_p4, neck_p5 = self.neck(backbone_features)

        # Heads expect features in order [P3, P4, P5] ( Ultralytics convention for Detect/Segment)
        head_inputs = [neck_p3, neck_p4, neck_p5]

        original_segment_training_state = self.segment.training
        # print(neck_p3.shape, neck_p4.shape, neck_p5.shape)

        try:
            if mode == "train":
                self.segment.train()

                segment_head_output = self.segment(head_inputs.copy())
                # Img classification uses the P5 feature map from the neck (lowest resolution)
                img_cls_output = self.cls_fc(self.cls_pool(neck_p5).flatten(1))
                return segment_head_output, img_cls_output

            elif mode == "infer":
                # self.segment.train()
                self.segment.eval()

                # Segment head in eval mode typically returns (seg_predictions_cat, (mask_coeffs, protos))
                seg_preds_cat, protos = self.segment(head_inputs.copy())
                det_preds_cat = seg_preds_cat[:, : 4 + self.nc_det]  # slice boxes+cls
                img_logits = self.cls_fc(self.cls_pool(neck_p5).flatten(1))

                return {
                    "detect_preds_cat": det_preds_cat,  # Concatenated predictions
                    "segment_protos": protos,
                    "segment_preds_cat": seg_preds_cat,
                    # Concatenated segment predictions (boxes + per-class mask coeffs)
                    "img_cls_logits": img_logits,
                    "img_cls_probs": img_logits.softmax(dim=1),
                }
            else:
                raise ValueError(
                    f"Unknown mode for ConvNeXtBiFPNYOLO.forward: {mode}. Expected 'train' or 'infer'."
                )
        finally:
            self.segment.training = original_segment_training_state


# ───────────────────────────────────────────────────────────
#  helper: verbose per-head transfer report (User Provided)
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def load_pretrained_heads(model: nn.Module, segment_ckpt_path: str = None):
    """
    Loads pretrained weights into the detection and segmentation heads of the model.
    Args:
        model (nn.Module): The target model whose heads will be updated.
        detect_ckpt_path (str, optional): Path to the YOLO checkpoint for the Detect head (e.g., "yolov8s.pt").
        segment_ckpt_path (str, optional): Path to the YOLO checkpoint for the Segment head (e.g., "yolov8s-seg.pt").
    """

    def get_source_modulelist(ckpt_path):
        if ckpt_path is None:
            return None
        print(f"\nLoading YOLO checkpoint from: {ckpt_path}")
        try:
            yolo_ckpt_model = YOLO(ckpt_path).model
        except Exception as e:
            print(f"Error loading YOLO model from {ckpt_path}: {e}")
            return None

        src_nn_modulelist = None
        if hasattr(yolo_ckpt_model, "model") and isinstance(
            yolo_ckpt_model.model, (nn.ModuleList, nn.Sequential)
        ):
            src_nn_modulelist = yolo_ckpt_model.model
            print(
                f"  Successfully accessed model layers from yolo_ckpt_model.model (type: {type(src_nn_modulelist)})."
            )
        elif isinstance(yolo_ckpt_model, (nn.ModuleList, nn.Sequential)):
            src_nn_modulelist = yolo_ckpt_model
            print(
                f"  yolo_ckpt_model itself is an nn.ModuleList or nn.Sequential (type: {type(src_nn_modulelist)})."
            )

        if src_nn_modulelist is None:
            print(
                f"  Could not find main layer sequence in the loaded YOLO model (Top level type: {type(yolo_ckpt_model)})."
            )
        return src_nn_modulelist

    def find_head_in_modulelist(module_list, head_class, head_name):
        if module_list is None:
            return None
        for module in reversed(list(module_list)):
            if isinstance(module, head_class):
                print(f"  Found {head_name} (type: {type(module)}) in source model.")
                return module
        print(
            f"  Could not find {head_name} of type {head_class.__name__} in the provided module list."
        )
        return None

    def copy_named_params(src_mod, dst_mod, mod_name_print=""):
        # (Implementation is the same as in the previous version, using dst_p.copy_(src_p))
        copied_count = total_params = 0
        if src_mod is None or dst_mod is None:
            print(
                f"    Skipping {mod_name_print}: Source or destination module is None."
            )
            return 0, 0
        src_dict = dict(src_mod.named_parameters())
        dst_dict = dict(dst_mod.named_parameters())
        if not src_dict:
            print(
                f"    Warning: Source module {mod_name_print} ({src_mod.__class__.__name__}) has no parameters."
            )
        if not dst_dict:
            print(
                f"    Warning: Destination module {mod_name_print} ({dst_mod.__class__.__name__}) has no parameters."
            )
        for name, dst_p in dst_mod.named_parameters():
            total_params += 1
            if name in src_dict:
                src_p = src_dict[name]
                if src_p.shape == dst_p.shape:
                    dst_p.copy_(src_p)
                    copied_count += 1
                else:
                    print(
                        f"    Shape mismatch for {mod_name_print} param '{name}': src {src_p.shape}, dst {dst_p.shape}"
                    )
            else:
                print(
                    f"    Param '{name}' not found in source module {mod_name_print}."
                )
        return copied_count, total_params

    # --- Load Segment Head ---
    src_segment_head = None
    if segment_ckpt_path:
        segment_module_list = get_source_modulelist(segment_ckpt_path)
        if segment_module_list:
            src_segment_head = find_head_in_modulelist(
                segment_module_list, Segment, "Segment Head"
            )

    seg_c_total, seg_t_total = 0, 0
    if hasattr(model, "segment") and model.segment is not None:
        if src_segment_head is not None:
            # Copy submodules of Segment head
            if hasattr(src_segment_head, "cv4") and hasattr(model.segment, "cv4"):
                if (
                    isinstance(src_segment_head.cv4, nn.ModuleList)
                    and isinstance(model.segment.cv4, nn.ModuleList)
                    and len(src_segment_head.cv4) == len(model.segment.cv4)
                ):
                    for i in range(len(src_segment_head.cv4)):
                        c, t = copy_named_params(
                            src_segment_head.cv4[i],
                            model.segment.cv4[i],
                            f"Segment.cv4[{i}] (Mask Coeffs Convs)",
                        )
                        seg_c_total += c
                        seg_t_total += t
            elif hasattr(src_segment_head, "mc") and hasattr(model.segment, "mc"):
                c, t = copy_named_params(
                    src_segment_head.mc,
                    model.segment.mc,
                    "Segment.mc (Mask Coeffs Convs)",
                )
                seg_c_total += c
                seg_t_total += t

            if hasattr(src_segment_head, "proto") and hasattr(model.segment, "proto"):
                c, t = copy_named_params(
                    src_segment_head.proto,
                    model.segment.proto,
                    "Segment.proto (Prototype Generator)",
                )
                seg_c_total += c
                seg_t_total += t

            # Copy inherited Detect components (cv2, cv3) from the source Segment head
            if hasattr(src_segment_head, "cv2") and hasattr(model.segment, "cv2"):
                if (
                    isinstance(src_segment_head.cv2, nn.ModuleList)
                    and isinstance(model.segment.cv2, nn.ModuleList)
                    and len(src_segment_head.cv2) == len(model.segment.cv2)
                ):
                    for i in range(len(src_segment_head.cv2)):
                        c, t = copy_named_params(
                            src_segment_head.cv2[i],
                            model.segment.cv2[i],
                            f"Segment.cv2[{i}] (Inherited Detect Bbox Reg)",
                        )
                        seg_c_total += c
                        seg_t_total += t

            if hasattr(src_segment_head, "cv3") and hasattr(model.segment, "cv3"):
                if (
                    isinstance(src_segment_head.cv3, nn.ModuleList)
                    and isinstance(model.segment.cv3, nn.ModuleList)
                    and len(src_segment_head.cv3) == len(model.segment.cv3)
                ):
                    for i in range(len(src_segment_head.cv3)):
                        c, t = copy_named_params(
                            src_segment_head.cv3[i],
                            model.segment.cv3[i],
                            f"Segment.cv3[{i}] (Inherited Detect Class Pred)",
                        )
                        seg_c_total += c
                        seg_t_total += t
            print(
                f"Segment head         : {seg_c_total}/{seg_t_total} tensors copied from {segment_ckpt_path}"
            )
        else:
            print(
                f"No source Segment head found or loaded from {segment_ckpt_path} to copy to model.segment."
            )
    elif not (hasattr(model, "segment") and model.segment is not None):
        print("Destination model has no 'segment' attribute or it's None.")

    total_copied = seg_c_total
    total_attempted = seg_t_total
    print(
        f"\nHead-weight summary  : {total_copied}/{total_attempted} tensors copied overall."
    )
    return model


# ───────────────────────── smoke-test (bottom of file) ─────
if __name__ == "__main__":
    torch.manual_seed(0)
    NUM_DET_CLASSES = 80
    NUM_IMG_CLASSES = 3
    PROTO_CHANNELS = 32
    BIFPN_OUT_CH = 128
    BIFPN_REPEATS = 2

    print("Attempting to initialize model...")
    net = ConvNeXtBiFPNYOLO(
        nc_det=NUM_DET_CLASSES,
        nc_img=NUM_IMG_CLASSES,
        proto_ch=PROTO_CHANNELS,
        bifpn_feature_size=BIFPN_OUT_CH,
        bifpn_num_layers=BIFPN_REPEATS,
    )
    print("Model initialized.")

    # --- Attempt to load pretrained heads ---
    # Provide paths to your local checkpoints if available
    # Example: yolov8s.pt for detection, yolov8s-seg.pt for segmentation
    # If files are not found, loading will be skipped.
    DETECT_CKPT_PATH = "yolov8s.pt"  # Pure detection checkpoint
    SEGMENT_CKPT_PATH = "yolov8s-seg.pt"  # Segmentation checkpoint

    import os

    if not os.path.exists(DETECT_CKPT_PATH):
        print(
            f"Warning: Detection checkpoint {DETECT_CKPT_PATH} not found. Detect head weights will not be loaded."
        )
        DETECT_CKPT_PATH = None  # Set to None so load_pretrained_heads skips it
    if not os.path.exists(SEGMENT_CKPT_PATH):
        print(
            f"Warning: Segmentation checkpoint {SEGMENT_CKPT_PATH} not found. Segment head weights will not be loaded."
        )
        SEGMENT_CKPT_PATH = None  # Set to None

    net = load_pretrained_heads(net, segment_ckpt_path=SEGMENT_CKPT_PATH)

    if torch.cuda.is_available():
        net = net.cuda()
        dummy_device = "cuda"
        print("Using CUDA.")
    else:
        net = net.eval()
        dummy_device = "cpu"
        print("CUDA not available, running on CPU.")

    dummy_img = torch.randn(2, 3, 640, 640, device=dummy_device)

    print("\nTesting TRAIN mode output:")
    net.train()

    train_out_segment_tuple, train_out_img_cls = net(dummy_img, mode="train")

    print(f"  Segment output (tuple) type: {type(train_out_segment_tuple)}")
    if (
        isinstance(train_out_segment_tuple, (list, tuple))
        and len(train_out_segment_tuple) == 3
    ):
        seg_raw_det, seg_mc, seg_p = train_out_segment_tuple
        print(f"    Segment item 0 (det_internal_raw_list) type: {type(seg_raw_det)}")
        if isinstance(seg_raw_det, (list, tuple)):
            for i, tnsr in enumerate(seg_raw_det):
                print(f"      Det internal item {i} shape: {tnsr.shape}")
        elif isinstance(seg_raw_det, torch.Tensor):
            print(f"      Det internal raw tensor shape: {seg_raw_det.shape}")

        print(f"    Segment item 1 (coeffs 'mc') shape: {seg_mc.shape}")
        print(f"    Segment item 2 (protos 'p') shape: {seg_p.shape}")
    else:
        print(f"  Segment output structure not as expected: {train_out_segment_tuple}")

    print(f"  Img Cls output shape: {train_out_img_cls.shape}")

    print("\nTesting INFER mode output:")
    net.eval()
    with torch.no_grad():
        infer_out_dict = net(dummy_img, mode="infer")

    print(f"  Infer output type: {type(infer_out_dict)}")
    for key, value in infer_out_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  Key '{key}' shape: {value.shape}, dtype: {value.dtype}")
        elif (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], torch.Tensor)
        ):
            print(
                f"  Key '{key}' type: {type(value)}, length: {len(value)}, First item shape: {value[0].shape}"
            )
        elif isinstance(value, (list, tuple)):
            print(f"  Key '{key}' type: {type(value)}, length: {len(value)}")
        else:
            print(f"  Key '{key}' type: {type(value)}")
    print("Smoke test finished.")
