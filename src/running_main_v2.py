# ────────────────────────────────────────────────── imports
from __future__ import annotations
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import matplotlib.pyplot as plt
import io
import time  # For timing diagnostics
from typing import List
import numpy as np
from multitask_logging import log_cls_metrics, log_seg_examples, log_det_examples

# Assuming these files are in the same directory or in PYTHONPATH
from main_model import ConvNeXtBiFPNYOLO, load_pretrained_heads
from dataset_btxrdv2 import BTXRD, collate_fn

from torchmetrics import F1Score as TorchMetricsF1Score  # Alias to avoid conflict
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.detection import MeanAveragePrecision
import torchvision


import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

# These constants can be tuned project‑wide in one place.
LOG_FREQ_TRAIN = 200  # images – keep low to avoid W&B quota burn‑up
MAX_VIZ_PER_CALL = 4  # safeguards against logging massive batches unintentionally
CONF_TH = 0.05  # confidence threshold
NMS_IOU = 0.6  # IoU for NMS
TOP_K = 300  # keep at most K boxes/image


def _img_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in −1…1 *or* 0…1 → HWC uint8 (0‑255)."""
    if img.dim() != 3:
        raise ValueError("Expected CHW image tensor, got dim=%d" % img.dim())
    img = img.detach().cpu()
    if img.min() < 0:
        img = (img + 1) / 2
    img = (img.clamp(0, 1) * 255).byte()
    return img.permute(1, 2, 0).numpy()


# ────────────────────────────────────────────────── Loss and Box Utils
def batch_bbox_iou(boxes1, boxes2, eps: float = 1e-7):
    """
    Compute IoU between two sets of boxes (x1,y1,x2,y2 format).
    boxes1: [N, 4]
    boxes2: [M, 4]
    Returns: [N, M] IoU matrix
    """
    if boxes1.device != boxes2.device:
        boxes2 = boxes2.to(boxes1.device)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    iou = intersection / (union + eps)
    return iou


def dist2bbox(distance, anchor_points, box_format="xyxy"):
    lt, rb = torch.split(distance, 2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == "xyxy":
        return torch.cat((x1y1, x2y2), dim=-1)
    elif box_format == "xywh":
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=-1)
    raise NotImplementedError(f"Box format '{box_format}' not implemented.")


def plot_confusion_matrix_to_wandb(cm_tensor, class_names_dict):
    if cm_tensor is None:
        return None
    class_names_list = [
        class_names_dict.get(i, str(i)) for i in range(cm_tensor.shape[0])
    ]

    fig, ax = plt.subplots(
        figsize=(max(6, len(class_names_list)), max(5, len(class_names_list) * 0.8))
    )
    cm_np = cm_tensor.cpu().numpy()

    annotation_format = ".2f"

    sns.heatmap(
        cm_np,
        annot=True,
        fmt=annotation_format,
        cmap="Blues",
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close(fig)
    return wandb.Image(pil_img)


# ────────────────────────────────────────────────── Lightning Module
class MultiTaskLitModel(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 640,
        lr: float = 1e-4,
        mask_log_period: int = 100,  # Used for log_seg_examples frequency
        box_log_period: int = 1,  # Used for log_det_examples frequency (val epoch)
        cls_log_period: int = 50,  # New: Frequency for logging cls metrics (train steps)
        nc_det: int = 3,
        num_img_classes: int = 2,
        proto_ch: int = 32,
        loss_weight_seg: float = 1.0,
        loss_weight_box_iou: float = 2.0,
        loss_weight_dfl: float = 1.5,
        loss_weight_cls_det: float = 0.5,
        loss_weight_img_cls: float = 1.0,
        iou_match_thresh: float = 0.5,
        det_conf_thresh_viz: float = 0.25,  # Used by log_det_examples
        map_max_detections: int = 300,
    ):
        super().__init__()
        # Add cls_log_period to hparams if you want to configure it
        self.save_hyperparameters(
            "img_size",
            "lr",
            "mask_log_period",
            "box_log_period",
            "cls_log_period",
            "nc_det",
            "num_img_classes",
            "proto_ch",
            "loss_weight_seg",
            "loss_weight_box_iou",
            "loss_weight_dfl",
            "loss_weight_cls_det",
            "loss_weight_img_cls",
            "iou_match_thresh",
            "det_conf_thresh_viz",
            "map_max_detections",
        )

        self.net = ConvNeXtBiFPNYOLO(
            nc_det=self.hparams.nc_det,
            nc_img=self.hparams.num_img_classes,
            proto_ch=self.hparams.proto_ch,
        )
        # load_pretrained_heads(self.net, ckpt_path="yolov8s-seg.pt") # Make sure this path is correct

        self.seg_proto_projector = nn.Conv2d(self.hparams.proto_ch, 1, kernel_size=1)

        self.img_cls_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = nn.BCEWithLogitsLoss()
        self.det_cls_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.det_dfl_loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.train_img_acc = MulticlassAccuracy(
            num_classes=self.hparams.num_img_classes,
            average="micro",
            dist_sync_on_step=True,
        )
        self.val_img_acc = MulticlassAccuracy(
            num_classes=self.hparams.num_img_classes,
            average="micro",
            dist_sync_on_step=True,
        )
        self.val_img_cm = MulticlassConfusionMatrix(
            num_classes=self.hparams.num_img_classes, normalize="true"
        )

        self.val_seg_f1 = TorchMetricsF1Score(task="binary", dist_sync_on_step=True)
        # ── Fast metric ─────────────────────────────────────────────────────────
        # • single IoU threshold (0.50)  ➜ ≈10 × faster than the full curve
        # • class_metrics=False          ➜ skips per-class bookkeeping
        # This one is available every epoch and is used for checkpointing / early-stop
        self.val_map_iou50 = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5],  # AP-50 only
            class_metrics=False,  # global mAP, no per-class breakdown
            max_detection_thresholds=[  # 1 list element is enough
                1,
                10,
                self.hparams.map_max_detections,
            ],
            dist_sync_on_step=True,
        )

        # ── Full COCO-style metric ──────────────────────────────────────────────
        # • 10 IoU thresholds (0.50:0.05:0.95)
        # • still global only; run it sparsely (e.g. every 5 epochs) in validation_step
        self.val_map_iou50_95 = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=torch.linspace(0.5, 0.95, 10).tolist(),
            class_metrics=False,  # turn on only if you truly need per-class AP
            max_detection_thresholds=[1, 10, self.hparams.map_max_detections],
            dist_sync_on_step=True,
        )

        self.val_det_cm = MulticlassConfusionMatrix(
            num_classes=self.hparams.nc_det, normalize="true"
        )
        self.temp_matched_preds_for_cm = []

        self.reg_max = (
            self.net.detect.reg_max if hasattr(self.net.detect, "reg_max") else 16
        )

        self.img_class_names = {
            i: f"imgC{i}" for i in range(self.hparams.num_img_classes)
        }
        self.det_class_names = {i: f"detC{i}" for i in range(self.hparams.nc_det)}
        # self.seg_class_labels_wandb is not needed if log_seg_examples handles its own labels

    def _multitask_loss(
        self,
        det_head_outputs,
        seg_head_outputs,
        img_cls_logits_pred,
        gt_det_boxes,
        gt_masks,
        gt_img_cls,
    ):
        loss_img_cls = self.img_cls_loss_fn(img_cls_logits_pred, gt_img_cls)

        if len(seg_head_outputs) == 3:
            _det_output_from_seg_module = seg_head_outputs[0]
            actual_protos_tensor = seg_head_outputs[2]
        elif (
            len(seg_head_outputs) == 2
            and isinstance(seg_head_outputs[1], (list, tuple))
            and len(seg_head_outputs[1]) == 2
        ):
            (
                _det_output_from_seg_module,
                (_potentially_3d_mask_coeffs, actual_protos_tensor),
            ) = seg_head_outputs
        else:
            raise ValueError(
                f"Critical Error: seg_head_outputs has an unhandled structure. "
                f"Expected 3-element tuple (det_internal, mask_coeffs_3D_or_protos_3D, protos_4D) OR "
                f"2-element tuple (det_internal, (mask_coeffs_3D, protos_4D)). "
                f"Got length {len(seg_head_outputs)}. Full output: {seg_head_outputs}"
            )

        if not (
            isinstance(actual_protos_tensor, torch.Tensor)
            and actual_protos_tensor.ndim == 4
        ):
            raise ValueError(
                f"actual_protos_tensor for seg_proto_projector must be 4D. Got shape: {actual_protos_tensor.shape if isinstance(actual_protos_tensor, torch.Tensor) else type(actual_protos_tensor)}. This was derived from seg_head_outputs."
            )
        if actual_protos_tensor.shape[1] != self.hparams.proto_ch:
            raise ValueError(
                f"actual_protos_tensor channel mismatch. Expected {self.hparams.proto_ch}, got {actual_protos_tensor.shape[1]}. Shape: {actual_protos_tensor.shape}"
            )

        seg_logits_projected = self.seg_proto_projector(actual_protos_tensor)
        seg_logits_resized = F.interpolate(
            seg_logits_projected,
            size=(self.hparams.img_size, self.hparams.img_size),
            mode="bilinear",
            align_corners=False,
        )
        loss_seg = self.seg_loss_fn(seg_logits_resized, gt_masks)
        self.seg_logits_for_logging = seg_logits_resized  # Store for logging
        loss_seg = self.seg_loss_fn(
            seg_logits_resized, gt_masks
        )  # Use local variable for loss

        current_device = det_head_outputs[0].device
        if not hasattr(self, "project") or self.project.device != current_device:
            self.project = torch.arange(
                self.reg_max, device=current_device, dtype=torch.float32
            )

        batch_size = det_head_outputs[0].shape[0]
        strides = [self.hparams.img_size / feat.shape[-1] for feat in det_head_outputs]

        pred_boxes_cat_list, pred_cls_logits_cat_list, pred_box_dist_cat_list = (
            [],
            [],
            [],
        )
        anchor_points_cat_list, stride_tensor_cat_list = [], []

        for i, pred_fm in enumerate(det_head_outputs):
            bs, ch, h, w = pred_fm.shape
            stride_val = strides[i]
            pred_fm_flat = pred_fm.permute(0, 2, 3, 1).reshape(bs, h * w, ch)
            box_dist_preds_raw = pred_fm_flat[..., : self.reg_max * 4]
            cls_logits_preds = pred_fm_flat[..., self.reg_max * 4 :]
            box_dist_reshaped = box_dist_preds_raw.view(bs, h * w, 4, self.reg_max)
            box_dist_probs = F.softmax(box_dist_reshaped, dim=-1)
            decoded_ltrb_dists = torch.einsum(
                "ijkl,l->ijk", box_dist_probs, self.project
            )

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=current_device, dtype=torch.float32),
                torch.arange(w, device=current_device, dtype=torch.float32),
                indexing="ij",
            )
            anchor_points_level = (
                torch.stack((grid_x + 0.5, grid_y + 0.5), dim=-1)
                .view(1, h * w, 2)
                .repeat(bs, 1, 1)
            )
            pred_boxes_xyxy_level = dist2bbox(
                decoded_ltrb_dists * stride_val, anchor_points_level * stride_val
            )

            pred_boxes_cat_list.append(pred_boxes_xyxy_level)
            pred_cls_logits_cat_list.append(cls_logits_preds)
            pred_box_dist_cat_list.append(
                box_dist_preds_raw.view(bs, h * w, 4, self.reg_max)
            )
            anchor_points_cat_list.append(anchor_points_level)
            stride_tensor_cat_list.append(
                torch.full(
                    (bs, h * w, 1),
                    stride_val,
                    device=current_device,
                    dtype=torch.float32,
                )
            )

        pred_boxes_abs_xyxy = torch.cat(pred_boxes_cat_list, dim=1)
        pred_cls_logits = torch.cat(pred_cls_logits_cat_list, dim=1)
        pred_box_dist_for_dfl = torch.cat(pred_box_dist_cat_list, dim=1)
        anchor_points_all = torch.cat(anchor_points_cat_list, dim=1)
        stride_tensor_all = torch.cat(stride_tensor_cat_list, dim=1)

        loss_box_iou_accum, loss_cls_det_accum, loss_dfl_accum = 0.0, 0.0, 0.0
        num_total_pos_matches = 0

        if (
            self.training
        ):  # Only clear for training, for validation it's cleared at epoch end
            self.temp_matched_preds_for_cm = []

        for b_idx in range(batch_size):
            gt_boxes_item_info = gt_det_boxes[gt_det_boxes[:, 0] == b_idx]
            if gt_boxes_item_info.numel() == 0:
                continue

            gt_classes_item = gt_boxes_item_info[:, 1].long()
            gt_boxes_cxcywh_norm_item = gt_boxes_item_info[:, 2:6]
            gt_boxes_xyxy_abs_item = torch.cat(
                [
                    (
                        gt_boxes_cxcywh_norm_item[:, 0]
                        - gt_boxes_cxcywh_norm_item[:, 2] / 2
                    )
                    * self.hparams.img_size,
                    (
                        gt_boxes_cxcywh_norm_item[:, 1]
                        - gt_boxes_cxcywh_norm_item[:, 3] / 2
                    )
                    * self.hparams.img_size,
                    (
                        gt_boxes_cxcywh_norm_item[:, 0]
                        + gt_boxes_cxcywh_norm_item[:, 2] / 2
                    )
                    * self.hparams.img_size,
                    (
                        gt_boxes_cxcywh_norm_item[:, 1]
                        + gt_boxes_cxcywh_norm_item[:, 3] / 2
                    )
                    * self.hparams.img_size,
                ],
                dim=-1,
            ).view(-1, 4)

            pred_boxes_item = pred_boxes_abs_xyxy[b_idx]
            pred_cls_logits_item = pred_cls_logits[b_idx]
            pred_box_dist_item = pred_box_dist_for_dfl[b_idx]
            anchor_points_item = anchor_points_all[b_idx]
            stride_tensor_item = stride_tensor_all[b_idx]

            if gt_boxes_xyxy_abs_item.numel() == 0 or pred_boxes_item.numel() == 0:
                continue
            ious_matrix = batch_bbox_iou(pred_boxes_item, gt_boxes_xyxy_abs_item)
            if ious_matrix.numel() == 0:
                continue

            pred_max_iou_vals, pred_best_gt_indices = ious_matrix.max(dim=1)
            positive_mask = pred_max_iou_vals > self.hparams.iou_match_thresh
            num_item_pos = positive_mask.sum()

            if num_item_pos > 0:
                num_total_pos_matches += num_item_pos

                matched_pred_boxes = pred_boxes_item[positive_mask]
                matched_gt_boxes = gt_boxes_xyxy_abs_item[
                    pred_best_gt_indices[positive_mask]
                ]
                iou_for_loss = batch_bbox_iou(
                    matched_pred_boxes, matched_gt_boxes
                ).diag()
                loss_box_iou_accum += (1.0 - iou_for_loss).sum()

                matched_pred_cls_logits = pred_cls_logits_item[positive_mask]
                matched_gt_classes = gt_classes_item[
                    pred_best_gt_indices[positive_mask]
                ]
                cls_targets_one_hot = F.one_hot(
                    matched_gt_classes, num_classes=self.hparams.nc_det
                ).float()
                loss_cls_det_accum += self.det_cls_loss_fn(
                    matched_pred_cls_logits, cls_targets_one_hot
                )

                # Store for CM only if in validation and loss calculation is active for val
                # This logic is a bit intertwined, but _multitask_loss is called for val too
                if (
                    not self.training
                ):  # Check if we are in validation context via this call
                    self.temp_matched_preds_for_cm.extend(
                        list(
                            zip(
                                matched_pred_cls_logits.argmax(dim=-1).tolist(),
                                matched_gt_classes.tolist(),
                            )
                        )
                    )

                gt_ltrb_target = (
                    torch.cat(
                        [
                            anchor_points_item[positive_mask]
                            * stride_tensor_item[positive_mask]
                            - matched_gt_boxes[:, :2],
                            matched_gt_boxes[:, 2:]
                            - anchor_points_item[positive_mask]
                            * stride_tensor_item[positive_mask],
                        ],
                        dim=-1,
                    )
                    / stride_tensor_item[positive_mask]
                )
                gt_ltrb_target = gt_ltrb_target.clamp(min=0, max=self.reg_max - 1.01)

                tl = gt_ltrb_target.floor().long()
                tr = tl + 1
                wl = tr.float() - gt_ltrb_target
                wr = gt_ltrb_target - tl.float()
                tl = tl.clamp(min=0, max=self.reg_max - 1)
                tr = tr.clamp(min=0, max=self.reg_max - 1)

                loss_dfl_item = 0.0
                target_pred_dist_for_dfl = pred_box_dist_item[positive_mask]
                for k_side in range(4):
                    loss_dfl_item_side_left = (
                        self.det_dfl_loss_fn(
                            target_pred_dist_for_dfl[:, k_side, :], tl[:, k_side]
                        )
                        * wl[:, k_side]
                    )
                    loss_dfl_item_side_right = (
                        self.det_dfl_loss_fn(
                            target_pred_dist_for_dfl[:, k_side, :], tr[:, k_side]
                        )
                        * wr[:, k_side]
                    )
                    loss_dfl_item += (
                        loss_dfl_item_side_left.sum() + loss_dfl_item_side_right.sum()
                    )
                loss_dfl_accum += loss_dfl_item

        avg_factor = (
            num_total_pos_matches if num_total_pos_matches > 0 else float(batch_size)
        )  # Prevent division by zero

        loss_box_iou_avg = loss_box_iou_accum / avg_factor
        loss_cls_det_avg = loss_cls_det_accum / avg_factor
        loss_dfl_avg = loss_dfl_accum / avg_factor

        total_loss = (
            self.hparams.loss_weight_seg * loss_seg
            + self.hparams.loss_weight_box_iou * loss_box_iou_avg
            + self.hparams.loss_weight_dfl * loss_dfl_avg
            + self.hparams.loss_weight_cls_det * loss_cls_det_avg
            + self.hparams.loss_weight_img_cls * loss_img_cls
        )

        return (
            total_loss,
            loss_seg,
            loss_box_iou_avg,
            loss_dfl_avg,
            loss_cls_det_avg,
            loss_img_cls,
        )

    def forward(self, x, mode="train"):
        return self.net(x, mode=mode)

    def training_step(self, batch, batch_idx):
        ids, imgs, det_boxes_gt, masks_gt, img_cls_gt = batch
        det_head_outputs, seg_head_outputs, img_cls_logits_pred = self(
            imgs, mode="train"
        )

        # _multitask_loss will populate self.seg_logits_for_logging
        total_loss, loss_s, loss_b_iou, loss_dfl, loss_c_det, loss_icls = (
            self._multitask_loss(
                det_head_outputs,
                seg_head_outputs,
                img_cls_logits_pred,
                det_boxes_gt,
                masks_gt,
                img_cls_gt,
            )
        )

        self.train_img_acc.update(img_cls_logits_pred, img_cls_gt)
        self.log_dict(
            {
                "train/loss_total": total_loss,
                "train/loss_seg": loss_s,
                "train/loss_box_iou": loss_b_iou,
                "train/loss_dfl": loss_dfl,
                "train/loss_det_cls": loss_c_det,
                "train/loss_img_cls": loss_icls,
                "train/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Log classification metrics using the new logger
        if self.global_step > 0 and self.global_step % self.hparams.cls_log_period == 0:
            if self.logger and hasattr(self.logger.experiment, "log"):
                log_cls_metrics(
                    self.logger,
                    img_cls_logits_pred,
                    img_cls_gt,
                    class_id_to_name=self.img_class_names,  # Use your class name mapping
                    stage="train",
                    step=self.global_step,
                )

        # Log segmentation examples using the new logger
        if (
            self.global_step > 0
            and self.global_step % self.hparams.mask_log_period == 0
        ):
            if (
                self.logger
                and hasattr(self.logger.experiment, "log")
                and hasattr(self, "seg_logits_for_logging")
            ):
                log_seg_examples(
                    self.logger,
                    imgs,
                    self.seg_logits_for_logging,  # Use pre-computed logits
                    masks_gt=masks_gt,
                    stage="train",
                    step=self.global_step,
                    max_samples=MAX_VIZ_PER_CALL,
                )
        # Log detection examples (typically not done every training step due to overhead)
        # If you need it, adapt the validation_step logic for detection logging here.

        # Manually commit if any loggers used commit=False, PTL usually handles this.
        if (
            self.logger
            and hasattr(self.logger.experiment, "log")
            and self.global_step > 0
            and (
                self.global_step % self.hparams.cls_log_period == 0
                or self.global_step % self.hparams.mask_log_period == 0
            )
        ):
            self.logger.experiment.log({}, commit=True, step=self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        ids, imgs, det_boxes_gt, masks_gt, img_cls_gt = batch
        # For validation, typically use "eval" mode if model behavior differs (e.g., dropout)
        # However, to be consistent with loss calculation and existing structure, using "train" mode output.
        # If "eval" mode is strictly needed for mAP, a second forward pass might be considered.
        det_outputs, seg_outputs, img_cls_logits = self(imgs, mode="train")

        # _multitask_loss will populate self.seg_logits_for_logging
        total_loss, loss_s, loss_b_iou, loss_dfl, loss_c_det, loss_icls = (
            self._multitask_loss(
                det_outputs,
                seg_outputs,
                img_cls_logits,
                det_boxes_gt,
                masks_gt,
                img_cls_gt,
            )
        )

        self.val_img_acc.update(img_cls_logits, img_cls_gt)
        self.val_img_cm.update(img_cls_logits.argmax(dim=-1), img_cls_gt)

        # Log classification metrics for validation batch (optional, epoch metrics are more common)
        # if self.logger and hasattr(self.logger.experiment, 'log'):
        # log_cls_metrics(self.logger, img_cls_logits, img_cls_gt,
        # class_id_to_name=self.img_class_names,
        # stage="val_batch", step=self.global_step) # Logged per batch

        # Segmentation F1 score (remains the same)
        if len(seg_outputs) == 3:
            actual_protos_tensor_val = seg_outputs[2]
        elif (
            len(seg_outputs) == 2
            and isinstance(seg_outputs[1], (list, tuple))
            and len(seg_outputs[1]) == 2
        ):
            _, (_, actual_protos_tensor_val) = seg_outputs
        # ... (rest of seg_f1 logic) ...
        else:
            print(
                f"Warning: Unexpected seg_outputs structure in validation_step for seg_f1."
            )
            actual_protos_tensor_val = None  # Handle appropriately

        if (
            actual_protos_tensor_val is not None
            and isinstance(actual_protos_tensor_val, torch.Tensor)
            and actual_protos_tensor_val.ndim == 4
            and actual_protos_tensor_val.shape[1] == self.hparams.proto_ch
        ):
            seg_logits_projected_val = self.seg_proto_projector(
                actual_protos_tensor_val
            )
            seg_logits_resized_val = F.interpolate(
                seg_logits_projected_val,
                size=(self.hparams.img_size, self.hparams.img_size),
                mode="bilinear",
                align_corners=False,
            )
            self.val_seg_f1.update(seg_logits_resized_val.sigmoid(), masks_gt.int())
        else:
            print(
                f"Warning: Skipping F1 update due to issues with actual_protos_tensor_val for seg_f1."
            )

        # --- Prepare predictions and targets for mAP and new detection logger ---
        map_preds_for_metric = []  # For torchmetrics mAP
        map_targets_for_metric = []
        det_preds_for_log = []  # For new log_det_examples
        det_gts_for_log = []  # For new log_det_examples

        batch_size_val = imgs.shape[0]
        strides_val = [self.hparams.img_size / feat.shape[-1] for feat in det_outputs]
        current_device_val = imgs.device
        if (
            not hasattr(self, "project_val")
            or self.project_val.device != current_device_val
        ):
            self.project_val = torch.arange(
                self.reg_max, device=current_device_val, dtype=torch.float32
            )

        # Decode detection outputs
        (
            pred_boxes_batch_list,
            pred_cls_scores_batch_list,
            pred_cls_indices_batch_list,
        ) = [], [], []

        for i_fm, pred_fm_val in enumerate(det_outputs):
            bs_v, ch_v, h_v, w_v = pred_fm_val.shape
            stride_val_v = strides_val[i_fm]
            pred_fm_flat_v = pred_fm_val.permute(0, 2, 3, 1).reshape(
                bs_v, h_v * w_v, ch_v
            )
            box_dist_raw_v = pred_fm_flat_v[..., : self.reg_max * 4]
            cls_logits_v = pred_fm_flat_v[..., self.reg_max * 4 :]

            box_dist_reshaped_v = box_dist_raw_v.view(bs_v, h_v * w_v, 4, self.reg_max)
            box_dist_probs_v = F.softmax(box_dist_reshaped_v, dim=-1)
            decoded_ltrb_v = torch.einsum(
                "ijkl,l->ijk", box_dist_probs_v, self.project_val
            )

            grid_y_v, grid_x_v = torch.meshgrid(
                torch.arange(h_v, device=current_device_val, dtype=torch.float32),
                torch.arange(w_v, device=current_device_val, dtype=torch.float32),
                indexing="ij",
            )
            anchor_points_v = (
                torch.stack((grid_x_v + 0.5, grid_y_v + 0.5), dim=-1)
                .view(1, h_v * w_v, 2)
                .repeat(bs_v, 1, 1)
            )
            pred_boxes_xyxy_v = dist2bbox(
                decoded_ltrb_v * stride_val_v, anchor_points_v * stride_val_v
            )

            pred_boxes_batch_list.append(pred_boxes_xyxy_v)  # List of [B, N_level, 4]
            pred_cls_scores_batch_list.append(
                cls_logits_v.sigmoid()
            )  # List of [B, N_level, Nc_det]

            # Post-process per image in the batch
        for b_val_idx in range(batch_size_val):
            # -------- 1) concat predictions from all feature levels ------------
            item_boxes = torch.cat(
                [lvl[b_val_idx] for lvl in pred_boxes_batch_list], dim=0
            )  # [N, 4]
            item_cls_scores = torch.cat(
                [lvl[b_val_idx] for lvl in pred_cls_scores_batch_list], dim=0
            )  # [N, Nc]

            # -------- 2) pick best class per box -------------------------------
            item_top_scores, item_top_labels = item_cls_scores.max(dim=1)  # [N]

            # -------- 3) conf-threshold & clamp -------------------------------
            keep = item_top_scores > CONF_TH
            if keep.sum() == 0:
                # no predictions – push empties and move on
                map_preds_for_metric.append(
                    {
                        "boxes": torch.empty((0, 4), device="cpu"),
                        "scores": torch.empty((0,), device="cpu"),
                        "labels": torch.empty((0,), device="cpu", dtype=torch.long),
                    }
                )
                det_preds_for_log.append(torch.empty((0, 6), device=current_device_val))
            else:
                item_boxes = item_boxes[keep].clamp_(0, self.hparams.img_size)
                item_top_scores = item_top_scores[keep]
            item_top_labels = item_top_labels[keep]

            # -------- 4) per-image NMS + top-K -----------------------------
            keep_idx = torchvision.ops.nms(item_boxes, item_top_scores, NMS_IOU)[:TOP_K]
            item_boxes = item_boxes[keep_idx]
            item_top_scores = item_top_scores[keep_idx]
            item_top_labels = item_top_labels[keep_idx]

            # ----- mAP input (CPU tensors) ---------------------------------
            map_preds_for_metric.append(
                {
                    "boxes": item_boxes.cpu(),
                    "scores": item_top_scores.cpu(),
                    "labels": item_top_labels.cpu(),
                }
            )

            # ----- logger input (GPU OK) -----------------------------------
            det_preds_for_log.append(
                torch.cat(
                    [
                        item_boxes,
                        item_top_scores.unsqueeze(1),
                        item_top_labels.unsqueeze(1).float(),
                    ],
                    dim=1,
                )
            )

        # ----------------- 5) build ground-truth ---------------------------
        gt_mask = det_boxes_gt[:, 0] == b_val_idx
        gt_this = det_boxes_gt[gt_mask]

        if gt_this.numel() > 0:
            gt_cxywh = gt_this[:, 2:6]
            gt_xyxy = (
                torch.cat(
                    [
                        (gt_cxywh[:, 0] - gt_cxywh[:, 2] / 2) * self.hparams.img_size,
                        (gt_cxywh[:, 1] - gt_cxywh[:, 3] / 2) * self.hparams.img_size,
                        (gt_cxywh[:, 0] + gt_cxywh[:, 2] / 2) * self.hparams.img_size,
                        (gt_cxywh[:, 1] + gt_cxywh[:, 3] / 2) * self.hparams.img_size,
                    ],
                    dim=-1,
                )
                .view(-1, 4)
                .clamp_(0, self.hparams.img_size)
            )

            gt_labels = gt_this[:, 1].long()
            map_targets_for_metric.append(
                {"boxes": gt_xyxy.cpu(), "labels": gt_labels.cpu()}
            )

            det_gts_for_log.append(
                torch.cat([gt_xyxy, gt_labels.unsqueeze(1).float()], dim=1)
            )
        else:
            map_targets_for_metric.append(
                {
                    "boxes": torch.empty((0, 4), device="cpu"),
                    "labels": torch.empty((0,), device="cpu", dtype=torch.long),
                }
            )
            det_gts_for_log.append(torch.empty((0, 5), device=current_device_val))
            if map_preds_for_metric and map_targets_for_metric:
                self.val_map_iou50.update(map_preds_for_metric, map_targets_for_metric)
                if (self.current_epoch % MAP_FULL_FREQ) == 0:
                    self.val_map_iou50_95.update(
                        map_preds_for_metric, map_targets_for_metric
                    )

            self.log_dict(
                {
                    "val/loss_total": total_loss,
                    "val/loss_seg": loss_s,
                    "val/loss_box_iou": loss_b_iou,
                    "val/loss_dfl": loss_dfl,
                    "val/loss_det_cls": loss_c_det,
                    "val/loss_img_cls": loss_icls,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        # Log detection examples using the new logger
        if batch_idx == 0 and self.current_epoch % self.hparams.box_log_period == 0:
            if self.logger and hasattr(self.logger.experiment, "log"):
                if det_preds_for_log:  # Check if there's something to log
                    log_det_examples(
                        self.logger,
                        imgs,
                        det_preds_for_log,
                        gts=det_gts_for_log,
                        class_id_to_name=self.det_class_names,
                        stage="val",
                        step=self.global_step,
                        conf_th=self.hparams.det_conf_thresh_viz,
                        max_samples=MAX_VIZ_PER_CALL,
                    )
        # Log segmentation examples for validation (first batch only)
        if (
            batch_idx == 0 and self.current_epoch % self.hparams.mask_log_period == 0
        ):  # Or a different val log period
            if (
                self.logger
                and hasattr(self.logger.experiment, "log")
                and hasattr(self, "seg_logits_for_logging")
            ):
                log_seg_examples(
                    self.logger,
                    imgs,
                    self.seg_logits_for_logging,
                    masks_gt=masks_gt,
                    stage="val",
                    step=self.global_step,
                    max_samples=MAX_VIZ_PER_CALL,
                )

        # Manually commit if any loggers used commit=False
        if (
            self.logger
            and hasattr(self.logger.experiment, "log")
            and batch_idx == 0
            and (
                self.current_epoch % self.hparams.box_log_period == 0
                or self.current_epoch % self.hparams.mask_log_period == 0
            )
        ):  # Check other conditions if val batch cls metrics are logged
            self.logger.experiment.log({}, commit=True, step=self.global_step)

        return total_loss

    def on_train_epoch_end(self):
        self.log(
            "train_epoch/img_accuracy_epoch",
            self.train_img_acc.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_img_acc.reset()
        # Note: log_cls_metrics in training_step logs batch-wise P/R/F1/Acc.
        # If epoch-wise P/R/F1 for classification is needed, implement with torchmetrics like val_img_acc.

    def on_validation_epoch_end(self):
        print(f"\nEpoch {self.current_epoch}: Validation epoch end starting...")
        epoch_val_end_start_time = time.time()

        # Image Classification Metrics (Accuracy from torchmetrics, CM from torchmetrics)
        current_op_time = time.time()
        print("  Computing and logging image accuracy (epoch)...")
        val_img_accuracy = self.val_img_acc.compute()
        self.log(
            "val_epoch/img_accuracy_epoch",
            val_img_accuracy,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        print(
            f"    Image accuracy logged: {val_img_accuracy:.4f} (took {time.time() - current_op_time:.4f}s)"
        )

        current_op_time = time.time()
        print("  Computing and logging image confusion matrix (epoch)...")
        img_cm_tensor = self.val_img_cm.compute()
        img_cm_fig = plot_confusion_matrix_to_wandb(img_cm_tensor, self.img_class_names)
        if img_cm_fig and self.logger and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log(
                {
                    "val_epoch/img_confusion_matrix_epoch": img_cm_fig,
                    "global_step": self.global_step,
                }
            )
        print(
            f"    Image CM plotted & logged. (took {time.time() - current_op_time:.4f}s)"
        )

        # If you want epoch-level Precision, Recall, F1 for image classification (like log_cls_metrics does for batches):
        # You would need to accumulate all img_cls_logits and img_cls_gt from validation_step
        # then compute them here using torchmetrics.functional or your log_cls_metrics logic on the aggregated tensors.
        # For simplicity, if batch-wise from log_cls_metrics (if called in val_step) is enough, this is not needed.
        # Otherwise, you'd add P/R/F1 torchmetrics objects, update them in val_step, and compute here.

        # Segmentation F1 score (from torchmetrics)
        current_op_time = time.time()
        print("  Computing and logging segmentation F1 score (epoch)...")
        val_seg_f1_score = self.val_seg_f1.compute()
        self.log(
            "val_epoch/seg_f1_score_epoch",
            val_seg_f1_score,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        print(
            f"    Segmentation F1 logged: {val_seg_f1_score:.4f} (took {time.time() - current_op_time:.4f}s)"
        )

        current_op_time_log = time.time()
        # --- mAP@0.5:0.95 (full COCO curve) ------------------------------------
        # Run it sparsely to keep validation fast
        if (self.current_epoch % MAP_FULL_FREQ) == 0:
            current_op_time = time.time()
            print("  Computing mAP@0.5:0.95 metrics…")
            try:
                map_metrics_50_95 = self.val_map_iou50_95.compute()
                elapsed = time.time() - current_op_time
                print(f"    Done in {elapsed:.4f}s")
            except Exception as e_map:
                print(f"    ERROR computing mAP@0.5:0.95: {e_map}")
                map_metrics_50_95 = None

            # ── logging --------------------------------------------------------
            log_map_dict_50_95 = {}
            if map_metrics_50_95:
                for k, v in map_metrics_50_95.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        log_map_dict_50_95[f"val_epoch_map_iou50_95/{k}"] = v.item()
                    # per-class AP only if you turned class_metrics=True
                    elif (
                        k == "map_per_class"
                        and isinstance(v, torch.Tensor)
                        and v.ndim == 1
                    ):
                        for i_cls, ap_val in enumerate(v):
                            if i_cls < self.hparams.nc_det:
                                cls_name = self.det_class_names.get(
                                    i_cls, f"cls_{i_cls}"
                                )
                                log_map_dict_50_95[
                                    f"val_epoch_map_iou50_95_class/{cls_name}"
                                ] = ap_val.item()

                if log_map_dict_50_95:
                    self.log_dict(log_map_dict_50_95, logger=True, sync_dist=True)
            else:
                print("    mAP@0.5:0.95 empty — nothing logged.")

            # Reset metric so state doesn’t accumulate
            self.val_map_iou50_95.reset()

        else:
            # Skip this epoch, but still reset the internal buffers
            self.val_map_iou50_95.reset()
            print(
                f"  Skipping mAP@0.5:0.95 (only every {MAP_FULL_FREQ} epochs, "
                f"current = {self.current_epoch})."
            )
        # --- mAP@0.5 ---
        current_op_time = time.time()
        print("  Computing mAP@0.5 metrics...")
        map_metrics_50 = None
        try:
            map_metrics_50 = self.val_map_iou50.compute()
            print(
                f"    mAP@0.5 metrics computed. (took {time.time() - current_op_time:.4f}s)"
            )
        except Exception as e_map_compute:
            print(f"    ERROR computing mAP@0.5: {e_map_compute}")

        current_op_time_log = time.time()
        log_map_dict_50 = {}
        if map_metrics_50:
            for k, v in map_metrics_50.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    log_map_dict_50[f"val_epoch_map_iou50/{k}"] = v.item()
                elif (
                    k == "map_per_class" and isinstance(v, torch.Tensor) and v.ndim == 1
                ):
                    for i_cls, ap_val in enumerate(v):  # Renamed i to i_cls
                        if i_cls < self.hparams.nc_det:
                            log_map_dict_50[
                                f"val_epoch_map_iou50_class/{self.det_class_names.get(i_cls, f'cls_{i_cls}')}"
                            ] = ap_val.item()
            if log_map_dict_50:
                self.log_dict(
                    log_map_dict_50, logger=True, sync_dist=True
                )  # prog_bar=False
            print(
                f"    mAP@0.5 metrics logged. (took {time.time() - current_op_time_log:.4f}s for logging part)"
            )
        else:
            print("    mAP@0.5 metrics were empty or None, skipping logging.")

        # Detection Confusion Matrix (from torchmetrics, using temp_matched_preds_for_cm)
        current_op_time = time.time()
        print("  Computing and logging detection confusion matrix (epoch)...")
        # ... (det_cm logic remains the same) ...
        if self.temp_matched_preds_for_cm:
            preds_cm_list, gts_cm_list = zip(*self.temp_matched_preds_for_cm)
            if preds_cm_list and gts_cm_list:  # Check if lists are non-empty
                try:
                    preds_tensor = torch.tensor(
                        list(preds_cm_list), device=self.device, dtype=torch.long
                    )
                    gts_tensor = torch.tensor(
                        list(gts_cm_list), device=self.device, dtype=torch.long
                    )
                    self.val_det_cm.update(preds_tensor, gts_tensor)
                    det_cm_tensor = self.val_det_cm.compute()
                    det_cm_fig = plot_confusion_matrix_to_wandb(
                        det_cm_tensor, self.det_class_names
                    )
                    if (
                        det_cm_fig
                        and self.logger
                        and hasattr(self.logger.experiment, "log")
                    ):
                        self.logger.experiment.log(
                            {
                                "val_epoch/det_confusion_matrix_epoch": det_cm_fig,
                                "global_step": self.global_step,
                            }
                        )
                except Exception as e_det_cm:
                    print(f"    Error processing detection CM: {e_det_cm}")
            else:  # This case means temp_matched_preds_for_cm had content but one of the lists after zip was empty
                print(
                    "    preds_cm_list or gts_cm_list became empty after zip, skipping det CM."
                )
        else:
            print("    No matched predictions stored for detection CM this epoch.")
        print(
            f"  Logged val_epoch/det_confusion_matrix_epoch (took {time.time() - current_op_time:.4f}s)"
        )

        # Reset metrics
        current_op_time = time.time()
        print("  Resetting metrics...")
        self.val_img_acc.reset()
        self.val_img_cm.reset()
        self.val_seg_f1.reset()
        self.val_map_iou50_95.reset()
        self.val_map_iou50.reset()
        self.val_det_cm.reset()
        self.temp_matched_preds_for_cm = []  # Crucial to clear for the next validation epoch
        print(f"    Metrics reset. (took {time.time() - current_op_time:.4f}s)")
        print(
            f"Epoch {self.current_epoch}: Validation epoch end finished. Total time: {time.time() - epoch_val_end_start_time:.2f}s\n"
        )

    # configure_optimizers method remains the same
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(
            trainable_params, lr=self.hparams.lr, weight_decay=0.0005
        )
        t_max_epochs = (
            self.trainer.max_epochs
            if self.trainer
            and self.trainer.max_epochs is not None
            and self.trainer.max_epochs > 0
            else 50
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=t_max_epochs, eta_min=self.hparams.lr * 0.01
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }


# ────────────────────────────────────────────────── DataModule
class BTXRDDataModule(pl.LightningDataModule):
    def __init__(self, root="btxrd_ready", batch_size=4, num_workers=4, img_size=640):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set = BTXRD(
            self.hparams.root, split="train", img_size=self.hparams.img_size
        )
        self.val_set = BTXRD(
            self.hparams.root, split="val", img_size=self.hparams.img_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.hparams.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.hparams.num_workers > 0),
        )


# ────────────────────────────────────────────────── Main
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(123, workers=True)
    CLS_LOG_PERIOD = 10
    IMG_SIZE = 640
    BATCH_SIZE = 4  # Reduce if OOM, try 2 or 1
    NUM_WORKERS = 2  # Reduce if issues, try 0
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 100
    NC_DET = 2  # Number of detection classes
    NUM_IMG_CLASSES = 2  # Number of image classification classes
    PROTO_CH = 32
    IOU_MATCH_THRESH = 0.5  # For assigning positive anchors in loss
    LOSS_WEIGHT_SEG = 1.0
    LOSS_WEIGHT_BOX_IOU = 7.5
    LOSS_WEIGHT_DFL = 1.5
    LOSS_WEIGHT_CLS_DET = 0.5
    LOSS_WEIGHT_IMG_CLS = 1.0
    MASK_LOG_PERIOD = 10  # Log segmentation masks every N global steps
    BOX_LOG_PERIOD = 10  # Log detection examples every N validation epochs
    DET_CONF_THRESH_VIZ = 0.1  # Confidence threshold for visualizing detection boxes
    MAP_MAX_DETECTIONS = 300
    MAP_FULL_FREQ = 5
    # Check if your dataset has more objects per image on average.
    # Setting to a single value applies it across all recall thresholds.
    # Or use [100, 100, 100] if you want to match the previous [self.hparams.map_max_detections]*3

    wandb_logger = WandbLogger(
        project="BTXRD-MultiTask-AdvancedMetrics-v2", log_model="all"
    )  # New project name or version

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints_wandb_{wandb_logger.version if wandb_logger.version else 'local'}",
        filename="btrxd-multitask-{epoch:02d}-{val/loss_total:.2f}-{val_epoch_map_iou50_95/map:.3f}",  # Added mAP to filename
        save_top_k=3,
        monitor="val_epoch_map_iou50/map",  # Monitor mAP for checkpointing
        mode="max",  # mAP is maximized
        save_last=True,
    )
    # You can also add a checkpoint for val_loss if needed
    # model_checkpoint_loss = ModelCheckpoint(
    #     dirpath=f"checkpoints_wandb_{wandb_logger.version if wandb_logger.version else 'local'}/loss",
    #     filename='btrxd-loss-{epoch:02d}-{val/loss_total:.2f}',
    #     save_top_k=1, monitor='val/loss_total', mode='min')

    early_stopping = EarlyStopping(
        monitor="val_epoch_map_iou50/map",  # Monitor mAP for early stopping
        patience=20,
        mode="max",  # mAP is maximized
        verbose=True,
    )

    data_module = BTXRDDataModule(
        root="btxrd_ready",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )
    model = MultiTaskLitModel(
        img_size=IMG_SIZE,
        lr=LEARNING_RATE,
        mask_log_period=MASK_LOG_PERIOD,
        box_log_period=BOX_LOG_PERIOD,
        cls_log_period=CLS_LOG_PERIOD,  # Pass the new hparam
        nc_det=NC_DET,
        num_img_classes=NUM_IMG_CLASSES,
        proto_ch=PROTO_CH,
        loss_weight_seg=LOSS_WEIGHT_SEG,
        loss_weight_box_iou=LOSS_WEIGHT_BOX_IOU,
        loss_weight_dfl=LOSS_WEIGHT_DFL,
        loss_weight_cls_det=LOSS_WEIGHT_CLS_DET,
        loss_weight_img_cls=LOSS_WEIGHT_IMG_CLS,
        iou_match_thresh=IOU_MATCH_THRESH,
        det_conf_thresh_viz=DET_CONF_THRESH_VIZ,
        map_max_detections=MAP_MAX_DETECTIONS,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        gradient_clip_val=10.0,
        log_every_n_steps=10,  # Log training loss more frequently if desired
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[lr_monitor, model_checkpoint, early_stopping],
    )  # Add model_checkpoint_loss if using

    print("Starting training with PyTorch Lightning...")
    try:
        # Find learning rate (optional, but good practice)
        # tuner = pl.tuner.Tuner(trainer)
        # lr_finder = tuner.lr_find(model, datamodule=data_module, num_training=100) # Run for 100 batches
        # print(f"Optimal LR found: {lr_finder.suggestion()}")
        # model.hparams.lr = lr_finder.suggestion() # Update learning rate
        # fig = lr_finder.plot(suggest=True)
        # if wandb.run: wandb.log({"lr_finder_plot": wandb.Image(fig)})

        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if wandb.run:
            wandb.finish()
    print("Training finished")
