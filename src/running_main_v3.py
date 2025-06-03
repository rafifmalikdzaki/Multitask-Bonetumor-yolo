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

from torchmetrics.classification import (
    BinaryPrecision, # For segmentation
    BinaryRecall,    # For segmentation
    BinaryAccuracy,  # For segmentation
    MulticlassAccuracy, # For image classification
    MulticlassConfusionMatrix # For image classification
)
from torchmetrics.segmentation import (
    DiceScore,  # Dice Coefficient for segmentation
    # JaccardIndex # <<< REMOVED >>>
)
from torchmetrics import F1Score as TorchMetricsF1Score # Can be used for image cls or binary seg

import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE pyplot import
import matplotlib.pyplot as plt


import io
import time  # For timing diagnostics
from typing import List, Dict, Optional, Sequence, Mapping, Any # Expanded typing
import numpy as np
from multitask_logging import log_cls_metrics, log_seg_examples, log_det_examples

# Assuming these files are in the same directory or in PYTHONPATH
from main_model import ConvNeXtBiFPNYOLO, load_pretrained_heads # Ensure these exist
from dataset_btxrdv2 import BTXRD, collate_fn # Ensure these exist

from torchmetrics.detection import MeanAveragePrecision # Used for bbox detection AND segm mAP
import torchvision


import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

# These constants can be tuned project‑wide in one place.
LOG_FREQ_TRAIN = 100
MAX_VIZ_PER_CALL = 10
CONF_TH = 0.05  # Initial confidence threshold for considering boxes before NMS
NMS_IOU = 0.6
TOP_K = 100 # Number of boxes to keep per image after NMS for mAP and visualization


def _img_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in −1…1 *or* 0…1 → HWC uint8 (0‑255)."""
    if img.dim() != 3:
        raise ValueError("Expected CHW image tensor, got dim=%d" % img.dim())
    img = img.detach().cpu()
    if img.min() < -1e-5: # More robust check for -1 to 1 range
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
        figsize=(max(6, len(class_names_list)*0.9), max(5, len(class_names_list) * 0.8))
    )
    cm_np = cm_tensor.cpu().numpy()
    annotation_format = ".2f" if np.issubdtype(cm_np.dtype, np.floating) else "d"


    sns.heatmap(
        cm_np, annot=True, fmt=annotation_format, cmap="Blues",
        xticklabels=class_names_list, yticklabels=class_names_list,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

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
        mask_log_period: int = 100,
        box_log_period: int = 1,
        cls_log_period: int = 50,
        nc_det: int = 3,
        num_img_classes: int = 2,
        proto_ch: int = 32,
        loss_weight_seg: float = 1.0,
        loss_weight_box_iou: float = 2.0,
        loss_weight_dfl: float = 1.5,
        loss_weight_cls_det: float = 0.5,
        loss_weight_img_cls: float = 1.0,
        iou_match_thresh: float = 0.5,
        det_conf_thresh_viz: float = 0.25,
        map_max_detections: int = 300,
        map_full_freq: int = 5,
        det_label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            "img_size", "lr", "mask_log_period", "box_log_period", "cls_log_period",
            "nc_det", "num_img_classes", "proto_ch",
            "loss_weight_seg", "loss_weight_box_iou", "loss_weight_dfl",
            "loss_weight_cls_det", "loss_weight_img_cls",
            "iou_match_thresh", "det_conf_thresh_viz", "map_max_detections",
            "map_full_freq",
            "det_label_smoothing",
        )

        self.net = ConvNeXtBiFPNYOLO(
            nc_det=self.hparams.nc_det,
            nc_img=self.hparams.num_img_classes,
            proto_ch=self.hparams.proto_ch,
        )
        self.seg_proto_projector = nn.Conv2d(self.hparams.proto_ch, 1, kernel_size=1)

        self.img_cls_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = nn.BCEWithLogitsLoss()
        self.det_cls_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.det_dfl_loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.train_img_acc = MulticlassAccuracy(num_classes=self.hparams.num_img_classes, average="micro", dist_sync_on_step=True)
        self.val_img_acc = MulticlassAccuracy(num_classes=self.hparams.num_img_classes, average="micro", dist_sync_on_step=True)
        self.val_img_cm = MulticlassConfusionMatrix(num_classes=self.hparams.num_img_classes, normalize="true", dist_sync_on_step=True)

        # Segmentation Metrics
        self.val_seg_f1 = TorchMetricsF1Score(task="binary", threshold=0.5, dist_sync_on_step=True)
        self.val_seg_prec = BinaryPrecision(threshold=0.5, dist_sync_on_step=True)
        self.val_seg_rec = BinaryRecall(threshold=0.5, dist_sync_on_step=True)
        self.val_seg_acc = BinaryAccuracy(threshold=0.5, dist_sync_on_step=True)
        self.val_seg_dice = DiceScore(num_classes=1, dist_sync_on_step=True) # num_classes=1 for binary dice
        # <<< ADDED MeanAveragePrecision for Segmentation >>>
        # For binary segmentation, class_metrics=False is fine.
        # The metric will calculate mAP over IoU thresholds for the single foreground class.
        self.val_seg_map = MeanAveragePrecision(iou_type='segm', class_metrics=False, dist_sync_on_step=True)


        self.val_map_iou50 = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=False,
            max_detection_thresholds=[1, 10, self.hparams.map_max_detections], dist_sync_on_step=True,
        )
        self.val_map_iou50_95 = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", iou_thresholds=torch.linspace(0.5, 0.95, 10).tolist(),
            class_metrics=False, max_detection_thresholds=[1, 10, self.hparams.map_max_detections],
            dist_sync_on_step=True,
        )
        self.val_det_cm = MulticlassConfusionMatrix(num_classes=self.hparams.nc_det, normalize="true", dist_sync_on_step=True)
        self.temp_matched_preds_for_cm = []

        self.reg_max = (self.net.detect.reg_max if hasattr(self.net.detect, "reg_max") else 16)
        
        self.project: Optional[torch.Tensor] = None
        self.project_val: Optional[torch.Tensor] = None


        self.img_class_names: Dict[int, str] = {i: f"imgC{i}" for i in range(self.hparams.num_img_classes)}
        self.det_class_names: Dict[int, str] = {i: f"detC{i}" for i in range(self.hparams.nc_det)}
        self.seg_logits_for_logging: Optional[torch.Tensor] = None


    def _multitask_loss(
        self,
        det_head_outputs, seg_head_outputs, img_cls_logits_pred,
        gt_det_boxes, gt_masks, gt_img_cls,
    ):
        loss_img_cls = self.img_cls_loss_fn(img_cls_logits_pred, gt_img_cls)

        if len(seg_head_outputs) == 3:
            actual_protos_tensor = seg_head_outputs[2]
        elif (len(seg_head_outputs) == 2 and isinstance(seg_head_outputs[1], (list, tuple)) and len(seg_head_outputs[1]) == 2):
            _, (_, actual_protos_tensor) = seg_head_outputs
        else:
            raise ValueError(f"Critical Error: seg_head_outputs structure. Got {len(seg_head_outputs)}")

        if not (isinstance(actual_protos_tensor, torch.Tensor) and actual_protos_tensor.ndim == 4):
            raise ValueError(f"actual_protos_tensor must be 4D. Shape: {actual_protos_tensor.shape if isinstance(actual_protos_tensor, torch.Tensor) else type(actual_protos_tensor)}")
        if actual_protos_tensor.shape[1] != self.hparams.proto_ch:
            raise ValueError(f"actual_protos_tensor channel mismatch. Expected {self.hparams.proto_ch}, got {actual_protos_tensor.shape[1]}")

        seg_logits_projected = self.seg_proto_projector(actual_protos_tensor)
        seg_logits_resized = F.interpolate(
            seg_logits_projected, size=(self.hparams.img_size, self.hparams.img_size),
            mode="bilinear", align_corners=False,
        )
        self.seg_logits_for_logging = seg_logits_resized 
        loss_seg = self.seg_loss_fn(seg_logits_resized, gt_masks)

        current_device = det_head_outputs[0].device
        if self.project is None or self.project.device != current_device: 
            self.project = torch.arange(self.reg_max, device=current_device, dtype=torch.float32)

        batch_size = det_head_outputs[0].shape[0]
        strides = [self.hparams.img_size / feat.shape[-1] for feat in det_head_outputs]
        pred_boxes_cat_list, pred_cls_logits_cat_list, pred_box_dist_cat_list = [], [], []
        anchor_points_cat_list, stride_tensor_cat_list = [], []

        for i, pred_fm in enumerate(det_head_outputs):
            bs, ch_fm, h, w = pred_fm.shape 
            stride_val = strides[i]
            pred_fm_flat = pred_fm.permute(0, 2, 3, 1).reshape(bs, h * w, ch_fm)
            box_dist_preds_raw = pred_fm_flat[..., : self.reg_max * 4]
            cls_logits_preds = pred_fm_flat[..., self.reg_max * 4 :]
            box_dist_reshaped = box_dist_preds_raw.view(bs, h * w, 4, self.reg_max)
            box_dist_probs = F.softmax(box_dist_reshaped, dim=-1)
            decoded_ltrb_dists = torch.einsum("ijkl,l->ijk", box_dist_probs, self.project) 
            grid_y, grid_x = torch.meshgrid(torch.arange(h, device=current_device, dtype=torch.float32), torch.arange(w, device=current_device, dtype=torch.float32), indexing="ij")
            anchor_points_level = torch.stack((grid_x + 0.5, grid_y + 0.5), dim=-1).view(1, h * w, 2).repeat(bs, 1, 1)
            pred_boxes_xyxy_level = dist2bbox(decoded_ltrb_dists * stride_val, anchor_points_level * stride_val)
            pred_boxes_cat_list.append(pred_boxes_xyxy_level)
            pred_cls_logits_cat_list.append(cls_logits_preds)
            pred_box_dist_cat_list.append(box_dist_preds_raw.view(bs, h * w, 4, self.reg_max))
            anchor_points_cat_list.append(anchor_points_level)
            stride_tensor_cat_list.append(torch.full((bs, h * w, 1), stride_val, device=current_device, dtype=torch.float32))

        pred_boxes_abs_xyxy = torch.cat(pred_boxes_cat_list, dim=1)
        pred_cls_logits = torch.cat(pred_cls_logits_cat_list, dim=1)
        pred_box_dist_for_dfl = torch.cat(pred_box_dist_cat_list, dim=1)
        anchor_points_all = torch.cat(anchor_points_cat_list, dim=1)
        stride_tensor_all = torch.cat(stride_tensor_cat_list, dim=1)

        loss_box_iou_accum, loss_cls_det_accum, loss_dfl_accum = 0.0, 0.0, 0.0
        num_total_pos_matches = 0 
        accum_iou_for_matched_pairs = 0.0 

        if self.training: self.temp_matched_preds_for_cm = [] 

        for b_idx in range(batch_size):
            gt_boxes_item_info = gt_det_boxes[gt_det_boxes[:, 0] == b_idx]
            if gt_boxes_item_info.numel() == 0: continue
            gt_classes_item = gt_boxes_item_info[:, 1].long()
            gt_boxes_cxcywh_norm_item = gt_boxes_item_info[:, 2:6]
            gt_boxes_xyxy_abs_item = torch.cat([
                (gt_boxes_cxcywh_norm_item[:, 0] - gt_boxes_cxcywh_norm_item[:, 2] / 2) * self.hparams.img_size,
                (gt_boxes_cxcywh_norm_item[:, 1] - gt_boxes_cxcywh_norm_item[:, 3] / 2) * self.hparams.img_size,
                (gt_boxes_cxcywh_norm_item[:, 0] + gt_boxes_cxcywh_norm_item[:, 2] / 2) * self.hparams.img_size,
                (gt_boxes_cxcywh_norm_item[:, 1] + gt_boxes_cxcywh_norm_item[:, 3] / 2) * self.hparams.img_size,
            ], dim=-1,).view(-1, 4)

            pred_boxes_item = pred_boxes_abs_xyxy[b_idx]
            pred_cls_logits_item = pred_cls_logits[b_idx]
            pred_box_dist_item = pred_box_dist_for_dfl[b_idx]
            anchor_points_item = anchor_points_all[b_idx]
            stride_tensor_item = stride_tensor_all[b_idx]

            if gt_boxes_xyxy_abs_item.numel() == 0 or pred_boxes_item.numel() == 0: continue
            ious_matrix = batch_bbox_iou(pred_boxes_item, gt_boxes_xyxy_abs_item)
            if ious_matrix.numel() == 0: continue

            pred_max_iou_vals, pred_best_gt_indices = ious_matrix.max(dim=1)
            positive_mask = pred_max_iou_vals > self.hparams.iou_match_thresh
            num_item_pos_tensor = positive_mask.sum() 
            num_item_pos = num_item_pos_tensor.item() 

            if num_item_pos > 0:
                num_total_pos_matches += num_item_pos 

                matched_pred_boxes = pred_boxes_item[positive_mask]
                matched_gt_boxes = gt_boxes_xyxy_abs_item[pred_best_gt_indices[positive_mask]]
                iou_for_loss_current_item = batch_bbox_iou(matched_pred_boxes, matched_gt_boxes).diag()
                loss_box_iou_accum += (1.0 - iou_for_loss_current_item).sum()
                accum_iou_for_matched_pairs += iou_for_loss_current_item.sum().item() 

                matched_pred_cls_logits = pred_cls_logits_item[positive_mask]
                matched_gt_classes = gt_classes_item[pred_best_gt_indices[positive_mask]]
                
                if self.hparams.det_label_smoothing > 0.0 and self.training: 
                    smoothing = self.hparams.det_label_smoothing
                    num_classes = self.hparams.nc_det
                    confidence = 1.0 - smoothing
                    cls_targets_soft = torch.full_like(matched_pred_cls_logits, smoothing / (num_classes - 1))
                    cls_targets_soft.scatter_(-1, matched_gt_classes.unsqueeze(1).long(), confidence)
                    det_cls_target = cls_targets_soft
                else: 
                    det_cls_target = F.one_hot(matched_gt_classes, num_classes=self.hparams.nc_det).float()
                
                loss_cls_det_accum += self.det_cls_loss_fn(matched_pred_cls_logits, det_cls_target)

                if not self.training:
                    self.temp_matched_preds_for_cm.extend(list(zip(matched_pred_cls_logits.argmax(dim=-1).tolist(), matched_gt_classes.tolist())))

                gt_ltrb_target = (torch.cat([
                    anchor_points_item[positive_mask] * stride_tensor_item[positive_mask] - matched_gt_boxes[:, :2],
                    matched_gt_boxes[:, 2:] - anchor_points_item[positive_mask] * stride_tensor_item[positive_mask],
                ], dim=-1,) / stride_tensor_item[positive_mask]).clamp(min=0, max=self.reg_max - 1.01)
                
                tl = gt_ltrb_target.floor().long().clamp(min=0, max=self.reg_max - 1)
                tr = (tl + 1).clamp(min=0, max=self.reg_max - 1)
                wl = tr.float() - gt_ltrb_target
                wr = gt_ltrb_target - tl.float()
                
                loss_dfl_item = 0.0
                target_pred_dist_for_dfl = pred_box_dist_item[positive_mask]
                for k_side in range(4):
                    loss_dfl_item_side_left = (self.det_dfl_loss_fn(target_pred_dist_for_dfl[:, k_side, :], tl[:, k_side]) * wl[:, k_side])
                    loss_dfl_item_side_right = (self.det_dfl_loss_fn(target_pred_dist_for_dfl[:, k_side, :], tr[:, k_side]) * wr[:, k_side])
                    loss_dfl_item += (loss_dfl_item_side_left.sum() + loss_dfl_item_side_right.sum())
                loss_dfl_accum += loss_dfl_item
        
        avg_iou_of_matches = (accum_iou_for_matched_pairs / num_total_pos_matches) if num_total_pos_matches > 0 else 0.0

        avg_factor = float(num_total_pos_matches) if num_total_pos_matches > 0 else float(batch_size) 
        loss_box_iou_avg = loss_box_iou_accum / avg_factor
        loss_cls_det_avg = loss_cls_det_accum / avg_factor
        loss_dfl_avg = loss_dfl_accum / avg_factor

        total_loss = (
            self.hparams.loss_weight_seg * loss_seg +
            self.hparams.loss_weight_box_iou * loss_box_iou_avg +
            self.hparams.loss_weight_dfl * loss_dfl_avg +
            self.hparams.loss_weight_cls_det * loss_cls_det_avg +
            self.hparams.loss_weight_img_cls * loss_img_cls
        )
        if self.training:
            return (total_loss, loss_seg, loss_box_iou_avg, loss_dfl_avg, loss_cls_det_avg, loss_img_cls, float(num_total_pos_matches), float(avg_iou_of_matches))
        else:
            return (total_loss, loss_seg, loss_box_iou_avg, loss_dfl_avg, loss_cls_det_avg, loss_img_cls)


    def forward(self, x, mode="train"):
        return self.net(x, mode=mode)

    def training_step(self, batch, batch_idx):
        ids, imgs, det_boxes_gt, masks_gt, img_cls_gt = batch
        if self.training and batch_idx == 0 and self.current_epoch == 0 : 
            self.temp_matched_preds_for_cm = []

        det_head_outputs, seg_head_outputs, img_cls_logits_pred = self(imgs, mode="train")

        total_loss, loss_s, loss_b_iou, loss_dfl, loss_c_det, loss_icls, num_pos_matches, avg_iou_matches = (
            self._multitask_loss(
                det_head_outputs, seg_head_outputs, img_cls_logits_pred,
                det_boxes_gt, masks_gt, img_cls_gt,
            )
        )

        self.train_img_acc.update(img_cls_logits_pred, img_cls_gt)
        
        log_payload_step = {
            "train_step/loss_total": total_loss, "train_step/loss_seg": loss_s,
            "train_step/loss_box_iou": loss_b_iou, "train_step/loss_dfl": loss_dfl,
            "train_step/loss_det_cls": loss_c_det, "train_step/loss_img_cls": loss_icls,
            "train_step/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "train_step/num_positive_matches": num_pos_matches, 
            "train_step/avg_iou_of_matches": avg_iou_matches  
        }
        self.log_dict(log_payload_step, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_step/loss_total_progbar", total_loss, prog_bar=True, on_step=True, on_epoch=False, logger=False) 

        log_payload_epoch = {
            "train_epoch/loss_total_epoch": total_loss, "train_epoch/loss_seg_epoch": loss_s,
            "train_epoch/loss_box_iou_epoch": loss_b_iou, "train_epoch/loss_dfl_epoch": loss_dfl,
            "train_epoch/loss_det_cls_epoch": loss_c_det, "train_epoch/loss_img_cls_epoch": loss_icls,
            "train_epoch/num_positive_matches_epoch": num_pos_matches,
            "train_epoch/avg_iou_of_matches_epoch": avg_iou_matches
        }
        self.log_dict(log_payload_epoch, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


        if self.global_step > 0 and self.global_step % self.hparams.cls_log_period == 0:
            if self.logger and hasattr(self.logger.experiment, "log"):
                log_cls_metrics(
                    self.logger, img_cls_logits_pred, img_cls_gt,
                    class_id_to_name=self.img_class_names,
                    log_prefix="train_step/cls_batch", 
                    step=self.global_step,
                )

        if (self.global_step > 0 and self.global_step % self.hparams.mask_log_period == 0):
            if (self.logger and hasattr(self.logger.experiment, "log") and self.seg_logits_for_logging is not None):
                log_seg_examples(
                    self.logger, imgs, self.seg_logits_for_logging, masks_gt=masks_gt,
                    stage="train", step=self.global_step, max_samples=MAX_VIZ_PER_CALL,
                )
        return total_loss

    def validation_step(self, batch, batch_idx):
        ids, imgs, det_boxes_gt, masks_gt, img_cls_gt = batch
        det_outputs, seg_outputs, img_cls_logits = self(imgs, mode="train") 

        total_loss, loss_s, loss_b_iou, loss_dfl, loss_c_det, loss_icls = (
            self._multitask_loss( 
                det_outputs, seg_outputs, img_cls_logits,
                det_boxes_gt, masks_gt, img_cls_gt,
            )
        )

        self.val_img_acc.update(img_cls_logits, img_cls_gt)
        self.val_img_cm.update(img_cls_logits.argmax(dim=-1), img_cls_gt)

        # <<< MODIFIED: Segmentation metrics update and mAP seg data prep >>>
        map_seg_preds_batch: List[Dict[str, torch.Tensor]] = []
        map_seg_targets_batch: List[Dict[str, torch.Tensor]] = []

        if self.seg_logits_for_logging is not None:
            seg_probs_val = self.seg_logits_for_logging.sigmoid() # (B, 1, H, W)
            seg_gt_val_int = masks_gt.int() # (B, 1, H, W)

            # Update standard binary metrics
            self.val_seg_f1.update(seg_probs_val, seg_gt_val_int)
            self.val_seg_prec.update(seg_probs_val, seg_gt_val_int)
            self.val_seg_rec.update(seg_probs_val, seg_gt_val_int)
            self.val_seg_acc.update(seg_probs_val, seg_gt_val_int)
            self.val_seg_dice.update(seg_probs_val, seg_gt_val_int)
            # self.val_seg_jaccard.update(seg_probs_val, seg_gt_val_int) # Jaccard removed

            # Prepare data for segmentation mAP
            for i in range(seg_probs_val.shape[0]): # Iterate through batch
                pred_mask_bool = (seg_probs_val[i] > 0.5) # (1, H, W) boolean
                
                # Calculate score: average probability of the predicted foreground pixels
                # Add a small epsilon to prevent division by zero if pred_mask_bool is all False
                score_tensor = (seg_probs_val[i] * pred_mask_bool.float()).sum() / (pred_mask_bool.float().sum() + 1e-6)
                
                map_seg_preds_batch.append({
                    'masks': pred_mask_bool.cpu(), # (1, H, W)
                    'scores': score_tensor.unsqueeze(0).cpu(), # (1,)
                    'labels': torch.tensor([0], device='cpu', dtype=torch.long) # Single class 0
                })

                target_mask_bool = (seg_gt_val_int[i] > 0.5) # (1, H, W) boolean
                map_seg_targets_batch.append({
                    'masks': target_mask_bool.cpu(), # (1, H, W)
                    'labels': torch.tensor([0], device='cpu', dtype=torch.long) # Single class 0
                })
            
            if map_seg_preds_batch and map_seg_targets_batch:
                 self.val_seg_map.update(map_seg_preds_batch, map_seg_targets_batch)

        else:
            print(f"Warning: self.seg_logits_for_logging is None in validation_step for batch {batch_idx}. Segmentation metrics might be inaccurate.")


        map_preds_for_metric: List[Dict[str, torch.Tensor]] = []
        map_targets_for_metric: List[Dict[str, torch.Tensor]] = []
        det_preds_for_log: List[torch.Tensor] = []
        det_gts_for_log: List[torch.Tensor] = []

        batch_size_val = imgs.shape[0]
        strides_val = [self.hparams.img_size / feat.shape[-1] for feat in det_outputs]
        current_device_val = imgs.device
        if self.project_val is None or self.project_val.device != current_device_val:
            self.project_val = torch.arange(self.reg_max, device=current_device_val, dtype=torch.float32)

        pred_boxes_decoded_levels: List[torch.Tensor] = []
        pred_cls_scores_decoded_levels: List[torch.Tensor] = []

        for i_fm, pred_fm_val in enumerate(det_outputs):
            bs_v, ch_v, h_v, w_v = pred_fm_val.shape; stride_val_v = strides_val[i_fm]
            pred_fm_flat_v = pred_fm_val.permute(0, 2, 3, 1).reshape(bs_v, h_v * w_v, ch_v)
            box_dist_raw_v = pred_fm_flat_v[..., : self.reg_max * 4]
            cls_logits_v = pred_fm_flat_v[..., self.reg_max * 4 :]
            box_dist_reshaped_v = box_dist_raw_v.view(bs_v, h_v * w_v, 4, self.reg_max)
            box_dist_probs_v = F.softmax(box_dist_reshaped_v, dim=-1)
            decoded_ltrb_v = torch.einsum("ijkl,l->ijk", box_dist_probs_v, self.project_val)
            grid_y_v, grid_x_v = torch.meshgrid(torch.arange(h_v, device=current_device_val, dtype=torch.float32), torch.arange(w_v, device=current_device_val, dtype=torch.float32), indexing="ij")
            anchor_points_v = torch.stack((grid_x_v + 0.5, grid_y_v + 0.5), dim=-1).view(1, h_v * w_v, 2).repeat(bs_v, 1, 1)
            pred_boxes_xyxy_v_level = dist2bbox(decoded_ltrb_v * stride_val_v, anchor_points_v * stride_val_v)
            pred_boxes_decoded_levels.append(pred_boxes_xyxy_v_level)
            pred_cls_scores_decoded_levels.append(cls_logits_v.sigmoid())
        
        batch_pred_boxes_all_levels = torch.cat(pred_boxes_decoded_levels, dim=1)
        batch_pred_cls_scores_all_levels = torch.cat(pred_cls_scores_decoded_levels, dim=1)

        for b_val_idx in range(batch_size_val):
            item_boxes_all = batch_pred_boxes_all_levels[b_val_idx]
            item_cls_scores_all = batch_pred_cls_scores_all_levels[b_val_idx]
            item_top_scores, item_top_labels = item_cls_scores_all.max(dim=1)
            keep_conf = item_top_scores > CONF_TH 
            
            current_item_boxes_final = torch.empty((0, 4), device=current_device_val)
            current_item_scores_final = torch.empty((0,), device=current_device_val)
            current_item_labels_final = torch.empty((0,), device=current_device_val, dtype=torch.long)

            if keep_conf.sum() > 0:
                item_boxes_kept = item_boxes_all[keep_conf].clamp_(0, self.hparams.img_size)
                item_top_scores_kept = item_top_scores[keep_conf]
                item_top_labels_kept = item_top_labels[keep_conf]
                keep_nms_idx = torchvision.ops.nms(item_boxes_kept, item_top_scores_kept, NMS_IOU)[:TOP_K] 
                current_item_boxes_final = item_boxes_kept[keep_nms_idx]
                current_item_scores_final = item_top_scores_kept[keep_nms_idx]
                current_item_labels_final = item_top_labels_kept[keep_nms_idx]

            map_preds_for_metric.append({"boxes": current_item_boxes_final.cpu(), "scores": current_item_scores_final.cpu(), "labels": current_item_labels_final.cpu()})
            if current_item_boxes_final.numel() > 0:
                det_preds_for_log.append(torch.cat([current_item_boxes_final, current_item_scores_final.unsqueeze(1), current_item_labels_final.unsqueeze(1).float()], dim=1))
            else:
                det_preds_for_log.append(torch.empty((0, 6), device=current_device_val))

            gt_mask = det_boxes_gt[:, 0] == b_val_idx; gt_this_item = det_boxes_gt[gt_mask]
            current_gt_boxes_cpu = torch.empty((0, 4), device="cpu")
            current_gt_labels_cpu = torch.empty((0,), device="cpu", dtype=torch.long)
            current_gt_for_log = torch.empty((0,5), device=current_device_val)
            if gt_this_item.numel() > 0:
                gt_cxywh_item = gt_this_item[:, 2:6]
                gt_xyxy_item = (torch.cat([(gt_cxywh_item[:,0] - gt_cxywh_item[:,2]/2)*self.hparams.img_size, (gt_cxywh_item[:,1] - gt_cxywh_item[:,3]/2)*self.hparams.img_size, (gt_cxywh_item[:,0] + gt_cxywh_item[:,2]/2)*self.hparams.img_size, (gt_cxywh_item[:,1] + gt_cxywh_item[:,3]/2)*self.hparams.img_size], dim=-1).view(-1,4).clamp_(0,self.hparams.img_size))
                gt_labels_item = gt_this_item[:,1].long()
                current_gt_boxes_cpu = gt_xyxy_item.cpu(); current_gt_labels_cpu = gt_labels_item.cpu()
                current_gt_for_log = torch.cat([gt_xyxy_item, gt_labels_item.unsqueeze(1).float()], dim=1)
            map_targets_for_metric.append({"boxes": current_gt_boxes_cpu, "labels": current_gt_labels_cpu})
            det_gts_for_log.append(current_gt_for_log)

        if map_preds_for_metric and map_targets_for_metric: 
            self.val_map_iou50.update(map_preds_for_metric, map_targets_for_metric)
            self.val_map_iou50_95.update(map_preds_for_metric, map_targets_for_metric)


        self.log_dict({
            "val_epoch/loss_total": total_loss, "val_epoch/loss_seg": loss_s,
            "val_epoch/loss_box_iou": loss_b_iou, "val_epoch/loss_dfl": loss_dfl,
            "val_epoch/loss_det_cls": loss_c_det, "val_epoch/loss_img_cls": loss_icls,
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_epoch/loss_total_progbar", total_loss, prog_bar=True, on_step=False, on_epoch=True, logger=False)


        if batch_idx == 0 and (self.current_epoch % self.hparams.box_log_period == 0):
            if self.logger and hasattr(self.logger.experiment, "log") and any(p.numel() > 0 for p in det_preds_for_log):
                log_det_examples(
                    self.logger, imgs, det_preds_for_log, gts=det_gts_for_log,
                    class_id_to_name=self.det_class_names, stage="val", step=self.global_step,
                    conf_th=self.hparams.det_conf_thresh_viz, max_samples=MAX_VIZ_PER_CALL,
                )
        if batch_idx == 0 and (self.current_epoch % self.hparams.mask_log_period == 0):
            if (self.logger and hasattr(self.logger.experiment, "log") and self.seg_logits_for_logging is not None):
                log_seg_examples(
                    self.logger, imgs, self.seg_logits_for_logging, masks_gt=masks_gt,
                    stage="val", step=self.global_step, max_samples=MAX_VIZ_PER_CALL,
                )
        return total_loss

    def on_train_epoch_end(self):
        self.log("train_epoch/img_accuracy_epoch", self.train_img_acc.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.train_img_acc.reset()

    def on_validation_epoch_end(self):
        print(f"\nEpoch {self.current_epoch}: Validation epoch end starting...")
        epoch_val_end_start_time = time.time()

        val_img_accuracy = self.val_img_acc.compute()
        self.log("val_epoch/img_accuracy_epoch", val_img_accuracy, prog_bar=True, logger=True, sync_dist=True)
        print(f"    Image accuracy logged: {val_img_accuracy:.4f} (took {time.time() - epoch_val_end_start_time:.4f}s)") 

        img_cm_tensor = self.val_img_cm.compute()
        img_cm_fig = plot_confusion_matrix_to_wandb(img_cm_tensor, self.img_class_names)
        if img_cm_fig and self.logger and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({"val_epoch/img_confusion_matrix_epoch": img_cm_fig, "global_step": self.global_step})
        
        seg_f1 = self.val_seg_f1.compute()
        seg_precision = self.val_seg_prec.compute()
        seg_recall = self.val_seg_rec.compute()
        seg_accuracy = self.val_seg_acc.compute()
        seg_dice = self.val_seg_dice.compute()
        
        # <<< COMPUTE & LOG Segmentation mAP >>>
        seg_map_metrics = self.val_seg_map.compute()
        log_seg_map_dict = {}
        if seg_map_metrics:
            for k, v_tensor in seg_map_metrics.items():
                # Filter out 'classes' tensor and only log scalars
                if isinstance(v_tensor, torch.Tensor) and v_tensor.numel() == 1:
                    log_seg_map_dict[f"val_epoch/seg_map_{k}"] = v_tensor.item()
                elif isinstance(v_tensor, (int, float)): # Should not happen for mAP dict but good practice
                    log_seg_map_dict[f"val_epoch/seg_map_{k}"] = v_tensor

        self.log_dict({
            "val_epoch/seg_f1_epoch": seg_f1, "val_epoch/seg_precision_epoch": seg_precision,
            "val_epoch/seg_recall_epoch": seg_recall, "val_epoch/seg_accuracy_epoch": seg_accuracy,
            "val_epoch/seg_dice_epoch": seg_dice,
            **log_seg_map_dict, # Add seg mAP metrics
        }, prog_bar=False, logger=True, sync_dist=True) 
        
        self.log("val_epoch/seg_f1_epoch_progbar", seg_f1, prog_bar=True, logger=False, sync_dist=True)
        self.log("val_epoch/seg_dice_epoch_progbar", seg_dice, prog_bar=True, logger=False, sync_dist=True)
        if "val_epoch/seg_map_map" in log_seg_map_dict: # Log main seg_map to prog bar
             self.log("val_epoch/seg_map_progbar", log_seg_map_dict["val_epoch/seg_map_map"], prog_bar=True, logger=False, sync_dist=True)
        
        print(f"    Segmentation metrics logged. (F1: {seg_f1:.4f}, Dice: {seg_dice:.4f}, Seg_mAP: {log_seg_map_dict.get('val_epoch/seg_map_map', -1.0):.4f})")


        # Helper function to log mAP dictionary
        def _log_map_metrics_dict(metrics_dict_to_log, base_key_prefix, metric_object_for_class_check):
            log_to_wandb = {}
            if metrics_dict_to_log:
                for k, v_from_metric in metrics_dict_to_log.items():
                    full_metric_key = f"{base_key_prefix}_{k}" 
                    if k == "map_per_class": 
                        if metric_object_for_class_check.class_metrics and isinstance(v_from_metric, torch.Tensor):
                            for i_cls, ap_val_tensor in enumerate(v_from_metric):
                                if i_cls < self.hparams.nc_det:
                                    class_name_safe = self.det_class_names.get(i_cls, f'cls_{i_cls}')
                                    log_to_wandb[f"{base_key_prefix}_class_{class_name_safe}"] = ap_val_tensor.item()
                    elif isinstance(v_from_metric, torch.Tensor) and v_from_metric.numel() == 1:
                        metric_value = v_from_metric.item()
                        log_to_wandb[full_metric_key] = metric_value
                        if k == 'map' and hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback and \
                           full_metric_key == self.trainer.checkpoint_callback.monitor:
                            self.log(full_metric_key, metric_value, prog_bar=True, logger=False, sync_dist=True) 
                        elif k == 'map' and base_key_prefix == "val_epoch/map_iou50_95" and \
                             hasattr(self.trainer.checkpoint_callback, 'filename') and \
                             full_metric_key.replace('/', '_') in self.trainer.checkpoint_callback.filename : # Check if used in filename
                             self.log(full_metric_key, metric_value, logger=False, prog_bar=False, sync_dist=True)
                    elif isinstance(v_from_metric, (int, float)): 
                         log_to_wandb[full_metric_key] = v_from_metric
            
            if log_to_wandb:
                self.log_dict(log_to_wandb, logger=True, sync_dist=True) 
            return log_to_wandb


        if (self.current_epoch % self.hparams.map_full_freq) == 0: 
            map_metrics_50_95 = self.val_map_iou50_95.compute()
            print(f"    mAP@0.50:0.95 computed for epoch {self.current_epoch}: {map_metrics_50_95}")
            if map_metrics_50_95 and 'map' in map_metrics_50_95 and isinstance(map_metrics_50_95['map'], torch.Tensor) and map_metrics_50_95['map'].numel() == 1:
                map_50_95_value_for_filename = map_metrics_50_95['map'].item()
                # Check if the ModelCheckpoint filename uses this specific key format
                expected_filename_key = "val_epoch_map_iou50_95/map" # Note: PTL uses '/' not '_' for dict keys
                if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback and \
                   expected_filename_key in self.trainer.checkpoint_callback.filename:
                     self.log(expected_filename_key, map_50_95_value_for_filename, logger=True, prog_bar=False, sync_dist=True)

            _log_map_metrics_dict(map_metrics_50_95, "val_epoch/map_iou50_95", self.val_map_iou50_95)
        self.val_map_iou50_95.reset() 

        map_metrics_50 = self.val_map_iou50.compute()
        print(f"    mAP@0.50 computed: {map_metrics_50}")
        logged_map_50 = _log_map_metrics_dict(map_metrics_50, "val_epoch/map_iou50", self.val_map_iou50)
        
        # Ensure the primary monitored metric for checkpointing and early stopping is explicitly logged for prog_bar
        primary_monitor_key = "val_epoch_map_iou50/map" # This MUST match ModelCheckpoint's monitor
        if primary_monitor_key.replace('/', '_') in logged_map_50: # logged_map_50 uses '_' from helper
            self.log(primary_monitor_key, logged_map_50[primary_monitor_key.replace('/', '_')], prog_bar=True, logger=False, sync_dist=True)
        elif map_metrics_50 and 'map' in map_metrics_50: 
             map_val = map_metrics_50['map']
             if isinstance(map_val, torch.Tensor) and map_val.numel() == 1:
                self.log(primary_monitor_key, map_val.item(), prog_bar=True, logger=False, sync_dist=True)

        self.val_map_iou50.reset()


        if self.temp_matched_preds_for_cm:
            preds_cm_list, gts_cm_list = zip(*self.temp_matched_preds_for_cm)
            if preds_cm_list and gts_cm_list: 
                try:
                    preds_tensor = torch.tensor(list(preds_cm_list), device=self.device, dtype=torch.long)
                    gts_tensor = torch.tensor(list(gts_cm_list), device=self.device, dtype=torch.long)
                    self.val_det_cm.update(preds_tensor, gts_tensor)
                    det_cm_tensor = self.val_det_cm.compute()
                    det_cm_fig = plot_confusion_matrix_to_wandb(det_cm_tensor, self.det_class_names)
                    if det_cm_fig and self.logger and hasattr(self.logger.experiment, "log"):
                        self.logger.experiment.log({"val_epoch/det_confusion_matrix_epoch": det_cm_fig, "global_step": self.global_step})
                except Exception as e_cm:
                     print(f"Error computing/logging detection CM: {e_cm}")
        
        self.val_img_acc.reset(); self.val_img_cm.reset()
        self.val_seg_f1.reset(); self.val_seg_prec.reset(); self.val_seg_rec.reset()
        self.val_seg_acc.reset(); self.val_seg_dice.reset(); self.val_seg_map.reset() # <<< RESET Seg mAP >>>
        self.val_det_cm.reset()
        self.temp_matched_preds_for_cm = [] 
        print(f"    Metrics reset.")
        print(f"Epoch {self.current_epoch}: Validation epoch end finished. Total time: {time.time() - epoch_val_end_start_time:.2f}s\n")

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(trainable_params, lr=self.hparams.lr, weight_decay=0.0005)
        
        t_max_epochs = 50 
        if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs is not None and self.trainer.max_epochs > 0:
            t_max_epochs = self.trainer.max_epochs
        elif hasattr(self.hparams, 'max_epochs_from_main') and self.hparams.max_epochs_from_main > 0:
             t_max_epochs = self.hparams.max_epochs_from_main

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max_epochs, eta_min=self.hparams.lr * 0.01)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

# ────────────────────────────────────────────────── DataModule
class BTXRDDataModule(pl.LightningDataModule):
    def __init__(self, root="btxrd_ready", batch_size=4, num_workers=4, img_size=640):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set = BTXRD(self.hparams.root, split="train", img_size=self.hparams.img_size)
        self.val_set = BTXRD(self.hparams.root, split="val", img_size=self.hparams.img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, collate_fn=collate_fn, pin_memory=True,
            drop_last=True, persistent_workers=(self.hparams.num_workers > 0),
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, collate_fn=collate_fn, pin_memory=True,
            drop_last=False, persistent_workers=(self.hparams.num_workers > 0),
        )

# ────────────────────────────────────────────────── Main
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(123, workers=True)

    CLS_LOG_PERIOD = 10
    IMG_SIZE = 640
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-4 
    MAX_EPOCHS = 500
    NC_DET = 2
    NUM_IMG_CLASSES = 2
    PROTO_CH = 32
    IOU_MATCH_THRESH = 0.5
    LOSS_WEIGHT_SEG = 1.0
    LOSS_WEIGHT_BOX_IOU = 7.5 
    LOSS_WEIGHT_DFL = 1.5
    LOSS_WEIGHT_CLS_DET = 0.5
    LOSS_WEIGHT_IMG_CLS = 1.0
    MASK_LOG_PERIOD = 50
    BOX_LOG_PERIOD = 50
    DET_CONF_THRESH_VIZ = 0.25
    MAP_MAX_DETECTIONS = 100 
    MAP_FULL_FREQ = 5      
    DET_LABEL_SMOOTHING_VALUE = 0.1 

    wandb_logger = WandbLogger(project="BTXRD-MultiTask-AdvancedMetrics-v2", log_model=False)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints_wandb_{wandb_logger.version if wandb_logger.version else 'local'}",
        filename="btrxd-multitask-{epoch:02d}-{val_epoch/loss_total:.2f}-{val_epoch_map_iou50/map:.3f}", 
        save_top_k=2,
        monitor="val_epoch_map_iou50/map", 
        mode="max", save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_epoch_map_iou50/map", patience=50, mode="max", verbose=True)

    data_module = BTXRDDataModule(root="btxrd_ready", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=IMG_SIZE)
    model = MultiTaskLitModel(
        img_size=IMG_SIZE, lr=LEARNING_RATE,
        mask_log_period=MASK_LOG_PERIOD, box_log_period=BOX_LOG_PERIOD,
        cls_log_period=CLS_LOG_PERIOD, nc_det=NC_DET, num_img_classes=NUM_IMG_CLASSES,
        proto_ch=PROTO_CH, loss_weight_seg=LOSS_WEIGHT_SEG,
        loss_weight_box_iou=LOSS_WEIGHT_BOX_IOU, loss_weight_dfl=LOSS_WEIGHT_DFL,
        loss_weight_cls_det=LOSS_WEIGHT_CLS_DET, loss_weight_img_cls=LOSS_WEIGHT_IMG_CLS,
        iou_match_thresh=IOU_MATCH_THRESH, det_conf_thresh_viz=DET_CONF_THRESH_VIZ,
        map_max_detections=MAP_MAX_DETECTIONS,
        map_full_freq=MAP_FULL_FREQ, 
        det_label_smoothing=DET_LABEL_SMOOTHING_VALUE, 
    )
    model.hparams.max_epochs_from_main = MAX_EPOCHS 

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto", devices="auto", precision="bf16-mixed",
        gradient_clip_val=10.0, log_every_n_steps=10, check_val_every_n_epoch=1,
        logger=wandb_logger, callbacks=[lr_monitor, model_checkpoint, early_stopping],
    )

    print("Starting training with PyTorch Lightning...")
    try:
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if wandb.run:
            wandb.finish()
    print("Training finished")

