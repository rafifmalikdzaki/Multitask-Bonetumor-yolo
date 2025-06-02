# ────────────────────────────────────────────────── imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image  # Import PIL.Image
import time  # For timing diagnostics

# Assuming these files are in the same directory or in PYTHONPATH
from main_model import ConvNeXtBiFPNYOLO, load_pretrained_heads
from dataset_btxrd import BTXRD, collate_fn

from torchmetrics import F1Score as TorchMetricsF1Score  # Alias to avoid conflict
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.detection import MeanAveragePrecision


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
        mask_log_period: int = 100,
        box_log_period: int = 1,
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = ConvNeXtBiFPNYOLO(
            nc_det=self.hparams.nc_det,
            nc_img=self.hparams.num_img_classes,
            proto_ch=self.hparams.proto_ch,
        )
        load_pretrained_heads(self.net, ckpt_path="yolov8s-seg.pt")

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

        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            class_metrics=True,
            iou_type="bbox",
            dist_sync_on_step=True,
            max_detection_thresholds=[
                self.hparams.map_max_detections
            ],  # Corrected argument
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
        self.seg_class_labels_wandb = {0: "background", 1: "tumor_region"}

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

        if self.training:
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

                if not self.training:
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
        )

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

        if (
            self.global_step > 0
            and self.global_step % self.hparams.mask_log_period == 0
            and self.logger
            and hasattr(self.logger.experiment, "log")
        ):
            self._log_segmentation_example(
                imgs, seg_head_outputs, masks_gt, ids, batch_idx
            )
        return total_loss

    def validation_step(self, batch, batch_idx):
        ids, imgs, det_boxes_gt, masks_gt, img_cls_gt = batch
        det_train_out, seg_train_out, img_cls_train_logits = self(imgs, mode="train")

        total_loss, loss_s, loss_b_iou, loss_dfl, loss_c_det, loss_icls = (
            self._multitask_loss(
                det_train_out,
                seg_train_out,
                img_cls_train_logits,
                det_boxes_gt,
                masks_gt,
                img_cls_gt,
            )
        )

        self.val_img_acc.update(img_cls_train_logits, img_cls_gt)
        self.val_img_cm.update(img_cls_train_logits.argmax(dim=-1), img_cls_gt)

        if len(seg_train_out) == 3:
            actual_protos_tensor_val = seg_train_out[2]
        elif (
            len(seg_train_out) == 2
            and isinstance(seg_train_out[1], (list, tuple))
            and len(seg_train_out[1]) == 2
        ):
            _, (_, actual_protos_tensor_val) = seg_train_out
        else:
            print(
                f"Warning: Unexpected seg_train_out structure in validation_step for seg_f1. Using dummy tensor for protos."
            )
            actual_protos_tensor_val = torch.zeros(
                imgs.shape[0],
                self.hparams.proto_ch,
                self.hparams.img_size // 8,
                self.hparams.img_size // 8,
                device=imgs.device,
            )

        if not (
            isinstance(actual_protos_tensor_val, torch.Tensor)
            and actual_protos_tensor_val.ndim == 4
        ):
            print(
                f"Warning: actual_protos_tensor_val for seg_f1 metric must be 4D. Got shape: {actual_protos_tensor_val.shape if isinstance(actual_protos_tensor_val, torch.Tensor) else type(actual_protos_tensor_val)}. Skipping F1 update for this batch."
            )
        elif actual_protos_tensor_val.shape[1] != self.hparams.proto_ch:
            print(
                f"Warning: actual_protos_tensor_val channel mismatch for seg_f1. Expected {self.hparams.proto_ch}, got {actual_protos_tensor_val.shape[1]}. Skipping F1 update."
            )
        else:
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

        map_preds_batch = []
        map_targets_batch = []
        batch_size_val = imgs.shape[0]
        strides_val = [self.hparams.img_size / feat.shape[-1] for feat in det_train_out]

        current_device_val = imgs.device
        if (
            not hasattr(self, "project_val")
            or self.project_val.device != current_device_val
        ):
            self.project_val = torch.arange(
                self.reg_max, device=current_device_val, dtype=torch.float32
            )

        pred_boxes_val_list, pred_cls_scores_val_list = [], []

        for i_fm, pred_fm_val in enumerate(det_train_out):
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

            pred_boxes_val_list.append(pred_boxes_xyxy_v)
            pred_cls_scores_val_list.append(cls_logits_v.sigmoid())

        for b_val_idx in range(batch_size_val):
            item_boxes = torch.cat(
                [level_preds[b_val_idx] for level_preds in pred_boxes_val_list], dim=0
            )
            item_scores_all_cls = torch.cat(
                [level_scores[b_val_idx] for level_scores in pred_cls_scores_val_list],
                dim=0,
            )
            item_scores, item_labels = torch.max(item_scores_all_cls, dim=1)

            map_preds_batch.append(
                {
                    "boxes": item_boxes.clamp(min=0, max=self.hparams.img_size),
                    "scores": item_scores,
                    "labels": item_labels,
                }
            )

            gt_boxes_item_info_val = det_boxes_gt[det_boxes_gt[:, 0] == b_val_idx]
            gt_classes_item_val = gt_boxes_item_info_val[:, 1].long()
            gt_boxes_cxcywh_norm_val = gt_boxes_item_info_val[:, 2:6]
            gt_boxes_xyxy_abs_val = torch.cat(
                [
                    (
                        gt_boxes_cxcywh_norm_val[:, 0]
                        - gt_boxes_cxcywh_norm_val[:, 2] / 2
                    )
                    * self.hparams.img_size,
                    (
                        gt_boxes_cxcywh_norm_val[:, 1]
                        - gt_boxes_cxcywh_norm_val[:, 3] / 2
                    )
                    * self.hparams.img_size,
                    (
                        gt_boxes_cxcywh_norm_val[:, 0]
                        + gt_boxes_cxcywh_norm_val[:, 2] / 2
                    )
                    * self.hparams.img_size,
                    (
                        gt_boxes_cxcywh_norm_val[:, 1]
                        + gt_boxes_cxcywh_norm_val[:, 3] / 2
                    )
                    * self.hparams.img_size,
                ],
                dim=-1,
            ).view(-1, 4)
            map_targets_batch.append(
                {
                    "boxes": gt_boxes_xyxy_abs_val.clamp(
                        min=0, max=self.hparams.img_size
                    ),
                    "labels": gt_classes_item_val,
                }
            )

        if map_preds_batch and map_targets_batch:
            self.val_map.update(map_preds_batch, map_targets_batch)

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

        if batch_idx == 0 and self.current_epoch % self.hparams.box_log_period == 0:
            if map_preds_batch:
                self._log_detection_example(
                    imgs, map_preds_batch, map_targets_batch, ids, batch_idx
                )
        return total_loss

    def on_train_epoch_end(self):
        # print(f"Epoch {self.current_epoch}: Training epoch end.") # Diagnostic
        self.log(
            "train_epoch/img_accuracy",
            self.train_img_acc.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_img_acc.reset()

    def on_validation_epoch_end(self):
        print(f"\nEpoch {self.current_epoch}: Validation epoch end starting...")
        epoch_val_end_start_time = time.time()

        current_op_time = time.time()
        print("  Computing and logging image accuracy...")
        val_img_accuracy = self.val_img_acc.compute()
        self.log(
            "val_epoch/img_accuracy",
            val_img_accuracy,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        print(
            f"    Image accuracy logged: {val_img_accuracy:.4f} (took {time.time() - current_op_time:.4f}s)"
        )

        current_op_time = time.time()
        print("  Computing and logging image confusion matrix...")
        img_cm_tensor = self.val_img_cm.compute()
        print(f"    Image CM computed. (took {time.time() - current_op_time:.4f}s)")
        current_op_time_plot = time.time()
        img_cm_fig = plot_confusion_matrix_to_wandb(img_cm_tensor, self.img_class_names)
        if img_cm_fig and self.logger and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log(
                {
                    "val_epoch/img_confusion_matrix": img_cm_fig,
                    "global_step": self.global_step,
                }
            )
        print(
            f"    Image CM plotted & logged. (took {time.time() - current_op_time_plot:.4f}s)"
        )

        current_op_time = time.time()
        print("  Computing and logging segmentation F1 score...")
        val_seg_f1_score = self.val_seg_f1.compute()
        self.log(
            "val_epoch/seg_f1_score",
            val_seg_f1_score,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        print(
            f"    Segmentation F1 logged: {val_seg_f1_score:.4f} (took {time.time() - current_op_time:.4f}s)"
        )

        current_op_time = time.time()
        print("  Computing mAP metrics...")
        map_metrics = None
        try:
            map_metrics = self.val_map.compute()
            print(
                f"    mAP metrics computed. (took {time.time() - current_op_time:.4f}s)"
            )
        except Exception as e_map_compute:
            print(f"    ERROR computing mAP: {e_map_compute}")

        current_op_time_log = time.time()
        log_map_dict = {}
        if map_metrics:
            for k, v in map_metrics.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    log_map_dict[f"val_epoch_map/{k}"] = v.item()
                elif (
                    k == "map_per_class" and isinstance(v, torch.Tensor) and v.ndim == 1
                ):
                    for i, ap_val in enumerate(v):
                        if i < self.hparams.nc_det:
                            log_map_dict[
                                f"val_epoch_map_class/{self.det_class_names.get(i, f'cls_{i}')}"
                            ] = ap_val.item()

            if log_map_dict:
                self.log_dict(log_map_dict, logger=True, sync_dist=True)
            print(
                f"    mAP metrics logged. (took {time.time() - current_op_time_log:.4f}s for logging part)"
            )
        else:
            print("    mAP metrics were empty or None, skipping logging.")

        current_op_time = time.time()
        print("  Computing and logging detection confusion matrix...")
        if self.temp_matched_preds_for_cm:
            preds_cm_list, gts_cm_list = zip(*self.temp_matched_preds_for_cm)
            if preds_cm_list and gts_cm_list:
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
                                "val_epoch/det_confusion_matrix": det_cm_fig,
                                "global_step": self.global_step,
                            }
                        )
                except Exception as e_det_cm:
                    print(f"    Error processing detection CM: {e_det_cm}")
            else:
                print("    preds_cm or gts_cm is empty for detection CM.")
        else:
            print("    No matched predictions for detection CM this epoch.")
        print(
            f"  Logged val_epoch/det_confusion_matrix (took {time.time() - current_op_time:.4f}s)"
        )

        current_op_time = time.time()
        print("  Resetting metrics...")
        self.val_img_acc.reset()
        self.val_img_cm.reset()
        self.val_seg_f1.reset()
        self.val_map.reset()
        self.val_det_cm.reset()
        self.temp_matched_preds_for_cm = []
        print(f"    Metrics reset. (took {time.time() - current_op_time:.4f}s)")
        print(
            f"Epoch {self.current_epoch}: Validation epoch end finished. Total time: {time.time() - epoch_val_end_start_time:.2f}s\n"
        )

    def _log_segmentation_example(
        self, imgs, seg_head_outputs, gt_masks, ids, batch_idx
    ):
        with torch.no_grad():
            if len(seg_head_outputs) == 3:
                actual_protos_tensor_viz = seg_head_outputs[2]
            elif (
                len(seg_head_outputs) == 2
                and isinstance(seg_head_outputs[1], (list, tuple))
                and len(seg_head_outputs[1]) == 2
            ):
                _, (_, actual_protos_tensor_viz) = seg_head_outputs
            else:
                print(
                    f"Warning: Unexpected seg_head_outputs structure in _log_segmentation_example. Using dummy protos for viz."
                )
                actual_protos_tensor_viz = torch.zeros(
                    imgs.shape[0],
                    self.hparams.proto_ch,
                    self.hparams.img_size // 16,
                    self.hparams.img_size // 16,
                    device=imgs.device,
                )

            if not (
                isinstance(actual_protos_tensor_viz, torch.Tensor)
                and actual_protos_tensor_viz.ndim == 4
                and actual_protos_tensor_viz.shape[1] == self.hparams.proto_ch
            ):
                print(
                    f"Warning: Protos tensor for viz has unexpected shape or channel count: {actual_protos_tensor_viz.shape}. Skipping segmentation viz."
                )
                return

            seg_logits_projected_viz = self.seg_proto_projector(
                actual_protos_tensor_viz
            )
            seg_pred_resized_viz = F.interpolate(
                seg_logits_projected_viz,
                size=(self.hparams.img_size, self.hparams.img_size),
                mode="bilinear",
                align_corners=False,
            )

            pred_mask_viz_prob = torch.sigmoid(seg_pred_resized_viz[0, 0]).cpu().numpy()
            gt_mask_viz = gt_masks[0, 0].cpu().numpy()
            img_to_log_chw = imgs[0].cpu()
            img_min, img_max = img_to_log_chw.min(), img_to_log_chw.max()
            if img_min < 0 or img_max > 1:
                img_to_log_chw = (img_to_log_chw - img_min) / (img_max - img_min + 1e-6)

            wandb_masks_data = {
                "predictions": {
                    "mask_data": pred_mask_viz_prob,
                    "class_labels": self.seg_class_labels_wandb,
                },
                "ground_truth": {
                    "mask_data": gt_mask_viz,
                    "class_labels": self.seg_class_labels_wandb,
                },
            }
            log_id = ids[0] if ids and len(ids) > 0 else f"train_batch{batch_idx}"
            if self.logger and hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log(
                    {
                        f"segmentation_examples_train/img_{log_id}": wandb.Image(
                            img_to_log_chw, masks=wandb_masks_data
                        ),
                        "global_step": self.global_step,
                    }
                )

    def _log_detection_example(
        self, imgs, map_preds_batch, map_targets_batch, ids, batch_idx
    ):
        if not map_preds_batch or not map_targets_batch:
            return

        img_to_log_chw = imgs[0].cpu()
        img_min, img_max = img_to_log_chw.min(), img_to_log_chw.max()
        if img_min < 0 or img_max > 1:
            img_to_log_chw = (img_to_log_chw - img_min) / (img_max - img_min + 1e-6)

        pred_item = map_preds_batch[0]
        pred_boxes_viz = []
        if "scores" in pred_item and pred_item["scores"].numel() > 0:
            conf_mask = pred_item["scores"] > self.hparams.det_conf_thresh_viz
            viz_boxes = pred_item["boxes"][conf_mask]
            viz_scores = pred_item["scores"][conf_mask]
            viz_labels = pred_item["labels"][conf_mask]

            for i in range(viz_boxes.shape[0]):
                box = viz_boxes[i].cpu().tolist()
                pred_boxes_viz.append(
                    {
                        "position": {
                            "minX": box[0] / self.hparams.img_size,
                            "minY": box[1] / self.hparams.img_size,
                            "maxX": box[2] / self.hparams.img_size,
                            "maxY": box[3] / self.hparams.img_size,
                        },
                        "class_id": viz_labels[i].item(),
                        "box_caption": f"{self.det_class_names.get(viz_labels[i].item(), 'N/A')}: {viz_scores[i]:.2f}",
                        "scores": {"conf": viz_scores[i].item()},
                    }
                )

        gt_item = map_targets_batch[0]
        gt_boxes_viz = []
        if "boxes" in gt_item and gt_item["boxes"].numel() > 0:
            for i in range(gt_item["boxes"].shape[0]):
                box = gt_item["boxes"][i].cpu().tolist()
                gt_boxes_viz.append(
                    {
                        "position": {
                            "minX": box[0] / self.hparams.img_size,
                            "minY": box[1] / self.hparams.img_size,
                            "maxX": box[2] / self.hparams.img_size,
                            "maxY": box[3] / self.hparams.img_size,
                        },
                        "class_id": gt_item["labels"][i].item(),
                        "box_caption": f"{self.det_class_names.get(gt_item['labels'][i].item(), 'N/A')}",
                    }
                )

        wandb_box_data = {}
        if pred_boxes_viz:
            wandb_box_data["predictions"] = {
                "box_data": pred_boxes_viz,
                "class_labels": self.det_class_names,
            }
        if gt_boxes_viz:
            wandb_box_data["ground_truth"] = {
                "box_data": gt_boxes_viz,
                "class_labels": self.det_class_names,
            }

        log_id = ids[0] if ids and len(ids) > 0 else f"val_batch{batch_idx}"
        if wandb_box_data and self.logger and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log(
                {
                    f"detection_examples_val/img_{log_id}": wandb.Image(
                        img_to_log_chw, boxes=wandb_box_data
                    ),
                    "global_step": self.global_step,
                }
            )

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(
            trainable_params, lr=self.hparams.lr, weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs
            if self.trainer and self.trainer.max_epochs > 0
            else 50,
            eta_min=self.hparams.lr * 0.01,
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

    pl.seed_everything(42, workers=True)

    IMG_SIZE = 640
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 100
    NC_DET = 3
    NUM_IMG_CLASSES = 2
    PROTO_CH = 32
    IOU_MATCH_THRESH = 0.5
    LOSS_WEIGHT_SEG = 1.0
    LOSS_WEIGHT_BOX_IOU = 7.5
    LOSS_WEIGHT_DFL = 1.5
    LOSS_WEIGHT_CLS_DET = 0.5
    LOSS_WEIGHT_IMG_CLS = 1.0
    MASK_LOG_PERIOD = 200
    BOX_LOG_PERIOD = 1
    DET_CONF_THRESH_VIZ = 0.25
    MAP_MAX_DETECTIONS = 300  # Tuned mAP parameter

    wandb_logger = WandbLogger(
        project="BTXRD-MultiTask-AdvancedMetrics", log_model="all"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints_wandb_{wandb_logger.version if wandb_logger.version else 'local'}",
        filename="btrxd-multitask-{epoch:02d}-{val/loss_total:.2f}",
        save_top_k=3,
        monitor="val/loss_total",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val/loss_total", patience=20, mode="min", verbose=True
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
        precision="16-mixed",
        gradient_clip_val=10.0,
        log_every_n_steps=100,
        logger=wandb_logger,
        callbacks=[lr_monitor, model_checkpoint, early_stopping],
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
