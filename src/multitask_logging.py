# -----------------------------------------------------------------------------
#  multitask_logging.py – unified helpers for W&B visualisation (v3)
# -----------------------------------------------------------------------------
#  Adds → log_cls_metrics          (image‑level classification head)
#        log_seg_examples (unchanged from v2 – real mask overlays)
#        log_det_examples (unchanged – bbox overlays)
#
#  Quick‑start (PyTorch‑Lightning):
#  ───────────────────────────────
#  from multitask_logging import log_cls_metrics, log_seg_examples, log_det_examples
#  … inside training/validation steps …
#      if self.global_step % self.hparams.log_every == 0:
#          # segmentation
#          log_seg_examples(self.logger, imgs, seg_logits, masks_gt=seg_masks,
#                           stage="train", step=self.global_step)
#          # detection
#          log_det_examples(self.logger, imgs, det_preds, gts=det_targets,
#                           class_id_to_name=self.id2label,
#                           stage="train", step=self.global_step)
#          # classification metrics (image‑level head)
#          log_cls_metrics(self.logger, img_logits, img_labels,
#                          class_id_to_name=self.id2label_img,
#                          stage="train", step=self.global_step)
#
#  What’s new in v3
#  ────────────────
#  ✓ **Classification head metrics:** logs accuracy, precision, recall, F1 (macro).
#  ✓ Keeps one key per stage (``cls_metrics_train`` / ``cls_metrics_val``), so
#    W&B plots a neat history curve automatically.
#  ✓ Zero extra deps – relies on torchmetrics.functional if present, otherwise
#    falls back to a simple torch implementation.
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
import wandb
import numpy as np
import torchvision.transforms.functional as TF
from typing import List, Sequence, Mapping, Any, Optional

# optional torchmetrics import -------------------------------------------------
_has_tm = False
try:
    from torchmetrics.functional.classification import (
        multiclass_precision,
        multiclass_recall,
        multiclass_f1_score,
    )

    _has_tm = True
except Exception:
    # we’ll compute metrics manually if torchmetrics is absent
    pass

# ----------------------------------------------------------------------------
#  Utilities
# ----------------------------------------------------------------------------


def _img_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in −1…1 *or* 0…1 → HWC uint8 (0‑255)."""
    if img.dim() != 3:
        raise ValueError("Expected CHW image tensor, got dim=%d" % img.dim())
    img = img.detach().cpu()
    if img.min() < 0:  # assume −1…1 range
        img = (img + 1) / 2
    img = (img.clamp(0, 1) * 255).byte()
    return img.permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------
#  💬 1. Segmentation examples (unchanged)
# ----------------------------------------------------------------------------


def log_seg_examples(
    logger: "pl.loggers.WandbLogger",
    images: torch.Tensor,  # B×C×H×W
    logits: torch.Tensor,  # B×1×H×W (pre‑sigmoid)
    *,
    masks_gt: Optional[torch.Tensor] = None,  # same shape as logits or None
    stage: str = "val",
    step: Optional[int] = None,  # explicitly control wandb step
    threshold: float = 0.5,
    max_samples: int = 4,
) -> None:
    if not hasattr(logger, "experiment"):
        raise RuntimeError("Expecting a Lightning WandbLogger instance.")
    run = logger.experiment  # type: ignore[attr-defined]

    b = min(images.size(0), max_samples)
    preds_bin = (logits.sigmoid() > threshold).detach().cpu()

    wb_imgs: List[wandb.Image] = []
    for i in range(b):
        img_np = _img_to_uint8(images[i])
        masks: dict[str, dict[str, Any]] = {}

        pred_mask = preds_bin[i, 0].byte().numpy()
        masks["prediction"] = {"mask_data": pred_mask, "class_labels": {1: "pred"}}

        if masks_gt is not None:
            gt_mask = masks_gt[i, 0].detach().cpu().byte().numpy()
            masks["ground_truth"] = {"mask_data": gt_mask, "class_labels": {1: "gt"}}

        wb_imgs.append(wandb.Image(img_np, masks=masks))

    run.log({f"seg_examples_{stage}": wb_imgs}, step=step, commit=False)


# ----------------------------------------------------------------------------
#  💬 2. Detection examples (unchanged)
# ----------------------------------------------------------------------------

_Box = Mapping[str, Any]


def _tensor_boxes_to_wb(
    boxes_xyxy: torch.Tensor,  # N×4
    scores: torch.Tensor,  # N
    labels: torch.Tensor,  # N
    class_id_to_name: Mapping[int, str],
    caption_prefix: str = "pred",
) -> List[_Box]:
    boxes_xyxy = boxes_xyxy.detach().cpu()
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    out: List[_Box] = []
    for (x1, y1, x2, y2), s, l in zip(
        boxes_xyxy.tolist(), scores.tolist(), labels.tolist()
    ):
        out.append(
            {
                "position": {
                    "minX": x1,
                    "minY": y1,
                    "maxX": x2,
                    "maxY": y2,
                },
                "class_id": int(l),
                "scores": {"conf": s},
                "box_caption": f"{caption_prefix} {class_id_to_name.get(int(l), l)} {s:.2f}",
            }
        )
    return out


def log_det_examples(
    logger: "pl.loggers.WandbLogger",
    images: torch.Tensor,  # B×C×H×W
    preds: Sequence[torch.Tensor],  # len(B) list, each (N,6) xyxy+score+cls
    *,
    gts: Optional[Sequence[torch.Tensor]] = None,  # len(B), each (M,5) xyxy+cls
    class_id_to_name: Mapping[int, str],
    stage: str = "val",
    step: Optional[int] = None,
    conf_th: float = 0.25,
    max_samples: int = 4,
    max_boxes: int = 100,
) -> None:
    if not hasattr(logger, "experiment"):
        raise RuntimeError("Expecting a Lightning WandbLogger instance.")
    run = logger.experiment  # type: ignore[attr-defined]

    wb_imgs: List[wandb.Image] = []
    b = min(len(preds), max_samples)

    for i in range(b):
        img_np = _img_to_uint8(images[i])
        pred = preds[i]
        if pred.ndim != 2 or pred.shape[1] != 6:
            raise ValueError("Each preds[i] must be (N,6): x1,y1,x2,y2,conf,cls")

        keep = pred[:, 4] > conf_th
        pred = pred[keep]
        if pred.shape[0] > max_boxes:
            pred = pred[:max_boxes]

        boxes_dict: dict[str, _Box] = {
            "box_data": _tensor_boxes_to_wb(
                pred[:, :4], pred[:, 4], pred[:, 5], class_id_to_name
            ),
            "class_labels": class_id_to_name,
        }

        overlay: dict[str, Any] = {"pred": boxes_dict}

        if gts is not None and i < len(gts):
            gt = gts[i]
            if gt.ndim == 2 and gt.shape[1] == 5:
                overlay["gt"] = {
                    "box_data": _tensor_boxes_to_wb(
                        gt[:, :4], torch.ones(len(gt)), gt[:, 4], class_id_to_name, "gt"
                    ),
                    "class_labels": class_id_to_name,
                }

        wb_imgs.append(wandb.Image(img_np, boxes=overlay))

    run.log({f"det_examples_{stage}": wb_imgs}, step=step, commit=False)


# ----------------------------------------------------------------------------
#  💬 3. Classification‑head metrics (NEW)
# ----------------------------------------------------------------------------


def _simple_macro_prec_recall_f1(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int
):
    """Fallback in case torchmetrics is unavailable."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds.tolist(), target.tolist()):
        cm[t, p] += 1

    TP = cm.diag()
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP

    precision = torch.where(TP + FP == 0, torch.tensor(0.0), TP.float() / (TP + FP))
    recall = torch.where(TP + FN == 0, torch.tensor(0.0), TP.float() / (TP + FN))
    f1 = torch.where(
        precision + recall == 0,
        torch.tensor(0.0),
        2 * precision * recall / (precision + recall),
    )

    return precision.mean().item(), recall.mean().item(), f1.mean().item()


def log_cls_metrics(
    logger: "pl.loggers.WandbLogger",
    logits: torch.Tensor,  # B×C (pre‑softmax)
    labels: torch.Tensor,  # B
    *,
    class_id_to_name: Mapping[int, str] | None = None,  # optional for future use
    stage: str = "val",
    step: Optional[int] = None,
) -> None:
    """Compute & log accuracy, precision, recall, F1 (macro‑average)."""

    if not hasattr(logger, "experiment"):
        raise RuntimeError("Expecting a Lightning WandbLogger instance.")
    run = logger.experiment  # type: ignore[attr-defined]

    preds = logits.softmax(dim=1).argmax(dim=1)
    labels = labels.detach().cpu()
    preds = preds.detach().cpu()

    num_classes = logits.shape[1]
    acc = (preds == labels).float().mean().item()

    if _has_tm:
        precision = multiclass_precision(
            preds, labels, num_classes=num_classes, average="macro"
        ).item()
        recall = multiclass_recall(
            preds, labels, num_classes=num_classes, average="macro"
        ).item()
        f1 = multiclass_f1_score(
            preds, labels, num_classes=num_classes, average="macro"
        ).item()
    else:
        precision, recall, f1 = _simple_macro_prec_recall_f1(preds, labels, num_classes)

    run.log(
        {
            f"cls_{stage}/accuracy": acc,
            f"cls_{stage}/precision": precision,
            f"cls_{stage}/recall": recall,
            f"cls_{stage}/f1": f1,
        },
        step=step,
        commit=False,
    )
