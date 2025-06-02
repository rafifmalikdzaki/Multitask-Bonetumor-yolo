# -----------------------------------------------------------------------------
#  multitask_logging.py – unified helpers for W&B visualisation (v4)
# -----------------------------------------------------------------------------
#  Adds → log_cls_metrics        (image‑level classification head)
#         log_seg_examples      (mask overlays - now with random sampling)
#         log_det_examples      (bbox overlays - now with random sampling)
#
#  Quick‑start (PyTorch‑Lightning):
#  ───────────────────────────────
#  from multitask_logging import log_cls_metrics, log_seg_examples, log_det_examples
#  … inside training/validation steps …
#      if self.global_step % self.hparams.log_every == 0:
#          # segmentation
#          log_seg_examples(self.logger, imgs, seg_logits, masks_gt=seg_masks,
#                           stage="train", step=self.global_step, max_samples=4)
#          # detection
#          log_det_examples(self.logger, imgs, det_preds, gts=det_targets,
#                           class_id_to_name=self.id2label,
#                           stage="train", step=self.global_step, max_samples=4)
#          # classification metrics (image‑level head)
#          log_cls_metrics(self.logger, img_logits, img_labels,
#                          class_id_to_name=self.id2label_img,
#                          stage="train", step=self.global_step)
#
#  What’s new in v4
#  ────────────────
#  ✓ **Random Sampling for Examples:** `log_seg_examples` and `log_det_examples`
#    now randomly sample `max_samples` from the batch if batch_size > max_samples,
#    providing more diverse visualizations over time.
#  ✓ Classification head metrics: logs accuracy, precision, recall, F1 (macro).
#  ✓ Keeps one key per stage (``cls_metrics_train`` / ``cls_metrics_val``), so
#    W&B plots a neat history curve automatically.
#  ✓ Zero extra deps – relies on torchmetrics.functional if present, otherwise
#    falls back to a simple torch implementation.
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
import wandb
import numpy as np
# import torchvision.transforms.functional as TF # Not strictly needed for these functions
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
except ImportError: # More specific exception
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
    if img.min() < -1e-5:  # Check for negative values (allow for small float inaccuracies)
        img = (img + 1) / 2
    img = (img.clamp(0, 1) * 255).byte()
    return img.permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------
#  💬 1. Segmentation examples (NOW WITH RANDOM SAMPLING)
# ----------------------------------------------------------------------------


def log_seg_examples(
    logger: "pl.loggers.WandbLogger", # type: ignore
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
    run = logger.experiment # type: ignore[attr-defined]

    actual_batch_size = images.size(0)
    num_to_log = min(actual_batch_size, max_samples)

    if actual_batch_size == 0:
        return # Nothing to log

    # <<< MODIFIED >>> Select indices for logging
    if actual_batch_size > max_samples:
        # Randomly select 'max_samples' distinct indices
        selected_indices = torch.randperm(actual_batch_size)[:num_to_log].tolist()
    else:
        # Log all available samples
        selected_indices = list(range(actual_batch_size))

    preds_bin = (logits.sigmoid() > threshold).detach().cpu()

    wb_imgs: List[wandb.Image] = []
    for original_idx in selected_indices: # Iterate over selected indices
        img_np = _img_to_uint8(images[original_idx])
        masks_payload: dict[str, dict[str, Any]] = {} # Renamed for clarity

        pred_mask = preds_bin[original_idx, 0].byte().numpy()
        masks_payload["prediction"] = {"mask_data": pred_mask, "class_labels": {1: "pred"}}

        if masks_gt is not None:
            # Ensure masks_gt has data for the selected index
            if original_idx < masks_gt.size(0):
                gt_mask = masks_gt[original_idx, 0].detach().cpu().byte().numpy()
                masks_payload["ground_truth"] = {"mask_data": gt_mask, "class_labels": {1: "gt"}}
            else:
                # This case should ideally not happen if masks_gt corresponds to images
                print(f"Warning: masks_gt for index {original_idx} not available in log_seg_examples.")


        wb_imgs.append(wandb.Image(img_np, masks=masks_payload))

    if wb_imgs: # Only log if there are images
        run.log({f"seg_examples_{stage}": wb_imgs}, step=step, commit=False)


# ----------------------------------------------------------------------------
#  💬 2. Detection examples (NOW WITH RANDOM SAMPLING)
# ----------------------------------------------------------------------------

_Box = Mapping[str, Any] # Type alias for a box dictionary


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
    for (x1, y1, x2, y2), s, l_tensor in zip( # Renamed l to l_tensor
        boxes_xyxy.tolist(), scores.tolist(), labels.tolist()
    ):
        l = int(l_tensor) # Ensure label is int for dictionary key
        out.append(
            {
                "position": {
                    "minX": x1, "minY": y1,
                    "maxX": x2, "maxY": y2,
                },
                "class_id": l,
                "domain": "pixel", # Added domain for W&B UI
                "scores": {"conf": float(s)}, # Ensure score is float
                "box_caption": f"{caption_prefix} {class_id_to_name.get(l, str(l))} {s:.2f}",
            }
        )
    return out


def log_det_examples(
    logger: "pl.loggers.WandbLogger", # type: ignore
    images: torch.Tensor,  # B×C×H×W
    preds: Sequence[torch.Tensor],  # len(B) list, each (N,6) xyxy+score+cls
    *,
    gts: Optional[Sequence[torch.Tensor]] = None,  # len(B), each (M,5) xyxy+cls
    class_id_to_name: Mapping[int, str],
    stage: str = "val",
    step: Optional[int] = None,
    conf_th: float = 0.25,
    max_samples: int = 4,
    max_boxes: int = 100, # Max boxes to show per image
) -> None:
    if not hasattr(logger, "experiment"):
        raise RuntimeError("Expecting a Lightning WandbLogger instance.")
    run = logger.experiment # type: ignore[attr-defined]

    actual_batch_size = images.size(0)
    if actual_batch_size == 0 or len(preds) == 0:
        return # Nothing to log

    # Ensure preds and images have compatible batch sizes for selection
    if len(preds) != actual_batch_size:
        print(f"Warning: Mismatch in image batch size ({actual_batch_size}) and preds length ({len(preds)}) in log_det_examples. Skipping.")
        return
    if gts is not None and len(gts) != actual_batch_size:
        print(f"Warning: Mismatch in image batch size ({actual_batch_size}) and gts length ({len(gts)}) in log_det_examples. GTs might be skipped for some images.")
        # We can proceed but be mindful that gts might not align for all selected_indices

    num_to_log = min(actual_batch_size, max_samples)

    # <<< MODIFIED >>> Select indices for logging
    if actual_batch_size > max_samples:
        selected_indices = torch.randperm(actual_batch_size)[:num_to_log].tolist()
    else:
        selected_indices = list(range(actual_batch_size))

    wb_imgs: List[wandb.Image] = []

    for original_idx in selected_indices: # Iterate over selected indices
        img_np = _img_to_uint8(images[original_idx])
        pred_item = preds[original_idx] # Get the predictions for the selected image

        if not (isinstance(pred_item, torch.Tensor) and pred_item.ndim == 2 and pred_item.shape[1] == 6):
            print(f"Warning: preds[{original_idx}] has unexpected shape or type: {type(pred_item)}, shape {getattr(pred_item, 'shape', 'N/A')}. Skipping this item for det_examples.")
            continue # Skip this problematic prediction item

        keep = pred_item[:, 4] > conf_th
        pred_item_filtered = pred_item[keep]
        if pred_item_filtered.shape[0] > max_boxes:
            # Optionally sort by score before taking top K if desired
            # _, sort_idx = pred_item_filtered[:, 4].sort(descending=True)
            # pred_item_filtered = pred_item_filtered[sort_idx[:max_boxes]]
            pred_item_filtered = pred_item_filtered[:max_boxes]


        boxes_payload: dict[str, Any] = { # Renamed for clarity
            "pred": { # Predictions key
                "box_data": _tensor_boxes_to_wb(
                    pred_item_filtered[:, :4], pred_item_filtered[:, 4], pred_item_filtered[:, 5], class_id_to_name
                ),
                "class_labels": class_id_to_name,
            }
        }

        if gts is not None and original_idx < len(gts):
            gt_item = gts[original_idx] # Get GT for the selected image
            if isinstance(gt_item, torch.Tensor) and gt_item.ndim == 2 and gt_item.shape[1] == 5:
                if gt_item.shape[0] > max_boxes: # Also cap GT boxes for visualization clarity
                    gt_item = gt_item[:max_boxes]
                boxes_payload["gt"] = { # Ground truth key
                    "box_data": _tensor_boxes_to_wb(
                        gt_item[:, :4], torch.ones(len(gt_item), device=gt_item.device), gt_item[:, 4], class_id_to_name, "gt"
                    ),
                    "class_labels": class_id_to_name,
                }
            elif gt_item.numel() > 0: # If gt_item is not empty but has wrong shape
                 print(f"Warning: gts[{original_idx}] has unexpected shape or type: {type(gt_item)}, shape {getattr(gt_item, 'shape', 'N/A')}. Skipping GT for this item.")


        wb_imgs.append(wandb.Image(img_np, boxes=boxes_payload))

    if wb_imgs: # Only log if there are images
        run.log({f"det_examples_{stage}": wb_imgs}, step=step, commit=False)


# ----------------------------------------------------------------------------
#  💬 3. Classification‑head metrics (Unchanged from your v3)
# ----------------------------------------------------------------------------


def _simple_macro_prec_recall_f1(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int
):
    """Fallback in case torchmetrics is unavailable."""
    # Ensure preds and target are on CPU for list conversion and cm update
    preds = preds.cpu()
    target = target.cpu()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device='cpu') # Operate on CPU

    for p_val, t_val in zip(preds.tolist(), target.tolist()): # Use p_val, t_val
        if 0 <= t_val < num_classes and 0 <= p_val < num_classes:
             cm[t_val, p_val] += 1
        else:
            print(f"Warning: Out of bounds label/pred in _simple_macro_prec_recall_f1. Target: {t_val}, Pred: {p_val}, Num Classes: {num_classes}")


    TP = cm.diag()
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP

    # Add epsilon to avoid division by zero, ensure results are float
    eps = 1e-6
    precision = TP.float() / (TP + FP + eps)
    recall = TP.float() / (TP + FN + eps)
    f1 = torch.where(
        precision + recall == 0,
        torch.tensor(0.0, device='cpu'), # Ensure tensor is on same device
        2 * precision * recall / (precision + recall + eps),
    )
    # Handle cases where a class might have no TPs, FPs, or FNs leading to NaNs if not careful
    # The .mean() will ignore NaNs if they occur due to no instances of a class in TP+FP or TP+FN
    return precision.nan_to_num(0.0).mean().item(), recall.nan_to_num(0.0).mean().item(), f1.nan_to_num(0.0).mean().item()


def log_cls_metrics(
    logger: "pl.loggers.WandbLogger", # type: ignore
    logits: torch.Tensor,  # B×C (pre‑softmax)
    labels: torch.Tensor,  # B
    *,
    class_id_to_name: Mapping[int, str] | None = None,  # optional for future use
    log_prefix: str = "cls", # <<< MODIFIED from 'stage' to 'log_prefix' for more flexibility
    step: Optional[int] = None,
) -> None:
    """Compute & log accuracy, precision, recall, F1 (macro‑average).
       log_prefix can be e.g. "train_step/cls", "val_epoch/cls_img"
    """

    if not hasattr(logger, "experiment"):
        raise RuntimeError("Expecting a Lightning WandbLogger instance.")
    run = logger.experiment # type: ignore[attr-defined]

    # Ensure logits and labels are on the same device for softmax and comparison
    # Typically, model outputs (logits) and labels are already on the correct device from the LightningModule
    preds_softmax = logits.softmax(dim=1)
    preds = preds_softmax.argmax(dim=1)

    # For metrics calculation, it's often safer to move to CPU, especially if using the simple fallback
    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()

    num_classes = logits.shape[1]
    if num_classes == 0: # Should not happen with valid logits
        print("Warning: num_classes is 0 in log_cls_metrics. Skipping.")
        return

    acc = (preds_cpu == labels_cpu).float().mean().item()

    if _has_tm:
        # torchmetrics functions expect preds and target on the same device.
        # If logits/labels were on GPU, preds/labels are also on GPU.
        # If _has_tm, use original device tensors for potentially better performance.
        precision = multiclass_precision(
            preds, labels, num_classes=num_classes, average="macro", validate_args=True
        ).item()
        recall = multiclass_recall(
            preds, labels, num_classes=num_classes, average="macro", validate_args=True
        ).item()
        f1 = multiclass_f1_score(
            preds, labels, num_classes=num_classes, average="macro", validate_args=True
        ).item()
    else:
        # Use CPU tensors for the simple fallback
        precision, recall, f1 = _simple_macro_prec_recall_f1(preds_cpu, labels_cpu, num_classes)

    # Use the log_prefix to structure the metric names
    log_data = {
        f"{log_prefix}/accuracy": acc,
        f"{log_prefix}/precision_macro": precision, # Clarified macro
        f"{log_prefix}/recall_macro": recall,    # Clarified macro
        f"{log_prefix}/f1_macro": f1,            # Clarified macro
    }
    # Add global_step if not provided by PTL's self.log()
    # However, PTL's self.log() usually handles the step.
    # This function is designed to be called with logger.experiment.log directly.
    if step is not None:
        log_data["global_step"] = step # Ensure global_step is part of the dict if provided

    run.log(log_data, step=step, commit=False) # commit=False as PTL usually commits at end of step

