# evaluate_model.py
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
from tqdm import tqdm
import numpy as np
from torch.functional import F

# Assuming running_main_v2.py and multitask_logging.py are in PYTHONPATH or same directory
from running_main_v2 import (
    MultiTaskLitModel,
    BTXRDDataModule,
    plot_confusion_matrix_to_wandb,
    MAX_VIZ_PER_CALL,
)
from multitask_logging import log_seg_examples, log_det_examples

# For image classification P/R/F1 if not using model's internal log_cls_metrics directly on accumulated
import torchmetrics.functional as TMF
import wandb


def evaluate(args):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.accelerator == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # --- Logger ---
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"eval_{args.wandb_run_name_suffix}_{args.checkpoint_path.split('/')[-1].replace('.ckpt', '')}",
        log_model=False,  # Don't log model artifact again
        offline=args.wandb_offline,
    )

    # --- DataModule ---
    data_module = BTXRDDataModule(
        root=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    data_module.setup("test")  # Ensure test_dataloader is prepared
    test_loader = data_module.test_dataloader()
    if not test_loader:
        print("Failed to load test data. Exiting.")
        return

    # --- Load Model from Checkpoint ---
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    try:
        # Load model and pass hparams that might have been added after checkpoint was saved
        # The model will use its own hparams for most things, but new ones need defaults
        # or to be passed if they affect metric/object initialization.
        # We will re-initialize map_detection_thresholds_list inside the model based on args.
        model = MultiTaskLitModel.load_from_checkpoint(
            args.checkpoint_path,
            map_strict=False,  # Allow loading if some keys are missing/unexpected in state_dict
            # cls_log_period=50, # Example: if this was a new hparam not in old checkpoints
            # map_max_detections will be loaded from hparams in checkpoint
        )
        print(
            f"Model loaded. Original hparams.map_max_detections: {model.hparams.get('map_max_detections', 'Not in hparams')}"
        )

        # Override mAP detection thresholds as per user request for this evaluation run
        model.map_detection_thresholds_list = [
            int(x) for x in args.map_thresholds.split(",")
        ]
        print(
            f"Using custom mAP detection thresholds for this evaluation: {model.map_detection_thresholds_list}"
        )

        # Re-initialize mAP metrics with new thresholds if they were already created with different ones
        # This is important if the model's __init__ used hparams.map_max_detections
        # which might differ from the [1,10,50] list desired for this specific evaluation.
        model.test_map_iou50_95 = pl.metrics.detection.MeanAveragePrecision(
            box_format="xyxy",
            class_metrics=True,
            iou_type="bbox",
            iou_thresholds=torch.linspace(0.5, 0.95, 10).tolist(),
            max_detection_thresholds=model.map_detection_thresholds_list,
        )
        model.test_map_iou50 = pl.metrics.detection.MeanAveragePrecision(
            box_format="xyxy",
            class_metrics=True,
            iou_type="bbox",
            iou_thresholds=[0.5],
            max_detection_thresholds=model.map_detection_thresholds_list,
        )

        model.to(device)
        model.eval()

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Prepare accumulators for image classification P/R/F1
    all_img_preds = []
    all_img_labels = []

    # Reset all test metrics on the model instance before starting
    model.test_img_acc.reset()
    model.test_img_cm.reset()
    model.test_seg_f1.reset()
    model.test_map_iou50_95.reset()
    model.test_map_iou50.reset()
    model.test_det_cm.reset()
    model.test_temp_matched_preds_for_cm = []

    print(f"Starting evaluation on {len(test_loader.dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            ids, imgs, det_boxes_gt, masks_gt, img_cls_gt = batch
            imgs = imgs.to(device)
            # det_boxes_gt, masks_gt, img_cls_gt are kept on CPU for metric updates if metrics are on CPU
            # Or move them to device if metrics are on device

            det_outputs, seg_outputs, img_cls_logits = model(imgs, mode="eval")

            # --- Image Classification Metrics ---
            img_preds_softmax = img_cls_logits.softmax(dim=1)
            img_preds_labels = img_preds_softmax.argmax(dim=1)

            model.test_img_acc.update(img_preds_labels.cpu(), img_cls_gt.cpu())
            model.test_img_cm.update(img_preds_labels.cpu(), img_cls_gt.cpu())
            all_img_preds.append(img_preds_labels.cpu())
            all_img_labels.append(img_cls_gt.cpu())

            # --- Segmentation Metrics & Logging ---
            # Calculate current_seg_logits (from model._multitask_loss or replicate logic)
            # Assuming model._multitask_loss sets self.seg_logits_for_logging correctly
            # For direct eval, we need to compute it:
            if len(seg_outputs) == 3:
                actual_protos_tensor_test = seg_outputs[2]
            elif (
                len(seg_outputs) == 2
                and isinstance(seg_outputs[1], (list, tuple))
                and len(seg_outputs[1]) == 2
            ):
                _, (_, actual_protos_tensor_test) = seg_outputs
            else:
                actual_protos_tensor_test = None

            current_seg_logits = None
            if (
                actual_protos_tensor_test is not None
                and isinstance(actual_protos_tensor_test, torch.Tensor)
                and actual_protos_tensor_test.ndim == 4
                and actual_protos_tensor_test.shape[1] == model.hparams.proto_ch
            ):
                seg_logits_projected_test = model.seg_proto_projector(
                    actual_protos_tensor_test
                )
                current_seg_logits = F.interpolate(
                    seg_logits_projected_test,
                    size=(model.hparams.img_size, model.hparams.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                model.test_seg_f1.update(
                    current_seg_logits.sigmoid().cpu(), masks_gt.int().cpu()
                )

            # --- Detection Metrics & Logging ---
            map_preds, map_targets, det_log_preds, det_log_gts = (
                model._prepare_det_outputs_for_metrics_and_logging(
                    det_outputs, det_boxes_gt, device, imgs.shape[0]
                )
            )

            if map_preds and map_targets:
                model.test_map_iou50_95.update(map_preds, map_targets)
                model.test_map_iou50.update(map_preds, map_targets)

            # For Detection CM: Call _multitask_loss to get matched pairs (or replicate matching)
            # This is a bit heavy if only for CM.
            # The model's test_step calls _multitask_loss. Here we are doing it manually.
            # We need `matched_preds_for_cm_this_batch` from `_multitask_loss`
            # For simplicity, we'll call it. Ensure it handles "eval" mode appropriately if it has mode-specific logic.
            # The `current_stage` argument was added to `_multitask_loss`.
            _, _, _, _, _, _, matched_preds_for_cm_this_batch = model._multitask_loss(
                det_outputs,
                seg_outputs,
                img_cls_logits,
                det_boxes_gt.to(device),
                masks_gt.to(device),
                img_cls_gt.to(device),
                # Ensure GTs are on device for loss
                current_stage="test",
            )
            for cm_preds, cm_gts in matched_preds_for_cm_this_batch:
                if cm_preds.numel() > 0 and cm_gts.numel() > 0:
                    model.test_temp_matched_preds_for_cm.extend(
                        list(zip(cm_preds.cpu().tolist(), cm_gts.cpu().tolist()))
                    )

            # --- Log Examples (first batch) ---
            if batch_idx == 0 and args.log_examples_wandb:
                print("Logging example images to W&B...")
                if current_seg_logits is not None:
                    log_seg_examples(
                        wandb_logger,
                        imgs.cpu(),
                        current_seg_logits.cpu(),
                        masks_gt=masks_gt.cpu(),
                        stage="test_eval",
                        step=0,
                        max_samples=MAX_VIZ_PER_CALL,
                    )
                if det_log_preds:
                    log_det_examples(
                        wandb_logger,
                        imgs.cpu(),
                        det_log_preds,
                        gts=det_log_gts,
                        class_id_to_name=model.det_class_names,
                        stage="test_eval",
                        step=0,
                        conf_th=model.hparams.det_conf_thresh_viz,
                        max_samples=MAX_VIZ_PER_CALL,
                    )
                wandb_logger.experiment.log(
                    {}, commit=True, step=0
                )  # Commit example logs

    # --- Compute and Log/Print Final Metrics ---
    print("\n--- Evaluation Results ---")
    final_metrics = {}

    # Image Classification
    img_acc_epoch = model.test_img_acc.compute().item()
    final_metrics["test_epoch/img_accuracy"] = img_acc_epoch
    print(f"Image Classification Accuracy: {img_acc_epoch:.4f}")

    all_img_preds_cat = torch.cat(all_img_preds)
    all_img_labels_cat = torch.cat(all_img_labels)

    num_img_classes = model.hparams.num_img_classes
    img_precision_macro = TMF.multiclass_precision(
        all_img_preds_cat,
        all_img_labels_cat,
        num_classes=num_img_classes,
        average="macro",
    ).item()
    img_recall_macro = TMF.multiclass_recall(
        all_img_preds_cat,
        all_img_labels_cat,
        num_classes=num_img_classes,
        average="macro",
    ).item()
    img_f1_macro = TMF.multiclass_f1_score(
        all_img_preds_cat,
        all_img_labels_cat,
        num_classes=num_img_classes,
        average="macro",
    ).item()

    final_metrics["test_epoch/img_precision_macro"] = img_precision_macro
    final_metrics["test_epoch/img_recall_macro"] = img_recall_macro
    final_metrics["test_epoch/img_f1_macro"] = img_f1_macro
    print(f"Image Classification Precision (Macro): {img_precision_macro:.4f}")
    print(f"Image Classification Recall (Macro): {img_recall_macro:.4f}")
    print(f"Image Classification F1-Score (Macro): {img_f1_macro:.4f}")

    img_cm_tensor_epoch = model.test_img_cm.compute()
    img_cm_fig = plot_confusion_matrix_to_wandb(
        img_cm_tensor_epoch.cpu(), model.img_class_names
    )
    if img_cm_fig:
        final_metrics["test_epoch/img_confusion_matrix"] = img_cm_fig
    print(
        "Image Classification Confusion Matrix (normalized):\n",
        img_cm_tensor_epoch.cpu().numpy(),
    )

    # Segmentation
    seg_f1_epoch = model.test_seg_f1.compute().item()
    final_metrics["test_epoch/seg_f1_score"] = seg_f1_epoch
    print(f"\nSegmentation F1-Score: {seg_f1_epoch:.4f}")

    # Detection mAP
    map_50_95_results = model.test_map_iou50_95.compute()
    print("\nDetection mAP@0.5:0.95 Results:")
    for k, v in map_50_95_results.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            val = v.item()
            final_metrics[f"test_epoch_map_iou50_95/{k}"] = val
            print(f"  {k}: {val:.4f}")
        elif k == "map_per_class" and isinstance(v, torch.Tensor) and v.ndim == 1:
            print(f"  {k}:")
            for i_cls, ap_val in enumerate(v):
                cls_name = model.det_class_names.get(i_cls, f"cls_{i_cls}")
                final_metrics[f"test_epoch_map_iou50_95_class/{cls_name}"] = (
                    ap_val.item()
                )
                print(f"    {cls_name}: {ap_val.item():.4f}")

    map_50_results = model.test_map_iou50.compute()
    print("\nDetection mAP@0.5 Results:")
    for k, v in map_50_results.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            val = v.item()
            final_metrics[f"test_epoch_map_iou50/{k}"] = val
            print(f"  {k}: {val:.4f}")
        elif k == "map_per_class" and isinstance(v, torch.Tensor) and v.ndim == 1:
            print(f"  {k}:")
            for i_cls, ap_val in enumerate(v):
                cls_name = model.det_class_names.get(i_cls, f"cls_{i_cls}")
                final_metrics[f"test_epoch_map_iou50_class/{cls_name}"] = ap_val.item()
                print(f"    {cls_name}: {ap_val.item():.4f}")

    # Detection Confusion Matrix
    if model.test_temp_matched_preds_for_cm:
        preds_cm_list_test, gts_cm_list_test = zip(
            *model.test_temp_matched_preds_for_cm
        )
        if preds_cm_list_test and gts_cm_list_test:
            try:
                preds_tensor_test = torch.tensor(
                    list(preds_cm_list_test), device="cpu", dtype=torch.long
                )
                gts_tensor_test = torch.tensor(
                    list(gts_cm_list_test), device="cpu", dtype=torch.long
                )
                model.test_det_cm.update(
                    preds_tensor_test, gts_tensor_test
                )  # metric should be on CPU for this
                det_cm_tensor_epoch = model.test_det_cm.compute()
                det_cm_fig = plot_confusion_matrix_to_wandb(
                    det_cm_tensor_epoch.cpu(), model.det_class_names
                )
                if det_cm_fig:
                    final_metrics["test_epoch/det_confusion_matrix"] = det_cm_fig
                print(
                    "Detection Confusion Matrix (normalized):\n",
                    det_cm_tensor_epoch.cpu().numpy(),
                )
            except Exception as e_det_cm:
                print(f"Error processing detection CM: {e_det_cm}")
    else:
        print("No matched predictions for detection CM.")

    # Log all metrics to W&B
    if wandb_logger.experiment:
        wandb_logger.log_metrics(final_metrics)  # Log all at once
        print(f"\nAll metrics logged to W&B run: {wandb_logger.experiment.url}")

    # Reset metrics on model (good practice, though script is ending)
    model.test_img_acc.reset()
    model.test_img_cm.reset()
    model.test_seg_f1.reset()
    model.test_map_iou50_95.reset()
    model.test_map_iou50.reset()
    model.test_det_cm.reset()

    if wandb.run:
        wandb.finish()
    print("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch Lightning MultiTask Model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="btxrd_ready",
        help="Root directory of the dataset",
    )
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for DataLoader"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--map_thresholds",
        type=str,
        default="1,10,50",
        help="Comma-separated list of max detection thresholds for mAP (e.g., '1,10,50')",
    )
    parser.add_argument(
        "--log_examples_wandb", action="store_true", help="Log example images to W&B"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="BTXRD-MultiTask-Evaluations",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name_suffix", type=str, default="", help="Suffix for W&B run name"
    )
    parser.add_argument(
        "--wandb_offline", action="store_true", help="Run W&B in offline mode"
    )

    args = parser.parse_args()
    evaluate(args)
