import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import wandb
from dataset_btxrd_new import BTXRD, collate_fn
# from model import ConvNeXtBiFPNYOLO, load_pretrained_heads
from main_model import ConvNeXtBiFPNYOLO, load_pretrained_heads
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, \
                                         MulticlassRecall, MulticlassF1Score

# ────────────────────────────────────────────────── Loss helpers

def _bbox_iou(box1, box2, eps: float = 1e-7):
    """Compute IoU between two sets of boxes (x1,y1,x2,y2)."""
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1 + area2 - inter + eps)


def batch_bbox_iou(boxes1, boxes2, eps: float = 1e-7):
    """
    Compute IoU between two batches of boxes.
    boxes1: [N, 4] - First batch of boxes (x1,y1,x2,y2)
    boxes2: [M, 4] - Second batch of boxes (x1,y1,x2,y2)
    Returns: [N, M] - IoU matrix
    """
    # Get box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0:1], boxes2[:, 1:2], boxes2[:, 2:3], boxes2[:, 3:4]

    # Calculate intersection area
    inter_x1 = torch.max(b1_x1, b2_x1.T)
    inter_y1 = torch.max(b1_y1, b2_y1.T)
    inter_x2 = torch.min(b1_x2, b2_x2.T)
    inter_y2 = torch.min(b1_y2, b2_y2.T)
    w = (inter_x2 - inter_x1).clamp(min=0)
    h = (inter_y2 - inter_y1).clamp(min=0)
    
    # Calculate intersection area and IoU
    inter_area = w * h
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    iou = inter_area / (union + eps)
    
    return iou


# ────────────────────────────────────────────────── Lightning Module
class MultiTaskLitModel(pl.LightningModule):
    def __init__(self, img_size: int = 640, lr: float = 1e-4, mask_log_period: int = 50):
        super().__init__()
        self.save_hyperparameters()
        self.num_img_classes = 2
        self.net = ConvNeXtBiFPNYOLO(nc_det=3, nc_img=2)
        load_pretrained_heads(self.net)
        self.ce = torch.nn.CrossEntropyLoss()
        self.img_size = img_size
        self.lr = lr
        self.mask_log_period = mask_log_period  # Log masks every N batches
          # ── metrics (stateful, automatically moved to device & reset each epoch) ──
        self.train_acc   = MulticlassAccuracy(self.num_img_classes, average='micro')
        self.train_prec  = MulticlassPrecision(self.num_img_classes, average='micro')
        self.train_rec   = MulticlassRecall(self.num_img_classes, average='micro')
        self.train_f1    = MulticlassF1Score(self.num_img_classes, average='micro')

        self.val_acc     = MulticlassAccuracy(self.num_img_classes, average='micro')
        self.val_prec    = MulticlassPrecision(self.num_img_classes, average='micro')
        self.val_rec     = MulticlassRecall(self.num_img_classes, average='micro')
        self.val_f1      = MulticlassF1Score(self.num_img_classes, average='micro')

    # ----------------------- custom multitask loss (simplified)
    def _multitask_loss(self, det_out, seg_out, targets, masks):
        # ─── SEGMENTATION LOSS ────────────────────────────────────
        # seg_out may be (out, coeffs, protos) or (coeffs, protos)
        if isinstance(seg_out, (list, tuple)):
            if len(seg_out) == 3:
                _, coeffs, protos = seg_out
            elif len(seg_out) == 2:
                coeffs, protos = seg_out
            else:
                raise ValueError("Unexpected seg_out length")
        else:
            raise TypeError("seg_out must be list/tuple")

                # unwrap arbitrary nesting of 1‑element lists/tuples → tensor
        def _to_tensor(t):
            while isinstance(t, (list, tuple)):
                if len(t) == 0:
                    raise ValueError("Empty list/tuple where tensor expected")
                t = t[0]
            return t

        coeffs = _to_tensor(coeffs)
        protos = _to_tensor(protos)

        print("coeffs shape:", coeffs.shape)
        print("protos shape:", protos.shape)
        
        # Reshape coefficients to match the prototype channels
        batch_size, num_queries, coeff_dim = coeffs.shape
        proto_channels = protos.shape[1]
        proto_height, proto_width = protos.shape[2], protos.shape[3]
        
        # Reshape coefficients from [B, Q, 8400] to [B, Q, 67]
        # The 8400 dimension likely comes from flattened spatial dimensions (e.g., 80x80 + offsets)
        if coeff_dim != proto_channels:
            print(f"Reshaping coefficients from {coeff_dim} to {proto_channels} channels")
            
            # Try to determine if coefficients can be reshaped as spatial feature maps
            # For example, if coeff_dim is 8400, it might be 80x105 or other spatial layout
            found_spatial_arrangement = False
            
            # Common spatial dimensions to try (for prototype feature maps)
            spatial_candidates = [
                (80, 105),  # 8400 = 80×105
                (90, 93),   # 8370 ≈ 8400
                (84, 100),  # 8400 = 84×100
                (70, 120),  # 8400 = 70×120
            ]
            
            # Try different spatial arrangements
            for height, width in spatial_candidates:
                if abs(height * width - coeff_dim) < 100:  # Allow some tolerance
                    try:
                        # Try to reshape as [B, Q, H, W]
                        spatial_coeffs = coeffs.reshape(batch_size, num_queries, height, width)
                        
                        # Now use adaptive pooling to reduce to prototype dimensions
                        # Reshape to [B*Q, 1, H, W] for 2D pooling
                        flat_spatial = spatial_coeffs.reshape(-1, 1, height, width)
                        
                        # Pool to match prototype channels - creates [B*Q, 1, proto_channels, 1]
                        pooled = F.adaptive_avg_pool2d(flat_spatial, (proto_channels, 1))
                        
                        # Reshape back to [B, Q, proto_channels]
                        coeffs_reshaped = pooled.reshape(batch_size, num_queries, proto_channels)
                        
                        print(f"Successfully reshaped using spatial dimensions {height}x{width}")
                        found_spatial_arrangement = True
                        break
                    except (RuntimeError, ValueError) as e:
                        print(f"Failed to reshape with {height}x{width}: {e}")
                        continue
            
            # Fall back to previous methods if spatial arrangement wasn't found
            if not found_spatial_arrangement:
                # Check if 8400 is approximately equal to proto_height*proto_width*proto_channels
                if abs(coeff_dim - (proto_height * proto_width * proto_channels)) < proto_channels:
                    # This suggests coefficients are flattened [C, H, W] format
                    print(f"Coefficients appear to be in flattened [C, H, W] format")
                    # Reshape to [B, Q, C, H, W]
                    reshaped = coeffs.reshape(batch_size, num_queries, proto_channels, proto_height, proto_width)
                    # Average over spatial dimensions to get [B, Q, C]
                    coeffs_reshaped = reshaped.mean(dim=(3, 4))
                elif coeff_dim > proto_channels:
                    # Alternative: If coefficients have extra dimensions, use slicing
                    print(f"Using slicing to reduce coefficient dimensions")
                    
                    # Approach 1: Use adaptive pooling
                    # Reshape to [B*Q, 1, 1, coeff_dim]
                    flat_coeffs = coeffs.reshape(-1, 1, 1, coeff_dim)
                    # Use adaptive pooling to reduce to proto_channels
                    pooled = F.adaptive_avg_pool2d(flat_coeffs, (1, proto_channels))
                    # Reshape back to [B, Q, proto_channels]
                    coeffs_reshaped = pooled.reshape(batch_size, num_queries, proto_channels)
                else:
                    # If all else fails, just use the first proto_channels values
                    print(f"Using first {proto_channels} coefficient values")
                    coeffs_reshaped = coeffs[:, :, :proto_channels]
        else:
            coeffs_reshaped = coeffs
            
        seg_pred = torch.einsum("bqc,bchw->bqhw", coeffs_reshaped, protos)
        
        # Print shapes for debugging
        print(f"seg_pred shape before resize: {seg_pred.shape}")
        print(f"masks shape: {masks.shape}")
        
        # Resize segmentation predictions to match mask size (upsampling)
        # Use bilinear interpolation for smoother results
        # masks shape is [B, 1, H, W] (e.g. [4, 1, 640, 640])
        # seg_pred shape is [B, Q, h, w] (e.g. [4, 39, 80, 80])
        
        # Need to handle the case where masks have a single channel but predictions have multiple
        if masks.shape[1] == 1 and seg_pred.shape[1] > 1:
            # Option 1: If each prediction channel needs to match the same mask
            # Duplicate mask across channels to match prediction shape
            masks_expanded = masks.expand(-1, seg_pred.shape[1], -1, -1)
            
            # Upsample predictions to match mask spatial dimensions
            seg_pred_resized = F.interpolate(
                seg_pred, 
                size=(masks.shape[2], masks.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )

            loss_seg = F.binary_cross_entropy_with_logits(seg_pred_resized, masks_expanded)
        else:
            # Option 2: If we need to compute loss between each prediction channel and mask channel
            # Upsample predictions to match mask spatial dimensions
            seg_pred_resized = F.interpolate(
                seg_pred, 
                size=(masks.shape[2], masks.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )

            loss_seg = F.binary_cross_entropy_with_logits(seg_pred_resized, masks)

        print(f"seg_pred shape after resize: {seg_pred_resized.shape}")

        # ─── DETECTION LOSS (IoU‑only placeholder) ────────────────
        # Handle nested detection output structure, similar to seg_out
        print(f"det_out type: {type(det_out)}")
        if isinstance(det_out, (list, tuple)):
            # Unwrap nested list/tuple structure
            det_tensors = []
            for i, p in enumerate(det_out):
                # Use the _to_tensor helper to extract tensors from nested structures
                tensor = _to_tensor(p)
                print(f"Output tensor {i} shape: {tensor.shape if isinstance(tensor, torch.Tensor) else 'Not a tensor'}")
                
                # YOLOv8-style outputs are multi-scale feature maps with shape [batch, channels, height, width]
                # Feature maps from different scales (e.g., 80x80, 40x40, 20x20)
                if isinstance(tensor, torch.Tensor):
                    # Check if this is a detection tensor (vs. prototype tensor)
                    # Detection tensors are typically 4D tensors [batch, channels, height, width]
                    # where channels include coordinates and objectness scores
                    if len(tensor.shape) == 4:
                        print(f"Found detection feature map with shape: {tensor.shape}")
                        det_tensors.append(tensor)
                    else:
                        print(f"Skipping non-detection tensor with shape: {tensor.shape}")
            
            # Only proceed if we have valid detection tensors
            if det_tensors:
                # Process all detection feature maps 
                # (YOLOv8-style multi-scale detection features)
                print(f"Processing {len(det_tensors)} detection feature maps")
                all_predictions = []
                
                # Process each scale's feature map
                for det_idx, det_feat in enumerate(det_tensors):
                    # Extract shape information
                    batch_size, channels, height, width = det_feat.shape
                    print(f"Processing feature map {det_idx}: {det_feat.shape}")
                    
                    # Determine stride based on feature map size relative to input image
                    # If input is 640x640 and feature map is 80x80, stride is 8
                    stride = self.img_size / height
                    print(f"Feature map stride: {stride}")
                    
                    # Create grid coordinates
                    grid_y, grid_x = torch.meshgrid([torch.arange(height, device=det_feat.device),
                                                     torch.arange(width, device=det_feat.device)], 
                                                     indexing='ij')
                    
                    # Reshape feature map: [B, C, H, W] -> [B, H, W, C]
                    det_feat = det_feat.permute(0, 2, 3, 1).contiguous()
                    
                    # Split the channels appropriately for box regression and classification
                    # Assuming first 4 channels are for bounding box (x, y, w, h)
                    box_regression = det_feat[..., :4]
                    
                    # Create grid of shape [H, W, 2] for all positions in feature map
                    grid = torch.stack([grid_x, grid_y], dim=-1).float()
                    
                    # Process box coordinates - convert from grid-relative to absolute coordinates
                    # Typically: center_x = (sigmoid(x) + grid_x) * stride
                    #           center_y = (sigmoid(y) + grid_y) * stride
                    #           width = exp(w) * anchor_w * stride
                    #           height = exp(h) * anchor_h * stride
                    # Simplified version (without anchors):
                    x_center = (torch.sigmoid(box_regression[..., 0]) + grid[..., 0]) * stride 
                    y_center = (torch.sigmoid(box_regression[..., 1]) + grid[..., 1]) * stride
                    w = torch.exp(box_regression[..., 2]) * stride # Can use anchor multiplication if needed
                    h = torch.exp(box_regression[..., 3]) * stride # Can use anchor multiplication if needed
                    
                    # Convert to format expected by IoU calculation
                    pred_boxes = torch.stack([
                        x_center / self.img_size,  # Normalize to [0, 1]
                        y_center / self.img_size,
                        w / self.img_size,
                        h / self.img_size
                    ], dim=-1)
                    
                    # Objectness scores (typically channel 4)
                    if det_feat.shape[-1] > 4:  # If we have objectness scores
                        obj_scores = torch.sigmoid(det_feat[..., 4:5])
                        
                        # Class scores (if present, typically channels 5+)
                        if det_feat.shape[-1] > 5:
                            class_scores = torch.sigmoid(det_feat[..., 5:])
                            # Combine with objectness for class-specific confidence
                            class_scores = obj_scores * class_scores
                        else:
                            class_scores = obj_scores
                    else:
                        # If no objectness, use a dummy value of 1.0
                        obj_scores = torch.ones(batch_size, height, width, 1, device=det_feat.device)
                        class_scores = obj_scores
                    
                    # Combine box coordinates and confidence scores
                    # Format: [batch, height*width, 4+num_classes]
                    predictions = torch.cat([pred_boxes, class_scores], dim=-1)
                    
                    # Reshape to [batch, height*width, 4+num_classes]
                    predictions = predictions.reshape(batch_size, height * width, -1)
                    
                    all_predictions.append(predictions)
                
                # Concatenate predictions from all feature maps
                # Result shape: [batch, sum(h*w), 4+num_classes]
                preds = torch.cat(all_predictions, dim=1)
                print(f"Combined predictions shape: {preds.shape}")
            else:
                print("No valid detection tensors found, skipping detection loss")
                return loss_seg
        else:
            # If det_out is already a tensor (unlikely based on error)
            if isinstance(det_out, torch.Tensor):
                preds = det_out.reshape(-1, det_out.shape[-1])
            else:
                print(f"Unexpected det_out type: {type(det_out)}, skipping detection loss")
                return loss_seg
        
        # Check if we have valid predictions and targets
        if preds.shape[0] == 0 or targets.numel() == 0:
            print("Empty predictions or targets, skipping detection loss")
            return loss_seg

        # Process IoU calculation per batch to maintain batch structure
        print(f"Target boxes shape: {targets.shape}")
        
        batch_size = preds.shape[0]
        total_loss_iou = 0.0
        
        # Print detection tensor shape for debugging
        print(f"Processing detection tensor with shape: {preds.shape}")
        
        # Threshold predictions based on objectness/confidence
        # If we have confidence scores (which should be at index 4)
        confidence_threshold = 0.1  # Adjust as needed
        if preds.shape[2] > 4:  # If we have more than just box coordinates
            # Get confidence scores (usually at index 4)
            confidence_scores = preds[..., 4]
            print(f"Confidence scores shape: {confidence_scores.shape}")
            print(f"Confidence stats: min={confidence_scores.min().item():.4f}, max={confidence_scores.max().item():.4f}, mean={confidence_scores.mean().item():.4f}")
        
        # Calculate IoU loss for each batch separately
        for b in range(batch_size):
            # Get predictions for this batch
            # preds shape is [batch, num_preds, features]
            batch_preds = preds[b]  # shape: [num_preds, features]
            
            # Skip if no predictions or targets for this batch
            if batch_preds.shape[0] == 0 or targets.shape[0] == 0:
                continue
            
            # Print the feature dimension for debugging
            print(f"Batch {b} prediction features dimension: {batch_preds.shape[1]}")
            
            # Filter predictions based on confidence if available
            if batch_preds.shape[1] > 4:
                confident_mask = batch_preds[:, 4] > confidence_threshold
                if confident_mask.sum() > 0:
                    print(f"Batch {b}: Keeping {confident_mask.sum().item()}/{batch_preds.shape[0]} predictions with confidence > {confidence_threshold}")
                    batch_preds = batch_preds[confident_mask]
                else:
                    print(f"Batch {b}: No predictions above confidence threshold {confidence_threshold}, using all")
                
            # Extract coordinates - these are already in normalized format [0-1]
            # The box format from our processing is [x_center, y_center, width, height]
            pred_x, pred_y, pred_w, pred_h = (batch_preds[:, 0:4].T * self.img_size)
            
            # Convert to [x1, y1, x2, y2] format for IoU calculation
            pred_boxes = torch.stack([
                pred_x - pred_w / 2,  # x1
                pred_y - pred_h / 2,  # y1
                pred_x + pred_w / 2,  # x2
                pred_y + pred_h / 2   # y2
            ], 1)
            
            # Get target boxes for this batch
            # We need to find which targets belong to this batch
            # In a typical YOLO dataset, targets have shape [num_targets, 6] 
            # where targets[:,0] is the batch index
            batch_targets = targets[targets[:, 0] == b]
            
            # Skip if no targets for this batch
            if batch_targets.shape[0] == 0:
                continue
                
            # Extract coordinates and convert to absolute
            # Target format is typically [batch_idx, class_id, x, y, w, h]
            tgt_x, tgt_y, tgt_w, tgt_h = (batch_targets[:, 2:6].T * self.img_size)
            
            # Convert to [x1, y1, x2, y2] format
            tgt_boxes = torch.stack([
                tgt_x - tgt_w / 2,  # x1
                tgt_y - tgt_h / 2,  # y1
                tgt_x + tgt_w / 2,  # x2
                tgt_y + tgt_h / 2   # y2
            ], 1)
            
            # Calculate IoU for this batch
            print(f"Batch {b}: pred_boxes shape {pred_boxes.shape}, tgt_boxes shape {tgt_boxes.shape}")
            
            # Vectorized IoU calculation for efficiency
            ious = torch.zeros((len(pred_boxes), len(tgt_boxes)), device=pred_boxes.device)
            
            # Compute IoU matrix between all predictions and all targets
            if len(pred_boxes) > 0 and len(tgt_boxes) > 0:
                # Expand dimensions for broadcasting
                pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0:1], pred_boxes[:, 1:2], pred_boxes[:, 2:3], pred_boxes[:, 3:4]
                tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tgt_boxes[:, 0:1], tgt_boxes[:, 1:2], tgt_boxes[:, 2:3], tgt_boxes[:, 3:4]
                
                # Calculate intersection coordinates
                x1 = torch.max(pred_x1, tgt_x1.t())
                y1 = torch.max(pred_y1, tgt_y1.t())
                x2 = torch.min(pred_x2, tgt_x2.t())
                y2 = torch.min(pred_y2, tgt_y2.t())
                
                # Calculate intersection area
                w = (x2 - x1).clamp(min=0)
                h = (y2 - y1).clamp(min=0)
                intersection = w * h
                
                # Calculate union area
                pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
                union = pred_area + tgt_area.t() - intersection
                
                # Calculate IoU
                ious = intersection / (union + 1e-7)
                
                # Use best IoU for each prediction
                best_ious, _ = ious.max(dim=1)
                
                # Add debugging information about IoU calculations
                print(f"Batch {b}: Calculated IoU matrix of shape {ious.shape}")
                print(f"Batch {b}: IoU values stats - min: {ious.min().item():.4f}, max: {ious.max().item():.4f}, mean: {ious.mean().item():.4f}")
                
                # Calculate batch loss as 1 - mean of best IoUs
                batch_loss = 1.0 - best_ious.mean()
                total_loss_iou += batch_loss
            else:
                print(f"Batch {b}: No valid IoU calculations performed")
        
        # Average across batches that had valid predictions and targets
        loss_iou = total_loss_iou / batch_size if batch_size > 0 else torch.tensor(0.0, device=preds.device)
        
        # Weight the losses (you can adjust these weights as needed)
        loss_weight_seg = 1.0
        loss_weight_iou = 1.0
        
        # Combine losses with weights
        total_loss = loss_weight_seg * loss_seg + loss_weight_iou * loss_iou
        
        # Log final loss components
        print(f"Segmentation loss: {loss_seg.item():.4f}, IoU loss: {loss_iou.item():.4f}, Total loss: {total_loss.item():.4f}")
        
        return total_loss

    # ----------------------- Lightning hooks
    def forward(self, x, mode="train"):
        return self.net(x, mode=mode)

    def training_step(self, batch, batch_idx):
        ids, imgs, det_boxes, masks, img_cls = batch
        det_out, seg_out, cls_logits = self(imgs)

        # ── losses ───────────────────────────────────────
        loss_detseg = self._multitask_loss(det_out, seg_out, det_boxes, masks)
        loss_cls = self.ce(cls_logits, img_cls)
        loss = loss_detseg + loss_cls

        # ── metrics (logits are fine) ────────────────────
        self.train_acc(cls_logits, img_cls)
        self.train_prec(cls_logits, img_cls)
        self.train_rec(cls_logits, img_cls)
        self.train_f1(cls_logits, img_cls)

        # ── step-level scalar logging (unique keys) ─────
        self.log_dict(
            {"train/loss_step": loss,
             "train/cls_loss_step": loss_cls,
             "train/detseg_loss_step": loss_detseg},
            prog_bar=True, on_step=True, on_epoch=False, sync_dist=True
        )

        # Log segmentation masks periodically to wandb
        if self.global_step % self.mask_log_period == 0:
            # Extract predicted segmentation masks
            if isinstance(seg_out, (list, tuple)):
                if len(seg_out) == 3:
                    _, coeffs, protos = seg_out
                elif len(seg_out) == 2:
                    coeffs, protos = seg_out
                else:
                    return loss  # Skip logging if unexpected format
            else:
                return loss  # Skip logging if unexpected format
                
            # Helper function to extract tensors from nested lists/tuples
            def _to_tensor(t):
                while isinstance(t, (list, tuple)):
                    if len(t) == 0:
                        return None
                    t = t[0]
                return t
                
            coeffs = _to_tensor(coeffs)
            protos = _to_tensor(protos)
            
            if coeffs is None or protos is None:
                return loss  # Skip logging if tensors couldn't be extracted
                
            # Get batch size and dimensions for reshaping
            batch_size, num_queries, coeff_dim = coeffs.shape
            proto_channels = protos.shape[1]
            
            # Reshape coefficients to match prototype channels if needed
            if coeff_dim != proto_channels:
                # Try common spatial dimensions for feature maps
                spatial_candidates = [(80, 105), (90, 93), (84, 100), (70, 120)]
                found_spatial_arrangement = False
                
                for height, width in spatial_candidates:
                    if abs(height * width - coeff_dim) < 100:
                        try:
                            # Reshape as [B, Q, H, W]
                            spatial_coeffs = coeffs.reshape(batch_size, num_queries, height, width)
                            # Reshape to [B*Q, 1, H, W] for pooling
                            flat_spatial = spatial_coeffs.reshape(-1, 1, height, width)
                            # Pool to proto_channels
                            pooled = F.adaptive_avg_pool2d(flat_spatial, (proto_channels, 1))
                            # Reshape to [B, Q, proto_channels]
                            coeffs_reshaped = pooled.reshape(batch_size, num_queries, proto_channels)
                            found_spatial_arrangement = True
                            break
                        except (RuntimeError, ValueError):
                            continue
                
                # If spatial arrangement wasn't found, use previous methods
                if not found_spatial_arrangement:
                    if abs(coeff_dim - (protos.shape[2] * protos.shape[3] * proto_channels)) < proto_channels:
                        # Reshape from flattened format
                        reshaped = coeffs.reshape(batch_size, num_queries, proto_channels, protos.shape[2], protos.shape[3])
                        coeffs_reshaped = reshaped.mean(dim=(3, 4))
                    elif coeff_dim > proto_channels:
                        # Use average pooling to reduce dimensions
                        flat_coeffs = coeffs.reshape(-1, 1, 1, coeff_dim)
                        pooled = F.adaptive_avg_pool2d(flat_coeffs, (1, proto_channels))
                        coeffs_reshaped = pooled.reshape(batch_size, num_queries, proto_channels)
                    else:
                        # Use first proto_channels values
                        coeffs_reshaped = coeffs[:, :, :proto_channels]
            else:
                coeffs_reshaped = coeffs
                
            # Generate predicted segmentation masks
            seg_pred = torch.einsum("bqc,bchw->bqhw", coeffs_reshaped, protos)
            
            # Resize to match ground truth mask size
            seg_pred_resized = F.interpolate(
                seg_pred,
                size=(masks.shape[2], masks.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            # Only log first image in batch to save resources
            # Apply sigmoid to convert logits to probability
            pred_mask = torch.sigmoid(seg_pred_resized[0, 0]).detach().cpu().numpy()
            gt_mask = masks[0, 0].detach().cpu().numpy()
            
            # Scale to 0-255 for better visualization
            pred_mask_viz = (pred_mask * 255).astype(np.uint8)
            gt_mask_viz = (gt_mask * 255).astype(np.uint8)
            
            # Create side-by-side comparison
            mask_comparison = wandb.Image(
                np.hstack([pred_mask_viz, gt_mask_viz]),
                caption=f"Predicted Mask (left) vs Ground Truth (right)"
            )
            
            # Log to wandb
            self.logger.experiment.log({
                "segmentation/mask_comparison": mask_comparison,
                "global_step": self.global_step
            })
            
        return loss

    def validation_step(self, batch, batch_idx):
        ids, imgs, det_boxes, masks, img_cls = batch

        det_out, seg_out, cls_logits = self(imgs)
        loss_detseg = self._multitask_loss(det_out, seg_out, det_boxes, masks)
        loss_cls = self.ce(cls_logits, img_cls)
        loss = loss_detseg + loss_cls

        # ── update TorchMetrics (logits are fine; they arg-max internally) ──
        self.val_acc(cls_logits, img_cls)
        self.val_prec(cls_logits, img_cls)
        self.val_rec(cls_logits, img_cls)
        self.val_f1(cls_logits, img_cls)

        # ── optional: log step-level scalars (renamed *_step) ───────────────
        self.log_dict(
            {"val/loss_step": loss,
             "val/cls_loss_step": loss_cls,
             "val/detseg_loss_step": loss_detseg},
            prog_bar=False, on_step=True, on_epoch=False, sync_dist=True
        )

    def on_train_epoch_end(self):
        self.log_dict(
            {"train/accuracy": self.train_acc.compute(),
             "train/precision": self.train_prec.compute(),
             "train/recall": self.train_rec.compute(),
             "train/f1": self.train_f1.compute()},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

    # ──────────────────────── val-epoch end (no args) ───────────────────────────
    def on_validation_epoch_end(self):
        self.log_dict(
            {"val/accuracy": self.val_acc.compute(),
             "val/precision": self.val_prec.compute(),
             "val/recall": self.val_rec.compute(),
             "val/f1": self.val_f1.compute()},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

    def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
            return [opt], [sched]


# ────────────────────────────────────────────────── DataModule
class BTXRDDataModule(pl.LightningDataModule):
    def __init__(self, root="btxrd_ready", batch_size=4, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set = BTXRD(self.root, split="train", img_size=640)
        self.val_set = BTXRD(self.root, split="val", img_size=640)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)


# ────────────────────────────────────────────────── Main
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    logger = WandbLogger(project="BoneTumor-MultiTask", log_model=True)

    data = BTXRDDataModule(batch_size=4, num_workers=2)
    model = MultiTaskLitModel(lr=1e-4, mask_log_period=50)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=logger,
    )

    trainer.fit(model, datamodule=data)
