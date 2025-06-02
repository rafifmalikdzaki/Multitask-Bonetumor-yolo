import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchmetrics.functional import dice
from ultralytics.utils.metrics import ap_per_class
from dataset_btxrd_new import BTXRD, collate_fn
from main_model import ConvNeXtBiFPNYOLO, load_pretrained_heads
from light import MultiTaskLitModel
import numpy as np


# ------------------------------- Evaluation Helpers ----------------------------------
def compute_iou(preds, targets, eps=1e-7):
    inter = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    return (inter + eps) / (union + eps)


def compute_dice(preds, targets, eps=1e-7):
    inter = (preds & targets).float().sum((1, 2))
    return (2 * inter + eps) / (preds.sum((1, 2)) + targets.sum((1, 2)) + eps)


# ------------------------------- Data and Model Loading ----------------------------------
class BTXRDTestModule(pl.LightningDataModule):
    def __init__(self, root="btxrd_ready", batch_size=1, num_workers=2):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.test_set = BTXRD(self.root, split="test", img_size=640)

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


# ------------------------------- Evaluation Script ----------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = BTXRDTestModule()
    data.setup()
    test_loader = data.test_dataloader()

    model = MultiTaskLitModel.load_from_checkpoint("best_model.ckpt")
    model.eval()
    model.to(device)

    cls_preds, cls_targets = [], []
    seg_ious, seg_dices, seg_mses = [], [], []
    det_ious, det_ap50s = [], []

    with torch.no_grad():
        for ids, imgs, det_boxes, masks, img_cls in test_loader:
            imgs = imgs.to(device)
            det_boxes = det_boxes.to(device)
            masks = masks.to(device)
            img_cls = img_cls.to(device)

            det_out, seg_out, cls_logits = model(imgs)

            # Classification evaluation
            cls_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()
            cls_tgt = img_cls.cpu().numpy()
            cls_preds.extend(cls_pred)
            cls_targets.extend(cls_tgt)

            # Segmentation evaluation
            _, coeffs, protos = seg_out
            seg_pred = torch.einsum("bqc,bchw->bqhw", coeffs, protos)
            seg_pred = F.interpolate(
                seg_pred, size=masks.shape[2:], mode="bilinear", align_corners=False
            )
            seg_pred = torch.sigmoid(seg_pred) > 0.5

            seg_iou = compute_iou(seg_pred[:, 0], masks[:, 0].bool()).mean().item()
            seg_dice = compute_dice(seg_pred[:, 0], masks[:, 0].bool()).mean().item()
            seg_mse = F.mse_loss(seg_pred.float(), masks.float()).item()

            seg_ious.append(seg_iou)
            seg_dices.append(seg_dice)
            seg_mses.append(seg_mse)
            ious = compute_iou(pred_boxes.bool(), tgt_boxes.bool()).mean().item()
            det_ious.append(ious)

    # ------------------ Classification ------------------
    print("\nClassification Report:")
    print("Accuracy:", accuracy_score(cls_targets, cls_preds))
    print("Precision:", precision_score(cls_targets, cls_preds, average="micro"))
    print("Recall:", recall_score(cls_targets, cls_preds, average="micro"))
    print("F1 Score:", f1_score(cls_targets, cls_preds, average="micro"))

    # ------------------ Segmentation ------------------
    print("\nSegmentation Report:")
    print("Mean IoU:", np.mean(seg_ious))
    print("Mean Dice:", np.mean(seg_dices))
    print("Mean MSE:", np.mean(seg_mses))

    # ------------------ Detection ------------------
    print("\nDetection Report:")
    print("Mean IoU:", np.mean(det_ious))


if __name__ == "__main__":
    evaluate()
