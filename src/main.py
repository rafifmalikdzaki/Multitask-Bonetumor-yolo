import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics import YOLO
from dataset_btxrd_new import BTXRD, collate_fn
from tqdm import tqdm
from model import ConvNeXtBiFPNYOLO, load_pretrained_heads

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Ready!")

# Load model
model = ConvNeXtBiFPNYOLO(nc_det=3, nc_img=3).to(device)
model = load_pretrained_heads(model).to(device)
print("Model Ready!")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Load YOLO pretrained criterion (seg variant)
yolo_model = YOLO("yolo11m-seg.pt")
yolo_model.model.train()
criterion = yolo_model.model.criterion  # Now it should be initialized
print(criterion)

# Classification loss
classification_loss_fn = torch.nn.CrossEntropyLoss()

# Data loaders
train_loader = DataLoader(
    BTXRD(split="train"), batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    BTXRD(split="val"), batch_size=4, shuffle=False, collate_fn=collate_fn
)
print("Data Ready!")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for ids, imgs, det_boxes, masks, img_cls in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        det_boxes = det_boxes.to(device)
        img_cls = img_cls.to(device)

        # print("IDs:", ids)
        # print("Images shape:", imgs.shape)
        # print("Detection boxes:", det_boxes.shape)
        # print("Masks shape:", masks.shape, "Unique values:", torch.unique(masks[0]))
        # print("Image classes:", img_cls)

        optimizer.zero_grad()
        det_out, seg_out, cls_logits = model(imgs, mode="train")

        yolo_loss, _ = criterion((det_out, seg_out), det_boxes, masks)
        cls_loss = classification_loss_fn(cls_logits, img_cls)

        loss = yolo_loss + cls_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, imgs, det_boxes, masks, img_cls in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            det_boxes = det_boxes.to(device)
            img_cls = img_cls.to(device)

            det_out, seg_out, cls_logits = model(imgs, mode="train")

            yolo_loss, _ = criterion((det_out, seg_out), det_boxes, masks)
            cls_loss = classification_loss_fn(cls_logits, img_cls)
            loss = yolo_loss + cls_loss

            total_loss += loss.item()

            pred_cls = cls_logits.argmax(1)
            correct += (pred_cls == img_cls).sum().item()
            total += img_cls.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Training loop
epochs = 20
for epoch in tqdm(range(epochs)):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    # ids, imgs, det_boxes, masks, img_cls = next(iter(train_loader))
    # print("IDs:", ids)
    # print("Images shape:", imgs.shape)
    # print("Detection boxes:", det_boxes.shape)
    # print("Masks shape:", masks.shape, "Unique values:", torch.unique(masks[0]))
    # print("Image classes:", img_cls)

    # print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | "
    #       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
