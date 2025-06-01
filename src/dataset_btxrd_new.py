from pathlib import Path
import csv, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class BTXRD(Dataset):
    def __init__(self, root_dir="btxrd_ready",
                 split='train',
                 img_size=640,
                 transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir  = self.root_dir/"images"
        self.mask_dir = self.root_dir/"masks"
        self.det_dir  = self.root_dir/"labels_det"

        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.cls_rows = pd.read_csv(self.root_dir/"img_cls.csv", header=None, names=['filename', 'class']).to_dict(orient='records')
        self.cls_lookup = { r['filename']: r['class'] for r in self.cls_rows }

        self.items = []
        for idx, img_path in enumerate(sorted(self.img_dir.glob("*.jpeg"))):
            fname = img_path.name                    # e.g. "IMG000001.jpeg"
            txt_path = self.det_dir / f"{img_path.stem}.txt"
            msk_path = self.mask_dir / f"{img_path.stem}.png"

            if txt_path.exists() and msk_path.exists() and fname in self.cls_lookup:
                class_id = self.cls_lookup[fname]
                self.items.append((idx, img_path, txt_path, msk_path, class_id))

        n = len(self.items)
        self.items = self.items[:int(.8*n)] if split == 'train' else self.items[int(.8*n):]

    def __len__(self):
        return len(self.items)

    def _letterbox(self, img, mask):
        H0, W0 = img.shape[:2]
        S = self.img_size
        scale = S / max(H0, W0)
        new_w, new_h = int(W0 * scale), int(H0 * scale)
        img = cv2.resize(img, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_w, pad_h = S - new_w, S - new_h
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return img, mask, scale, pad_w, pad_h

    def __getitem__(self, idx):
        id, img_fp, txt_fp, msk_fp, class_id = self.items[idx]

        img = cv2.imread(str(img_fp))
        mask = cv2.imread(str(msk_fp), cv2.IMREAD_UNCHANGED)

        img, mask, scale, pad_w, pad_h = self._letterbox(img, mask)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img_t = torch.from_numpy(img).permute(2,0,1)
        mask_t = torch.from_numpy(mask[None]).float() / 127.

        # Detection boxes
        det_rows = []
        for line in txt_fp.read_text().splitlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = (xc - bw/2) * img.shape[2]   # after resize+pad img.shape[2] == S
            y1 = (yc - bh/2) * img.shape[1]
            x2 = (xc + bw/2) * img.shape[2]
            y2 = (yc + bh/2) * img.shape[1]

            # re‑normalise
            xc_new = (x1+x2)/2 / self.img_size
            yc_new = (y1+y2)/2 / self.img_size
            bw_new = (x2-x1)   / self.img_size
            bh_new = (y2-y1)   / self.img_size
            det_rows.append([0, cls, xc_new, yc_new, bw_new, bh_new])

        det_boxes = (torch.tensor(det_rows, dtype=torch.float32) if det_rows else torch.zeros((0,6), dtype=torch.float32))
        img_cls = torch.tensor(class_id, dtype=torch.long)
        return id, img_t, det_boxes, mask_t, img_cls

def collate_fn(batch):
    ids, imgs, dets, masks, img_cls = zip(*batch)

    imgs  = torch.stack(imgs)                              # images
    masks = torch.stack(masks)                             # semantic masks
    img_cls = torch.stack([torch.as_tensor(c)              # ensure Tensor
                           for c in img_cls])

    batch_boxes = []
    for i, boxes in enumerate(dets):
        if boxes.numel():                                  # at least 1 box
            boxes = boxes.clone()                          # avoid in‑place on dataset tensor
            boxes[:, 0] = i                                # set batch‑id column
            batch_boxes.append(boxes)

    det_boxes = (torch.cat(batch_boxes, 0)
                 if batch_boxes else
                 torch.zeros((0, 6), dtype=torch.float32))

    return list(ids), imgs, det_boxes, masks, img_cls

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_ds = BTXRD(root_dir="btxrd_ready", split="train",
                     img_size=640)
    val_ds   = BTXRD(root_dir="btxrd_ready", split="val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=1, pin_memory=True,
                              collate_fn=collate_fn)

    id, imgs, det, msk, cls = next(iter(train_loader))
    print("id:", id)
    print("images:", imgs.shape)
    print("det sample:", det.shape)
    from pprint import pprint
    pprint(det)
    print("mask:", msk.shape, "unique", torch.unique(msk[0]))
    print("img class ids:", cls)

