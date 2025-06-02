from pathlib import Path
import cv2  # cv2 needs to be imported
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np


class BTXRD(Dataset):
    def __init__(
        self,
        root_dir: str | Path = "btxrd_ready",
        split: str = "train",
        img_size: int = 640,
        transform=None,
        train_ratio: float = 0.8,
        seed: int = 42,  # <-- new: reproducible shuffle
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.det_dir = self.root_dir / "labels_det"

        self.split = split.lower()
        self.img_size = img_size
        self.transform = transform
        self.train_ratio = train_ratio
        self.rng = np.random.RandomState(seed)

        # ---------- load image-level classes ----------
        try:
            cls_rows = pd.read_csv(
                self.root_dir / "img_cls.csv",
                header=None,
                names=["filename", "class"],
            ).to_dict(orient="records")
            self.cls_lookup = {r["filename"]: r["class"] for r in cls_rows}
        except FileNotFoundError:
            print(f"[BTXRD] img_cls.csv not found in {self.root_dir}")
            self.cls_lookup = {}

        # ---------- gather usable samples ----------
        complete_items = []
        for idx, img_path in enumerate(sorted(self.img_dir.glob("*.jpeg"))):
            stem = img_path.stem
            txt_path = self.det_dir / f"{stem}.txt"
            mask_path = self.mask_dir / f"{stem}.png"
            if (
                txt_path.exists()
                and mask_path.exists()
                and img_path.name in self.cls_lookup
            ):
                complete_items.append(
                    dict(
                        id=idx,
                        img_path=img_path,
                        txt_path=txt_path,
                        msk_path=mask_path,
                        class_id=self.cls_lookup[img_path.name],
                    )
                )

        if not complete_items:
            print(f"[BTXRD] No items found in {root_dir}")
            self.items = []
            return

        # ---------- STRATIFIED split ----------
        buckets = defaultdict(list)  # class_id ➜ list[item]
        for it in complete_items:
            buckets[it["class_id"]].append(it)

        train_items, val_items = [], []
        for cls, bucket in buckets.items():
            self.rng.shuffle(bucket)  # deterministic
            k = int(round(self.train_ratio * len(bucket)))
            train_items.extend(bucket[:k])
            val_items.extend(bucket[k:])

        if self.split == "train":
            self.items = train_items
        elif self.split in {"val", "valid", "validation"}:
            self.items = val_items
        else:  # e.g. "all", "test"
            self.items = complete_items

    def __len__(self):
        return len(self.items)

    def _letterbox(self, img, mask):
        """Resizes and pads image and mask to a square, maintaining aspect ratio."""
        H0, W0 = img.shape[:2]  # Original height, width of image passed to letterbox
        S = self.img_size  # Target size (square)

        # Calculate scale factor and new dimensions
        scale = S / max(H0, W0)
        new_w, new_h = int(W0 * scale), int(H0 * scale)

        # Resize image and mask
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Calculate padding
        # Padding is applied to right and bottom to align content to top-left
        pad_w = S - new_w
        pad_h = S - new_h
        top_pad, left_pad = 0, 0  # No padding on top/left

        # Apply padding
        img_letterboxed = cv2.copyMakeBorder(
            img_resized,
            top_pad,
            pad_h,
            left_pad,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )  # Common padding color
        mask_letterboxed = cv2.copyMakeBorder(
            mask_resized, top_pad, pad_h, left_pad, pad_w, cv2.BORDER_CONSTANT, value=0
        )  # Pad mask with 0

        return (
            img_letterboxed,
            mask_letterboxed,
            scale,
            left_pad,
            top_pad,
        )  # Return left_pad, top_pad for coord adjustment

    def __getitem__(self, idx):
        item_data = self.items[idx]
        img_fp = str(item_data["img_path"])
        txt_fp = item_data["txt_path"]
        msk_fp = str(item_data["msk_path"])
        class_id = item_data["class_id"]
        original_idx = item_data["id"]  # This is the 'id' to be returned

        # Load image
        img = cv2.imread(img_fp)
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_fp}")
        H0_orig, W0_orig = img.shape[
            :2
        ]  # Original dimensions for YOLO coord denormalization

        # Load mask (assuming it's a single-channel grayscale image, e.g., binary mask)
        mask = cv2.imread(msk_fp, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if mask is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {msk_fp}")

        # Apply letterboxing
        img_letterboxed, mask_letterboxed, scale, pad_l, pad_t = self._letterbox(
            img.copy(), mask.copy()
        )

        # Process image: BGR to RGB, normalize, permute to CHW
        img_processed = (
            cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        img_t = torch.from_numpy(img_processed).permute(2, 0, 1)

        # Process mask: add channel dim, normalize (assuming mask values are 0 or 255)
        # If mask values are already 0 or 1, remove division by 255.0
        mask_t = torch.from_numpy(mask_letterboxed[None, :, :]).float() / 255.0
        mask_t = (mask_t > 0.5).float()  # Ensure binary 0 or 1

        # Load and process detection boxes (YOLO format: class xc yc w h normalized)
        det_rows = []
        if txt_fp.exists():
            for line in txt_fp.read_text().splitlines():
                try:
                    cls, xc_yolo, yc_yolo, w_yolo, h_yolo = map(float, line.split())
                except ValueError:
                    print(f"Warning: Skipping malformed line in {txt_fp}: '{line}'")
                    continue

                # 1. Denormalize YOLO coords to absolute coords in original image
                abs_xc_orig = xc_yolo * W0_orig
                abs_yc_orig = yc_yolo * H0_orig
                abs_w_orig = w_yolo * W0_orig
                abs_h_orig = h_yolo * H0_orig

                # 2. Convert to x1,y1,x2,y2 absolute in original image
                abs_x1_orig = abs_xc_orig - abs_w_orig / 2
                abs_y1_orig = abs_yc_orig - abs_h_orig / 2
                abs_x2_orig = abs_xc_orig + abs_w_orig / 2
                abs_y2_orig = abs_yc_orig + abs_h_orig / 2

                # 3. Apply letterbox scaling
                scaled_x1 = abs_x1_orig * scale
                scaled_y1 = abs_y1_orig * scale
                scaled_x2 = abs_x2_orig * scale
                scaled_y2 = abs_y2_orig * scale

                # 4. Apply letterbox padding (pad_l, pad_t are 0 with current letterbox)
                # If padding was centered, you'd add pad_l to x-coords and pad_t to y-coords.
                # With top-left alignment, no addition of pad_l, pad_t is needed here as they are 0.
                # The coordinates are already relative to the top-left of the letterboxed image.

                # 5. Re-normalize to [0,1] with respect to self.img_size
                # And convert back to xc, yc, w, h for the final output tensor
                final_xc_norm = ((scaled_x1 + scaled_x2) / 2) / self.img_size
                final_yc_norm = ((scaled_y1 + scaled_y2) / 2) / self.img_size
                final_w_norm = (scaled_x2 - scaled_x1) / self.img_size
                final_h_norm = (scaled_y2 - scaled_y1) / self.img_size

                # Clamp coordinates to be within [0, 1] and ensure w, h are positive
                final_xc_norm = np.clip(final_xc_norm, 0.0, 1.0)
                final_yc_norm = np.clip(final_yc_norm, 0.0, 1.0)
                final_w_norm = np.clip(final_w_norm, 0.0, 1.0)
                final_h_norm = np.clip(final_h_norm, 0.0, 1.0)

                # First element is placeholder for batch index, set in collate_fn
                det_rows.append(
                    [
                        0.0,
                        float(cls),
                        final_xc_norm,
                        final_yc_norm,
                        final_w_norm,
                        final_h_norm,
                    ]
                )

        det_boxes = (
            torch.tensor(det_rows, dtype=torch.float32)
            if det_rows
            else torch.zeros((0, 6), dtype=torch.float32)
        )

        # Image-level classification
        img_cls = torch.tensor(class_id, dtype=torch.long)

        return original_idx, img_t, det_boxes, mask_t, img_cls


def collate_fn(batch):
    # Unzip the batch
    ids, imgs, dets, masks, img_cls_list = zip(*batch)

    # Stack images, masks, and image classes
    imgs_stacked = torch.stack(imgs)
    masks_stacked = torch.stack(masks)
    # Ensure img_cls are tensors before stacking
    img_cls_stacked = torch.stack(
        [torch.as_tensor(c, dtype=torch.long) for c in img_cls_list]
    )

    # Collate detection boxes (they have variable numbers per image)
    batch_det_boxes = []
    for i, boxes in enumerate(dets):  # i is the batch index for this item
        if boxes.numel() > 0:  # Check if there are any boxes
            # boxes_cloned = boxes.clone() # Not strictly necessary if original tensor is not modified elsewhere
            boxes[:, 0] = float(
                i
            )  # Set the first column (batch_idx) to the current image's index in the batch
            batch_det_boxes.append(boxes)

    if batch_det_boxes:
        det_boxes_collated = torch.cat(batch_det_boxes, 0)
    else:
        # If no boxes in the entire batch, create an empty tensor with correct shape
        det_boxes_collated = torch.zeros((0, 6), dtype=torch.float32)

    return list(ids), imgs_stacked, det_boxes_collated, masks_stacked, img_cls_stacked


from collections import Counter
import pprint

from tqdm import tqdm


def class_histogram(ds):
    """Return a {class_id: count, …} dict for the dataset."""
    hist = Counter()
    for _, _, _, _, cls in tqdm(ds):  # cls is a 0-D tensor
        hist[int(cls)] += 1
    return hist


def print_sample(ds, idx: int = 0):
    """Fetch one sample and dump its fields."""
    orig_id, img_t, det_boxes, mask_t, cls = ds[idx]

    print("\n--- sample inspection --------------------------------")
    print(f"dataset idx   : {idx}")
    print(f"original id   : {orig_id}")
    print(f"class id      : {cls.item()}")
    print(
        f"image tensor  : shape={tuple(img_t.shape)}, "
        f"dtype={img_t.dtype}, min={img_t.min():.3f}, max={img_t.max():.3f}"
    )
    print(
        f"mask tensor   : shape={tuple(mask_t.shape)}, "
        f"unique values={torch.unique(mask_t).tolist()}"
    )
    print(f"det-boxes     : {det_boxes.shape[0]} box(es)")
    if det_boxes.numel():
        print(det_boxes)  # (first col is batch_idx placeholder, second is class)


if __name__ == "__main__":
    root = "btxrd_ready"  # ← change to your real path
    train_ds = BTXRD(root_dir=root, split="train", seed=42)
    val_ds = BTXRD(root_dir=root, split="val", seed=42)

    train_hist = class_histogram(train_ds)
    val_hist = class_histogram(val_ds)

    print("\n--- smoke-test: class distributions ---")
    print("train split:")
    pprint.pprint(train_hist)
    print("val split:")
    pprint.pprint(val_hist)

    # Optional sanity check ─ ratio per class ≅ the requested train_ratio
    train_ratio = train_ds.train_ratio
    for cls in sorted(set(train_hist) | set(val_hist)):
        n_train = train_hist.get(cls, 0)
        n_val = val_hist.get(cls, 0)
        if n_train + n_val == 0:
            continue
        actual_ratio = n_train / (n_train + n_val)
        assert (
            abs(actual_ratio - train_ratio) < 0.01
        ), f"class {cls}: expected ~{train_ratio:.2f}, got {actual_ratio:.2f}"
    print("✔️  stratified split passes ratio check")

    # Print one sample from each split
    print_sample(train_ds, idx=0)
    if len(val_ds):
        print_sample(val_ds, idx=0)
