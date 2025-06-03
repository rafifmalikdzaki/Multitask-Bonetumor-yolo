from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import numpy as np
import pprint
from tqdm import tqdm


class BTXRD(Dataset):
    def __init__(
        self,
        root_dir: str | Path = "btxrd_ready",
        split: str = "train",
        img_size: int = 640,
        transform=None, # Currently unused, but kept for future
        train_ratio: float = 0.8,
        seed: int = 42,
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
        self.rng = np.random.RandomState(seed) # For reproducible shuffling

        # ---------- load image-level classes ----------
        try:
            cls_rows = pd.read_csv(
                self.root_dir / "img_cls.csv",
                header=None,
                names=["filename", "class"],
            ).to_dict(orient="records")
            self.cls_lookup = {r["filename"]: r["class"] for r in cls_rows}
        except FileNotFoundError:
            print(f"[BTXRD] Warning: img_cls.csv not found in {self.root_dir}. Image classes will not be loaded.")
            self.cls_lookup = {}

        # ---------- gather usable samples ----------
        complete_items = []
        # Using .jpeg as per original, ensure this matches your file extensions
        for item_idx, img_path in enumerate(sorted(self.img_dir.glob("*.jpeg"))):
            stem = img_path.stem
            txt_path = self.det_dir / f"{stem}.txt"
            mask_path = self.mask_dir / f"{stem}.png" # Common mask extension
            
            # Check if image name (with extension) is in cls_lookup
            if img_path.name not in self.cls_lookup and self.cls_lookup: # Only warn if cls_lookup was loaded
                # print(f"[BTXRD] Warning: Image {img_path.name} not found in img_cls.csv. Skipping.")
                continue # Skip if image class is mandatory and not found

            if txt_path.exists() and mask_path.exists():
                complete_items.append(
                    dict(
                        id=item_idx, # Using enumerate index as a unique ID for the item
                        img_path=img_path,
                        txt_path=txt_path,
                        msk_path=mask_path,
                        # Provide default class_id if not in lookup but cls_lookup is empty (optional behavior)
                        class_id=self.cls_lookup.get(img_path.name, 0) # Default to class 0 if not found
                    )
                )
            # else:
            #     print(f"[BTXRD] Debug: Missing components for {stem}. txt: {txt_path.exists()}, mask: {mask_path.exists()}")


        if not complete_items:
            print(f"[BTXRD] Error: No complete items found in {root_dir} with .jpeg images, .txt labels, and .png masks. Please check paths and file extensions.")
            self.items = []
            return

        # ---------- STRATIFIED split ----------
        buckets = defaultdict(list)
        for it in complete_items:
            buckets[it["class_id"]].append(it)

        train_items, val_items = [], []
        for cls_val, bucket in buckets.items(): # Renamed cls to cls_val to avoid conflict
            self.rng.shuffle(bucket)
            k = int(round(self.train_ratio * len(bucket)))
            train_items.extend(bucket[:k])
            val_items.extend(bucket[k:])
        
        self.rng.shuffle(train_items) # Shuffle the combined train items
        self.rng.shuffle(val_items)   # Shuffle the combined val items

        if self.split == "train":
            self.items = train_items
        elif self.split in {"val", "valid", "validation"}:
            self.items = val_items
        else: # e.g. "all", "test" - falls back to all complete items
            self.rng.shuffle(complete_items) # Shuffle all items if split is 'all'
            self.items = complete_items
        
        print(f"[BTXRD] Loaded {len(self.items)} items for split '{self.split}'.")


    def __len__(self):
        return len(self.items)

    def _letterbox(self, img, mask):
        """Resizes and pads image and mask to a square, maintaining aspect ratio."""
        H0, W0 = img.shape[:2]
        S = self.img_size

        scale = S / max(H0, W0)
        # Ensure new dimensions are at least 1 pixel
        new_w, new_h = max(1, int(W0 * scale)), max(1, int(H0 * scale))


        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_w = S - new_w
        pad_h = S - new_h
        top_pad, left_pad = 0, 0 # Align top-left

        img_letterboxed = cv2.copyMakeBorder(
            img_resized, top_pad, pad_h, left_pad, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        mask_letterboxed = cv2.copyMakeBorder(
            mask_resized, top_pad, pad_h, left_pad, pad_w,
            cv2.BORDER_CONSTANT, value=0 # Pad mask with 0
        )
        return img_letterboxed, mask_letterboxed, scale, left_pad, top_pad

    def __getitem__(self, idx):
        item_data = self.items[idx]
        img_fp = str(item_data["img_path"])
        txt_fp = item_data["txt_path"]
        msk_fp = str(item_data["msk_path"])
        class_id = item_data["class_id"]
        # 'id' from enumerate is now the original_idx for the item in complete_items list
        original_item_id = item_data["id"] 

        img = cv2.imread(img_fp)
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_fp}")
        H0_orig, W0_orig = img.shape[:2]

        mask = cv2.imread(msk_fp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {msk_fp}")

        img_letterboxed, mask_letterboxed, scale, pad_l, pad_t = self._letterbox(
            img.copy(), mask.copy()
        )

        img_processed = (
            cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        img_t = torch.from_numpy(img_processed).permute(2, 0, 1)
        
        # Ensure mask is binary 0 or 1 after potential division
        mask_np_float = mask_letterboxed.astype(np.float32) / 255.0 
        mask_t = torch.from_numpy(mask_np_float[None, :, :]).float()
        mask_t = (mask_t > 0.5).float()

        det_rows = []
        # Define a small epsilon for width/height check, relative to image size
        # e.g., a box must be at least 1 pixel wide/high in the final img_size space
        min_norm_dim = 1.0 / self.img_size 

        if txt_fp.exists():
            for line_idx, line in enumerate(txt_fp.read_text().splitlines()):
                try:
                    parts = line.split()
                    if len(parts) < 5: # Basic check for enough parts
                        print(f"Warning: Skipping malformed line (not enough parts) in {txt_fp} (line {line_idx+1}): '{line}'")
                        continue
                    cls_label, xc_yolo, yc_yolo, w_yolo, h_yolo = map(float, parts[:5])
                except ValueError:
                    print(f"Warning: Skipping malformed line (ValueError) in {txt_fp} (line {line_idx+1}): '{line}'")
                    continue

                # Check for non-positive width/height from YOLO file directly
                if w_yolo <= 0 or h_yolo <= 0:
                    print(f"Warning: Skipping box with non-positive w/h in YOLO file {txt_fp} (line {line_idx+1}): w={w_yolo}, h={h_yolo}")
                    continue

                abs_xc_orig = xc_yolo * W0_orig
                abs_yc_orig = yc_yolo * H0_orig
                abs_w_orig = w_yolo * W0_orig
                abs_h_orig = h_yolo * H0_orig

                abs_x1_orig = abs_xc_orig - abs_w_orig / 2
                abs_y1_orig = abs_yc_orig - abs_h_orig / 2
                abs_x2_orig = abs_xc_orig + abs_w_orig / 2
                abs_y2_orig = abs_yc_orig + abs_h_orig / 2

                scaled_x1 = abs_x1_orig * scale
                scaled_y1 = abs_y1_orig * scale
                scaled_x2 = abs_x2_orig * scale
                scaled_y2 = abs_y2_orig * scale
                
                # No need to add pad_l, pad_t as they are 0 for top-left alignment

                final_w_abs = scaled_x2 - scaled_x1
                final_h_abs = scaled_y2 - scaled_y1

                # <<< MODIFIED: Filter out zero or too small area boxes AFTER scaling >>>
                if final_w_abs < 1.0 or final_h_abs < 1.0: # Must be at least 1 pixel in scaled space
                    # print(f"Warning: Skipping box with near zero w/h after scaling in {txt_fp} (line {line_idx+1}): w={final_w_abs:.2f}, h={final_h_abs:.2f}")
                    continue
                
                final_xc_norm = ((scaled_x1 + scaled_x2) / 2) / self.img_size
                final_yc_norm = ((scaled_y1 + scaled_y2) / 2) / self.img_size
                final_w_norm = final_w_abs / self.img_size
                final_h_norm = final_h_abs / self.img_size
                
                # Clamp coordinates to be within [0, 1]
                # Also ensure that after clamping, width and height are still valid
                # Convert to x1y1x2y2 for robust clamping
                final_x1_norm = np.clip(final_xc_norm - final_w_norm / 2, 0.0, 1.0)
                final_y1_norm = np.clip(final_yc_norm - final_h_norm / 2, 0.0, 1.0)
                final_x2_norm = np.clip(final_xc_norm + final_w_norm / 2, 0.0, 1.0)
                final_y2_norm = np.clip(final_yc_norm + final_h_norm / 2, 0.0, 1.0)

                # Recalculate w, h from clamped x1y1x2y2
                final_w_norm_clamped = final_x2_norm - final_x1_norm
                final_h_norm_clamped = final_y2_norm - final_y1_norm

                # <<< MODIFIED: Stricter check for final valid box after all ops >>>
                if final_w_norm_clamped < min_norm_dim or final_h_norm_clamped < min_norm_dim:
                    # print(f"Warning: Skipping box with zero/too small w/h after all processing in {txt_fp} (line {line_idx+1}): w_norm={final_w_norm_clamped:.4f}, h_norm={final_h_norm_clamped:.4f}")
                    continue
                
                # Convert back to xc, yc for storage
                final_xc_norm_clamped = (final_x1_norm + final_x2_norm) / 2
                final_yc_norm_clamped = (final_y1_norm + final_y2_norm) / 2

                det_rows.append([
                    0.0, # Placeholder for batch index
                    float(cls_label),
                    final_xc_norm_clamped,
                    final_yc_norm_clamped,
                    final_w_norm_clamped,
                    final_h_norm_clamped,
                ])

        det_boxes = (
            torch.tensor(det_rows, dtype=torch.float32)
            if det_rows
            else torch.zeros((0, 6), dtype=torch.float32)
        )
        img_cls = torch.tensor(class_id, dtype=torch.long)

        # Return the original_item_id which is unique for each item from the initial scan
        return original_item_id, img_t, det_boxes, mask_t, img_cls


def collate_fn(batch):
    ids, imgs, dets, masks, img_cls_list = zip(*batch)

    imgs_stacked = torch.stack(imgs)
    masks_stacked = torch.stack(masks)
    img_cls_stacked = torch.stack(
        [torch.as_tensor(c, dtype=torch.long) for c in img_cls_list]
    )

    batch_det_boxes = []
    for i, boxes in enumerate(dets):
        if boxes.numel() > 0:
            # It's safer to clone if there's any doubt about in-place modification
            # For setting batch_idx, a view or direct assignment is usually fine if `boxes` is not used elsewhere with its original 0th col value
            current_item_boxes = boxes.clone() # Clone to be safe
            current_item_boxes[:, 0] = float(i) 
            batch_det_boxes.append(current_item_boxes)

    if batch_det_boxes:
        det_boxes_collated = torch.cat(batch_det_boxes, 0)
    else:
        det_boxes_collated = torch.zeros((0, 6), dtype=torch.float32)

    return list(ids), imgs_stacked, det_boxes_collated, masks_stacked, img_cls_stacked


# --- Main for testing ---
def class_histogram(ds):
    hist = Counter()
    print(f"Calculating histogram for {len(ds)} items...")
    for item_tuple in tqdm(ds): 
        # Ensure item_tuple has 5 elements as expected
        if len(item_tuple) == 5:
            _, _, _, _, cls_tensor = item_tuple
            hist[int(cls_tensor.item())] += 1 # Use .item() for 0-D tensor
        else:
            print(f"Warning: Unexpected item structure in dataset: {item_tuple}")
    return hist

def print_sample(ds, idx: int = 0):
    if idx >= len(ds):
        print(f"Index {idx} out of bounds for dataset of length {len(ds)}")
        return
    item_tuple = ds[idx]
    if len(item_tuple) != 5:
        print(f"Warning: Unexpected item structure at index {idx}: {item_tuple}")
        return

    original_id, img_t, det_boxes, mask_t, cls = item_tuple
    print("\n--- sample inspection --------------------------------")
    print(f"dataset idx   : {idx}")
    print(f"original id   : {original_id}") # This is the 'id' from item_data
    print(f"class id      : {cls.item()}")
    print(f"image tensor  : shape={tuple(img_t.shape)}, dtype={img_t.dtype}, min={img_t.min():.3f}, max={img_t.max():.3f}")
    print(f"mask tensor   : shape={tuple(mask_t.shape)}, unique values={torch.unique(mask_t).tolist()}")
    print(f"det-boxes     : {det_boxes.shape[0]} box(es)")
    if det_boxes.numel():
        print(det_boxes)


if __name__ == "__main__":
    root = "btxrd_ready" 
    print(f"Attempting to load dataset from: {Path(root).resolve()}")

    train_ds = BTXRD(root_dir=root, split="train", seed=42)
    val_ds = BTXRD(root_dir=root, split="val", seed=42)

    if len(train_ds) > 0 and len(val_ds) > 0:
        train_hist = class_histogram(train_ds)
        val_hist = class_histogram(val_ds)

        print("\n--- smoke-test: class distributions ---")
        print("train split:")
        pprint.pprint(dict(train_hist)) # Convert Counter to dict for pprint
        print("val split:")
        pprint.pprint(dict(val_hist))

        train_ratio_hparam = train_ds.train_ratio
        all_classes = sorted(set(train_hist.keys()) | set(val_hist.keys()))
        
        valid_split = True
        if not all_classes:
             print("Warning: No classes found in either train or val histograms for ratio check.")
        else:
            for cls_val_check in all_classes: # Renamed cls to avoid conflict
                n_train = train_hist.get(cls_val_check, 0)
                n_val = val_hist.get(cls_val_check, 0)
                if n_train + n_val == 0:
                    continue 
                actual_ratio = n_train / (n_train + n_val)
                if not (abs(actual_ratio - train_ratio_hparam) < 0.015): # Loosened tolerance slightly for small N
                    print(f"Class {cls_val_check}: expected ~{train_ratio_hparam:.2f}, got {actual_ratio:.2f} (train: {n_train}, val: {n_val}) - Ratio Mismatch!")
                    valid_split = False
            if valid_split:
                print("✔️  Stratified split passes ratio check (approx).")
            else:
                print("❌ Stratified split FAILED ratio check for one or more classes.")


        print_sample(train_ds, idx=0)
        if len(val_ds):
            print_sample(val_ds, idx=0)
        
        # Test DataLoader
        print("\n--- DataLoader test ---")
        try:
            train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
            for i, batch_data in enumerate(train_loader):
                ids_b, imgs_b, dets_b, masks_b, img_cls_b = batch_data
                print(f"Batch {i}:")
                print(f"  IDs: {ids_b}")
                print(f"  Images shape: {imgs_b.shape}")
                print(f"  Masks shape: {masks_b.shape}")
                print(f"  Image Cls shape: {img_cls_b.shape}")
                print(f"  Detections shape: {dets_b.shape}, num_dets: {dets_b.size(0)}")
                if dets_b.numel() > 0:
                    print(f"  Det batch indices: {torch.unique(dets_b[:,0]).tolist()}")
                if i == 1: # Print 2 batches
                    break
            print("✔️  DataLoader test successful.")
        except Exception as e:
            print(f"❌ DataLoader test FAILED: {e}")
            import traceback
            traceback.print_exc()

    elif len(train_ds) == 0:
        print("Training dataset is empty. Cannot perform further tests.")
    elif len(val_ds) == 0:
        print("Validation dataset is empty. Some tests might be skipped.")

