from pathlib import Path
import cv2  # cv2 needs to be imported
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader


class BTXRD(Dataset):
    def __init__(
        self, root_dir="btxrd_ready", split="train", img_size=640, transform=None
    ):  # transform is not used currently but kept for future
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.det_dir = self.root_dir / "labels_det"

        self.split = split
        self.img_size = img_size
        self.transform = transform  # Not used in this version

        # Load image classification data
        try:
            self.cls_rows = pd.read_csv(
                self.root_dir / "img_cls.csv", header=None, names=["filename", "class"]
            ).to_dict(orient="records")
            self.cls_lookup = {r["filename"]: r["class"] for r in self.cls_rows}
        except FileNotFoundError:
            print(
                f"Warning: img_cls.csv not found in {self.root_dir}. Image classification will not be available."
            )
            self.cls_lookup = {}

        self.items = []
        # Iterate through images and check for corresponding labels and masks
        for img_idx, img_path in enumerate(
            sorted(self.img_dir.glob("*.jpeg"))
        ):  # Assuming .jpeg, adjust if other extensions like .jpg, .png
            fname = img_path.name
            # Use .stem to get filename without extension for matching .txt and .png
            txt_path = self.det_dir / f"{img_path.stem}.txt"
            msk_path = self.mask_dir / f"{img_path.stem}.png"

            if txt_path.exists() and msk_path.exists() and fname in self.cls_lookup:
                class_id = self.cls_lookup[fname]
                # Store original index, image path, label path, mask path, and class_id
                self.items.append(
                    {
                        "id": img_idx,
                        "img_path": img_path,
                        "txt_path": txt_path,
                        "msk_path": msk_path,
                        "class_id": class_id,
                    }
                )
            # else:
            #     print(f"Skipping {fname}: Missing txt/msk or class_id. TXT: {txt_path.exists()}, MSK: {msk_path.exists()}, CLS: {fname in self.cls_lookup}")

        # Split data into training and validation sets
        n = len(self.items)
        if n == 0:
            print(
                f"Warning: No items loaded for split '{self.split}'. Check paths and file existence."
            )

        # Simple 80/20 split based on sorted order.
        # For more robust splitting, consider scikit-learn's train_test_split or pre-defined split files.
        split_idx = int(0.8 * n)
        if self.split == "train":
            self.items = self.items[:split_idx]
        elif self.split == "val":
            self.items = self.items[split_idx:]
        else:  # Allow for a 'all' split or other custom splits if needed
            pass  # self.items remains all items

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


if __name__ == "__main__":
    # Create a dummy directory structure for testing
    # You should replace "btxrd_ready_dummy" with your actual "btxrd_ready" path
    # or ensure the dummy structure matches your data for this test to run.
    dummy_root = Path("btxrd_ready_dummy")
    dummy_root.mkdir(exist_ok=True)
    (dummy_root / "images").mkdir(exist_ok=True)
    (dummy_root / "masks").mkdir(exist_ok=True)
    (dummy_root / "labels_det").mkdir(exist_ok=True)

    # Create dummy image files
    for i in range(1, 11):  # Create 10 dummy images
        img_name = f"IMG{i:07d}.jpeg"
        # Create a small black image
        dummy_img = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(dummy_root / "images" / img_name), dummy_img)

        # Create a dummy mask (half white, half black)
        dummy_mask = np.zeros((100, 120), dtype=np.uint8)
        dummy_mask[:, 60:] = 255
        cv2.imwrite(str(dummy_root / "masks" / f"IMG{i:07d}.png"), dummy_mask)

        # Create dummy detection labels
        with open(dummy_root / "labels_det" / f"IMG{i:07d}.txt", "w") as f:
            if i % 2 == 0:  # Add boxes for even numbered images
                f.write("0 0.5 0.5 0.2 0.2\n")  # class 0, center, 20% width/height
                f.write("1 0.3 0.3 0.1 0.15\n")  # class 1
            # else: no boxes for odd images

    # Create dummy classification CSV
    cls_data = []
    for i in range(1, 11):
        cls_data.append([f"IMG{i:07d}.jpeg", i % 2])  # class 0 or 1
    pd.DataFrame(cls_data).to_csv(dummy_root / "img_cls.csv", header=False, index=False)

    print(f"Dummy data created in {dummy_root.resolve()}")
    print("Testing BTXRD Dataset...")

    try:
        train_ds = BTXRD(root_dir=str(dummy_root), split="train", img_size=640)
        val_ds = BTXRD(root_dir=str(dummy_root), split="val", img_size=640)

        print(f"Train dataset size: {len(train_ds)}")
        print(f"Validation dataset size: {len(val_ds)}")

        if len(train_ds) > 0:
            train_loader = DataLoader(
                train_ds,
                batch_size=4,
                shuffle=True,
                num_workers=0,  # Set to 0 for easier debugging, >0 for parallel loading
                pin_memory=True,
                collate_fn=collate_fn,
            )

            print("\nIterating through one batch of train_loader...")
            ids_b, imgs_b, dets_b, masks_b, cls_b = next(iter(train_loader))

            print(f"Batch - IDs: {ids_b}")
            print(f"Batch - Images shape: {imgs_b.shape}, dtype: {imgs_b.dtype}")
            print(
                f"Batch - Detection boxes shape: {dets_b.shape}, dtype: {dets_b.dtype}"
            )
            if dets_b.numel() > 0:
                print("Batch - Sample detection box (first one if any):")
                from pprint import pprint

                pprint(dets_b[0].tolist())
            else:
                print("Batch - No detection boxes in this batch.")
            print(
                f"Batch - Masks shape: {masks_b.shape}, dtype: {masks_b.dtype}, unique values (first mask): {torch.unique(masks_b[0]) if masks_b.numel() > 0 else 'N/A'}"
            )
            print(f"Batch - Image class IDs: {cls_b.tolist()}, dtype: {cls_b.dtype}")

            # Test one item from dataset directly
            print("\nTesting __getitem__ for the first training sample...")
            original_idx_s, img_s, det_s, mask_s, cls_s = train_ds[0]
            print(f"Sample - Original Index: {original_idx_s}")
            print(f"Sample - Image shape: {img_s.shape}, dtype: {img_s.dtype}")
            print(
                f"Sample - Detection boxes shape: {det_s.shape}, dtype: {det_s.dtype}"
            )
            if det_s.numel() > 0:
                print("Sample - Detection boxes (first one if any):")
                pprint(det_s[0].tolist())
            print(
                f"Sample - Mask shape: {mask_s.shape}, dtype: {mask_s.dtype}, unique values: {torch.unique(mask_s)}"
            )
            print(f"Sample - Image class ID: {cls_s.item()}, dtype: {cls_s.dtype}")

        else:
            print("Train dataset is empty, skipping loader test.")

    except Exception as e:
        print(f"An error occurred during dataset testing: {e}")
        import traceback

        traceback.print_exc()

    # Clean up dummy directory (optional)
    # import shutil
    # shutil.rmtree(dummy_root)
    # print(f"Cleaned up dummy data directory: {dummy_root}")
