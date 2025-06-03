from dataset_btxrdv2 import BTXRD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# (1) First, make sure your BTXRD class is in scope (e.g. by having the class definition
#     in the same file or importing it). Then instantiate it:
ds = BTXRD(root_dir="btxrd_ready", split="val", img_size=640)

# (2) Choose an index to inspect (e.g. idx = 0). This will return:
#     original_id, img_t, det_boxes, mask_t, img_cls
idx = 20
original_id, img_t, det_boxes, mask_t, img_cls = ds[idx]

# (3) Convert the image tensor back to a uint8 H×W×3 array:
#     - img_t is in [3, H, W], with values in [0,1]
#     - multiply by 255 and cast to uint8; also permute to (H, W, 3)
img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

# (4) Prepare a Matplotlib figure and show the image.
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.imshow(img_np)
ax.axis("off")

# (5) Draw each bounding box onto the same axis:
#     det_boxes has shape (N, 6), columns = [batch_idx, class_label, xc_norm, yc_norm, w_norm, h_norm].
#     Since this is a single‐image sample, batch_idx will be 0. We un-normalize by img_size.
img_size = img_np.shape[0]  # should be 640 if you used img_size=640 in BTXRD

for box in det_boxes:
    _, cls_label, xc, yc, w, h = box.tolist()
    # Compute top-left corner (x1, y1) in pixels:
    x1 = (xc - w / 2) * img_size
    y1 = (yc - h / 2) * img_size
    width = w * img_size
    height = h * img_size

    # Create a rectangle patch (edge-only, no fill):
    rect = Rectangle(
        (x1, y1),
        width,
        height,
        fill=False,
        linewidth=2
    )
    ax.add_patch(rect)

plt.show()
