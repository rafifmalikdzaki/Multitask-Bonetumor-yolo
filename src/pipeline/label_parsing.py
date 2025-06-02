"""
Create detection txt, semantic masks, and image‑level csv
from BTXRD (Annotations/*.json, images/*.{png,jpg}) + dataset.xlsx.

usage:
    python preprocess_btxrd.py --src BTXRD --meta BTXRD/dataset.xlsx --dst btxrd_ready
"""

import argparse, json, os, cv2, pandas as pd, numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Sub‑type ➜ benign / malignant / normal bins (from Yao 2025 paper)
BENIGN_SUBTYPES = {
    "osteochondroma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
    "other bt",
}
MALIGN_SUBTYPES = {"osteosarcoma", "other mt"}

# CLS2ID = {"normal": 0, "B-tumor": 1, "M-tumor": 2}
CLS2ID = {"B-tumor": 0, "M-tumor": 1}
BOX2ID = {"B-tumor": 0, "M-tumor": 1}  # rectangles only
MASK_VAL = {"B-tumor": 1, "M-tumor": 2}  # pixel values
# ---------------------------------------------------------


def polygon_to_mask(poly, h, w, value):
    mask = np.zeros((h, w), np.uint8)
    pts = np.array(poly, np.int32)
    cv2.fillPoly(mask, [pts], value)
    return mask


def process_one(json_path: Path, out_det: Path, out_mask: Path, global_cls: str):
    js = json.load(json_path.open())
    h, w = js["imageHeight"], js["imageWidth"]

    full_mask = np.zeros((h, w), np.uint8)

    det_lines = []

    for sh in js["shapes"]:
        # lbl = sh["label"].strip()
        lbl = global_cls

        if sh["shape_type"] == "polygon" and lbl in MASK_VAL:
            full_mask = np.maximum(full_mask, polygon_to_mask(sh["points"], h, w, 1))

        elif sh["shape_type"] == "rectangle" and lbl in BOX2ID:
            (x1, y1), (x2, y2) = sh["points"]
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            det_lines.append(f"{BOX2ID[lbl]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    (out_det / f"{json_path.stem}.txt").write_text("\n".join(det_lines))

    cv2.imwrite(str(out_mask / f"{json_path.stem}.png"), full_mask * 127)

    return CLS2ID[global_cls]


# ---------------------------------------------------------
# def build_subtype_lookup(xlsx: Path) -> dict:
#     df = pd.read_excel(xlsx)
#     mapping = {Path(f).stem: cat.strip().lower() for f, cat in
#                zip(df['filename'], df['tumor_category'])}
#     return mapping


def build_type(xlsx: Path) -> dict:
    df = pd.read_excel(xlsx)
    mapping = {
        Path(f).stem: "B-tumor" if b else "M-tumor" if t else "normal"
        for f, t, b in zip(df["image_id"], df["tumor"], df["benign"])
    }
    return mapping


# ---------------------------------------------------------
def subtype_to_global(subtype: str) -> str:
    if subtype in BENIGN_SUBTYPES:
        return "B-tumor"
    if subtype in MALIGN_SUBTYPES:
        return "M-tumor"
    return "normal"


# ---------------------------------------------------------
def main():
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="BTXRD folder")
    ap.add_argument("--meta", required=True, help="dataset.xlsx")
    ap.add_argument("--dst", default="btxrd_ready", help="output dir")
    ap.add_argument("--img-ext", default=".jpeg", help="image extension")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "labels_det").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(parents=True, exist_ok=True)

    # subtype_map = build_subtype_lookup(args.meta)
    type_map = build_type(args.meta)
    img_cls_rows = []

    json_files = sorted((src / "Annotations").glob("*.json"))
    for js in tqdm(json_files):
        stem = js.stem
        # subtype = subtype_map.get(stem, "normal")
        type_cls = type_map.get(stem, "normal")
        # global_cls = subtype_to_global(subtype)

        class_id = process_one(
            js, out_det=dst / "labels_det", out_mask=dst / "masks", global_cls=type_cls
        )
        # link image
        img_src = src / "images" / f"{stem}{args.img_ext}"
        img_dst = dst / "images" / img_src.name
        if not img_dst.exists():
            os.link(img_src, img_dst)

        img_cls_rows.append([img_dst.name, class_id])

    # write image‑class CSV
    with open(dst / "img_cls.csv", "w", newline="") as f:
        for row in img_cls_rows:
            f.write(f"{row[0]},{row[1]}\n")

    print(f"Converted {len(json_files)} annotations →  {dst}")


if __name__ == "__main__":
    main()
