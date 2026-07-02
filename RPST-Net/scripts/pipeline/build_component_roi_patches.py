                     
                       

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

try:
    from tqdm import tqdm
except Exception:                    
    tqdm = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

                                                                                            
ROI_EXPAND_RATIOS = {
    "face": 0.35,
    "head": 0.30,
    "hand": 0.35,
    "cloth": 0.20,
    "figure": 0.12,
    "object": 0.22,
    "architecture": 0.12,
    "ornament": 0.18,
    "animal": 0.15,
    "plant": 0.15,
}

                                                      
DEFAULT_PATCH_SIZES = [256, 512, 1024]


YES_VALUES = {"yes", "y", "true", "1", "是", "YES", "True", "TRUE"}
ROI_ONLY_VALUES = {"roi_only", "partial", "ROI_ONLY"}
NO_VALUES = {"no", "n", "false", "0", "否", "NO", "False", "FALSE"}


def norm_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_token(x: object, default: str = "unknown") -> str:
    s = norm_str(x)
    if not s:
        return default
                                     
    bad = '<>:"/\\|?*\t\n\r '
    out = "".join("_" if ch in bad else ch for ch in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or default


def parse_bool_like(x: object) -> str:
    s = norm_str(x)
    if s in YES_VALUES:
        return "yes"
    if s in ROI_ONLY_VALUES:
        return "roi_only"
    if s in NO_VALUES:
        return "no"
    if s == "":
        return "unknown"
    return s.lower()


def md5_text(text: str, length: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def build_image_index(image_root: Path) -> Dict[str, List[Path]]:
    """Build an index from filename to paths under image_root."""
    index: Dict[str, List[Path]] = {}
    if not image_root.exists():
        return index
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.name, []).append(p)
    return index


def resolve_image_path(row: pd.Series, image_root: Path, image_index: Dict[str, List[Path]]) -> Optional[Path]:
    """Resolve image path using relative_path first, then current_folder/filename, then filename index."""
    rel = norm_str(row.get("relative_path", ""))
    if rel:
        p = image_root / Path(rel)
        if p.exists():
            return p
                                                                     
        p2 = image_root / Path(rel.replace("\\", "/"))
        if p2.exists():
            return p2

    folder = norm_str(row.get("current_folder", ""))
    filename = norm_str(row.get("filename", ""))
    if folder and filename:
        p = image_root / folder / filename
        if p.exists():
            return p

    if filename in image_index:
        paths = image_index[filename]
        if len(paths) == 1:
            return paths[0]
                                                                    
        if folder:
            for p in paths:
                if folder in p.parts:
                    return p
        return paths[0]
    return None


def load_image_rgb(path: Path) -> Image.Image:
    """Open image and convert to RGB while handling alpha on a white background."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        return bg.convert("RGB")
    if img.mode in {"LA", "P"}:
        return img.convert("RGBA").convert("RGB")
    return img.convert("RGB")


def median_color(img: Image.Image) -> Tuple[int, int, int]:
    small = img.resize((1, 1), Image.Resampling.BILINEAR)
    return tuple(int(v) for v in small.getpixel((0, 0)))


def clamp_box(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w, int(math.floor(xmin))))
    y1 = max(0, min(h, int(math.floor(ymin))))
    x2 = max(0, min(w, int(math.ceil(xmax))))
    y2 = max(0, min(h, int(math.ceil(ymax))))
    return x1, y1, x2, y2


def square_crop_with_padding(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    expand_ratio: float,
    pad_mode: str = "median",
) -> Tuple[Image.Image, Dict[str, int]]:
    """Crop a square region around bbox, allowing padding beyond image boundaries."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * expand_ratio)
    side = max(2, int(math.ceil(side)))

    sx1 = int(math.floor(cx - side / 2.0))
    sy1 = int(math.floor(cy - side / 2.0))
    sx2 = sx1 + side
    sy2 = sy1 + side

    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(w, sx2)
    iy2 = min(h, sy2)

    if ix2 <= ix1 or iy2 <= iy1:
                                                                     
        crop = img.crop((x1, y1, x2, y2))
        return crop, {
            "crop_xmin": x1,
            "crop_ymin": y1,
            "crop_xmax": x2,
            "crop_ymax": y2,
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 0,
            "square_side": max(crop.size),
        }

    crop_part = img.crop((ix1, iy1, ix2, iy2))
    if pad_mode == "white":
        bg_color = (255, 255, 255)
    elif pad_mode == "black":
        bg_color = (0, 0, 0)
    else:
        bg_color = median_color(crop_part)

    canvas = Image.new("RGB", (side, side), bg_color)
    off_x = ix1 - sx1
    off_y = iy1 - sy1
    canvas.paste(crop_part, (off_x, off_y))

    return canvas, {
        "crop_xmin": sx1,
        "crop_ymin": sy1,
        "crop_xmax": sx2,
        "crop_ymax": sy2,
        "pad_left": max(0, -sx1),
        "pad_top": max(0, -sy1),
        "pad_right": max(0, sx2 - w),
        "pad_bottom": max(0, sy2 - h),
        "square_side": side,
    }


def determine_crop_role(row: pd.Series) -> str:
    quality_type = norm_str(row.get("quality_type", "")).lower()
    use_for = norm_str(row.get("use_for", "")).lower()
    label_norm = norm_str(row.get("label_norm", row.get("label", ""))).lower()
    ref = parse_bool_like(row.get("is_reference_candidate", ""))
    clean = parse_bool_like(row.get("is_clean_candidate", ""))

    if quality_type == "outline" or "outline" in label_norm or "structural_guidance" in use_for:
        return "weak_structure"
    if quality_type == "partial" or "partial" in label_norm:
        return "partial_reference"
    if "damaged" in label_norm:
        return "damaged_target_only"
    if ref == "yes" or clean in {"yes", "roi_only"}:
        return "clean_reference"
    return "other"


def should_crop(row: pd.Series, include_outline: bool, include_partial: bool, include_other: bool) -> bool:
    role = determine_crop_role(row)
    if role == "weak_structure" and not include_outline:
        return False
    if role == "partial_reference" and not include_partial:
        return False
    if role == "other" and not include_other:
        return False
    return True


def make_patch_name(row: pd.Series, ann_index: int, patch_size: int) -> str:
    image_id = safe_token(row.get("image_id", "image"), "image")
    ann_id = safe_token(row.get("ann_id", f"ann{ann_index:06d}"), f"ann{ann_index:06d}")
    roi_type = safe_token(row.get("roi_type", "roi"), "roi")
    quality_type = safe_token(row.get("quality_type", "normal"), "normal")
    style = safe_token(row.get("final_style_group", "style"), "style")
    label = safe_token(row.get("label_norm", row.get("label", "label")), "label")
    short_hash = md5_text(f"{ann_id}|{label}|{patch_size}", 8)
    return f"{image_id}__{ann_id}__{roi_type}__{quality_type}__{style}__{short_hash}_{patch_size}.png"


def save_contact_sheet(
    patch_df: pd.DataFrame,
    output_dir: Path,
    roi_type: str,
    patch_size: int,
    max_images: int = 40,
    thumb: int = 128,
) -> Optional[Path]:
    sub = patch_df[(patch_df["roi_type"].astype(str) == roi_type) & (patch_df["patch_size"] == patch_size)]
    if sub.empty:
        return None
                                                          
    role_order = {"clean_reference": 0, "partial_reference": 1, "weak_structure": 2, "other": 3}
    sub = sub.copy()
    sub["_role_order"] = sub["crop_role"].map(role_order).fillna(9)
    sub = sub.sort_values(["_role_order", "final_style_group", "image_id"]).head(max_images)

    imgs = []
    labels = []
    for _, r in sub.iterrows():
        p = Path(r["patch_path"])
        if not p.exists():
            continue
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        except Exception:
            continue
        imgs.append(im)
        labels.append(f"{r.get('roi_type','')}|{r.get('quality_type','')}|{r.get('final_style_group','')}")

    if not imgs:
        return None

    cols = min(8, len(imgs))
    rows = math.ceil(len(imgs) / cols)
    caption_h = 22
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + caption_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + caption_h)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + thumb + 2), labels[i][:22], fill=(0, 0, 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"contact_sheet_{roi_type}_{patch_size}.jpg"
    sheet.save(out_path, quality=90)
    return out_path


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop official component patches from Tang tomb mural annotations.")
    parser.add_argument("--project_root", type=Path, default=Path(r"<PROJECT_ROOT>"))
    parser.add_argument("--input_csv", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=DEFAULT_PATCH_SIZES)
    parser.add_argument("--include_outline", action="store_true", default=True, help="Crop outline weak-structure boxes. Default: True.")
    parser.add_argument("--exclude_outline", action="store_true", help="Disable outline cropping.")
    parser.add_argument("--include_partial", action="store_true", default=True, help="Crop partial boxes. Default: True.")
    parser.add_argument("--exclude_partial", action="store_true", help="Disable partial cropping.")
    parser.add_argument("--include_other", action="store_true", help="Also crop rows that are neither reference nor structural guidance.")
    parser.add_argument("--min_bbox_size", type=int, default=8, help="Skip boxes whose width or height is smaller than this.")
    parser.add_argument("--pad_mode", choices=["median", "white", "black"], default="median")
    parser.add_argument("--image_format", choices=["png", "jpg"], default="png")
    parser.add_argument("--max_preview_per_roi", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None, help="Debug only: process first N annotations.")
    args = parser.parse_args()

    project_root = args.project_root
    input_csv = args.input_csv or (project_root / "splits" / "annotations_master_with_split.csv")
    image_root = args.image_root or (project_root / "curated_originals")
    output_dir = args.output_dir or (project_root / "reference_bank")

    include_outline = args.include_outline and not args.exclude_outline
    include_partial = args.include_partial and not args.exclude_partial

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    patches_dir = output_dir / "patches"
    metadata_dir = output_dir / "metadata"
    preview_dir = output_dir / "preview"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_flexible(input_csv)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    required = ["xmin", "ymin", "xmax", "ymax", "filename", "roi_type"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    print(f"[INFO] Input annotations: {len(df)}")
    print(f"[INFO] Input CSV: {input_csv}")
    print(f"[INFO] Image root: {image_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Patch sizes: {args.patch_sizes}")
    print(f"[INFO] include_outline={include_outline}; include_partial={include_partial}; include_other={args.include_other}")

    print("[INFO] Building image index...")
    image_index = build_image_index(image_root)
    print(f"[INFO] Indexed image files: {sum(len(v) for v in image_index.values())}")

    patch_records: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    missing_images: List[Dict[str, object]] = []
    image_cache: Dict[Path, Image.Image] = {}

    iterator = df.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(df), desc="Cropping component patches")

    for idx, row in iterator:
        ann_id = norm_str(row.get("ann_id", f"row_{idx}"))
        crop_role = determine_crop_role(row)
        if not should_crop(row, include_outline, include_partial, args.include_other):
            skipped.append({"ann_id": ann_id, "reason": f"excluded_by_role:{crop_role}", "label": row.get("label", "")})
            continue

        try:
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
        except Exception:
            skipped.append({"ann_id": ann_id, "reason": "invalid_bbox_values", "label": row.get("label", "")})
            continue

        if xmax <= xmin or ymax <= ymin:
            skipped.append({"ann_id": ann_id, "reason": "non_positive_bbox", "label": row.get("label", "")})
            continue

        if (xmax - xmin) < args.min_bbox_size or (ymax - ymin) < args.min_bbox_size:
            skipped.append({"ann_id": ann_id, "reason": "bbox_too_small", "label": row.get("label", "")})
            continue

        img_path = resolve_image_path(row, image_root, image_index)
        if img_path is None:
            missing_images.append({
                "ann_id": ann_id,
                "image_id": row.get("image_id", ""),
                "filename": row.get("filename", ""),
                "relative_path": row.get("relative_path", ""),
                "current_folder": row.get("current_folder", ""),
                "label": row.get("label", ""),
            })
            continue

        try:
            if img_path not in image_cache:
                image_cache[img_path] = load_image_rgb(img_path)
            img = image_cache[img_path]
        except (UnidentifiedImageError, OSError, ValueError) as e:
            skipped.append({"ann_id": ann_id, "reason": f"image_open_error:{e}", "image_path": str(img_path)})
            continue

        img_w, img_h = img.size
        bbox = clamp_box(xmin, ymin, xmax, ymax, img_w, img_h)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            skipped.append({"ann_id": ann_id, "reason": "bbox_outside_image", "label": row.get("label", "")})
            continue

        roi_type = norm_str(row.get("roi_type", "unknown")).lower() or "unknown"
        expand_ratio = ROI_EXPAND_RATIOS.get(roi_type, 0.20)
        square_crop, crop_info = square_crop_with_padding(img, bbox, expand_ratio=expand_ratio, pad_mode=args.pad_mode)

        for patch_size in args.patch_sizes:
            patch = square_crop.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            split = safe_token(row.get("split", "unsplit"), "unsplit")
            style_group = safe_token(row.get("final_style_group", "style_unknown"), "style_unknown")
            quality_type = safe_token(row.get("quality_type", "normal"), "normal")
            roi_token = safe_token(roi_type, "unknown")
            role_token = safe_token(crop_role, "role_unknown")

            out_subdir = patches_dir / str(patch_size) / split / roi_token / role_token
            out_subdir.mkdir(parents=True, exist_ok=True)
            patch_name = make_patch_name(row, idx, patch_size)
            if args.image_format == "jpg":
                patch_name = patch_name[:-4] + ".jpg"
            patch_path = out_subdir / patch_name

            if patch_path.exists() and not args.overwrite:
                                                       
                pass
            else:
                if args.image_format == "jpg":
                    patch.save(patch_path, quality=95)
                else:
                    patch.save(patch_path)

            rec = {
                "patch_id": patch_path.stem,
                "patch_path": str(patch_path),
                "patch_rel_path": str(patch_path.relative_to(output_dir)).replace("\\", "/"),
                "patch_size": patch_size,
                "crop_role": crop_role,
                "source_image_path": str(img_path),
                "source_image_rel_path": str(img_path.relative_to(image_root)).replace("\\", "/") if image_root in img_path.parents else str(img_path),
                "source_image_width": img_w,
                "source_image_height": img_h,
            }
                                              
            for col in [
                "ann_id", "image_id", "filename", "relative_path", "current_folder", "split",
                "tomb_id", "tomb_name", "final_style_group", "style_confidence",
                "composition_system", "scene_type", "quality_score_1to5", "preservation_quality", "damage_level",
                "is_reference_candidate", "is_clean_candidate", "label", "label_norm", "roi_type", "quality_type",
                "identity", "feature", "view", "gender", "action", "side", "style", "part", "color",
                "category", "name", "location", "type", "outline_scope", "outline_clue_type", "use_for",
                "parse_status", "parse_warning",
                "xmin", "ymin", "xmax", "ymax", "bbox_width", "bbox_height", "bbox_area", "bbox_ratio",
            ]:
                if col in row.index:
                    rec[col] = row.get(col)
            rec.update(crop_info)
            rec["expand_ratio"] = expand_ratio
            patch_records.append(rec)

    patch_df = pd.DataFrame(patch_records)
    skipped_df = pd.DataFrame(skipped)
    missing_df = pd.DataFrame(missing_images)

    patch_meta_path = metadata_dir / "roi_metadata.csv"
    skipped_path = metadata_dir / "skipped_annotations.csv"
    missing_path = metadata_dir / "missing_images.csv"

    patch_df.to_csv(patch_meta_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    missing_df.to_csv(missing_path, index=False, encoding="utf-8-sig")

                     
    summary_tables = []
    if not patch_df.empty:
        group_cols_sets = [
            ["patch_size"],
            ["split"],
            ["roi_type"],
            ["quality_type"],
            ["crop_role"],
            ["final_style_group"],
            ["patch_size", "split"],
            ["patch_size", "roi_type"],
            ["split", "roi_type", "quality_type"],
            ["final_style_group", "roi_type", "quality_type"],
        ]
        summary_xlsx_path = metadata_dir / "roi_summary.xlsx"
        with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
            for cols in group_cols_sets:
                valid_cols = [c for c in cols if c in patch_df.columns]
                if not valid_cols:
                    continue
                tab = patch_df.groupby(valid_cols, dropna=False).size().reset_index(name="patch_count")
                sheet_name = "by_" + "_".join(valid_cols)
                sheet_name = sheet_name[:31]
                tab.to_excel(writer, sheet_name=sheet_name, index=False)
                summary_tables.append((sheet_name, tab))

        simple_summary = patch_df.groupby(["patch_size", "roi_type", "crop_role"], dropna=False).size().reset_index(name="patch_count")
        simple_summary.to_csv(metadata_dir / "roi_summary.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(metadata_dir / "roi_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(input_csv),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "input_annotations": int(len(df)),
        "patch_rows": int(len(patch_df)),
        "unique_patch_files": int(patch_df["patch_path"].nunique()) if not patch_df.empty else 0,
        "skipped_annotations": int(len(skipped_df)),
        "missing_image_rows": int(len(missing_df)),
        "patch_sizes": args.patch_sizes,
        "include_outline": include_outline,
        "include_partial": include_partial,
        "include_other": bool(args.include_other),
    }
    if not patch_df.empty:
        summary["patch_count_by_role"] = patch_df.groupby("crop_role").size().to_dict()
        summary["patch_count_by_roi_type"] = patch_df.groupby("roi_type").size().to_dict()
        summary["patch_count_by_quality_type"] = patch_df.groupby("quality_type").size().to_dict()
    with open(metadata_dir / "roi_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

                                                              
    if not patch_df.empty:
        roi_types = ["face", "head", "figure", "hand", "cloth", "architecture", "ornament", "object"]
        for roi in roi_types:
            if roi not in set(patch_df["roi_type"].astype(str)):
                continue
            for ps in [s for s in args.patch_sizes if s in {256, 512}]:
                save_contact_sheet(patch_df, preview_dir, roi, ps, max_images=args.max_preview_per_roi)

    print("\n[DONE] 05 component patch cropping finished.")
    print(f"[DONE] Patch metadata: {patch_meta_path}")
    print(f"[DONE] Patch rows: {len(patch_df)}")
    print(f"[DONE] Skipped annotations: {len(skipped_df)} -> {skipped_path}")
    print(f"[DONE] Missing image rows: {len(missing_df)} -> {missing_path}")
    print(f"[DONE] Output patches: {patches_dir}")
    print(f"[DONE] Preview sheets: {preview_dir}")


if __name__ == "__main__":
    main()
