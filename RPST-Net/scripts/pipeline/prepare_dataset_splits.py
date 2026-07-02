                       

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


VALID_SPLITS = {"train", "val", "test"}


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Read CSV with a small encoding fallback."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_lower(x: object) -> str:
    return norm_text(x).lower()


def yes_no_uncertain(x: object) -> str:
    v = norm_lower(x)
    if v in {"yes", "y", "true", "1"}:
        return "yes"
    if v in {"no", "n", "false", "0"}:
        return "no"
    if v in {"roi_only", "roi-only", "roi only", "partial"}:
        return "roi_only"
    if v in {"uncertain", "unknown", "uk", ""}:
        return "uncertain"
    return v


def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Return train/val/test counts for a stratum.

    Rules are intentionally simple and conservative:
    - n == 1: all train
    - n == 2: 1 train, 0 val, 1 test
    - n >= 3: try to keep at least 1 val and 1 test.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    n_train = n - n_val - n_test

                                                                 
    if n_train < 1:
        deficit = 1 - n_train
                                         
        while deficit > 0 and (n_val > 1 or n_test > 1):
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            deficit -= 1
        n_train = n - n_val - n_test

    return n_train, n_val, n_test


def stratified_image_split(
    images: pd.DataFrame,
    stratify_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    out_rows: List[pd.DataFrame] = []

    tmp = images.copy()
    tmp[stratify_col] = tmp[stratify_col].fillna("unknown").astype(str).replace({"": "unknown"})

    for stratum, g in tmp.groupby(stratify_col, dropna=False, sort=True):
        g = g.sort_values(["tomb_id", "image_id", "filename"], na_position="last").copy()
        idxs = list(g.index)
        rng.shuffle(idxs)

        n_train, n_val, n_test = split_counts(len(idxs), train_ratio, val_ratio, test_ratio)
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train : n_train + n_val]
        test_idx = idxs[n_train + n_val : n_train + n_val + n_test]

        tmp.loc[train_idx, "split"] = "train"
        tmp.loc[val_idx, "split"] = "val"
        tmp.loc[test_idx, "split"] = "test"

    return tmp


def apply_manual_split(images: pd.DataFrame, manual_csv: Path) -> pd.DataFrame:
    manual = read_csv_smart(manual_csv)
    required = {"image_id", "split"}
    missing = required - set(manual.columns)
    if missing:
        raise ValueError(f"Manual split CSV is missing columns: {sorted(missing)}")

    manual = manual[["image_id", "split"]].copy()
    manual["image_id"] = manual["image_id"].astype(str).str.strip()
    manual["split"] = manual["split"].astype(str).str.strip().str.lower()

    invalid = sorted(set(manual["split"]) - VALID_SPLITS)
    if invalid:
        raise ValueError(f"Manual split CSV has invalid split values: {invalid}. Valid: {sorted(VALID_SPLITS)}")

    dup = manual[manual["image_id"].duplicated(keep=False)]
    if not dup.empty:
        raise ValueError(f"Manual split CSV has duplicated image_id values, examples: {dup['image_id'].head(10).tolist()}")

    out = images.merge(manual, on="image_id", how="left")
    missing_manual = out["split"].isna().sum()
    if missing_manual:
        raise ValueError(f"Manual split CSV does not assign split to {missing_manual} images.")
    return out


def list_join(values: Iterable[object]) -> str:
    vals = sorted({norm_text(v) for v in values if norm_text(v)})
    return ";".join(vals)


def build_image_table(ann: pd.DataFrame) -> pd.DataFrame:
    required = [
        "image_id",
        "filename",
        "relative_path",
        "current_folder",
        "tomb_id",
        "tomb_name",
        "final_style_group",
        "style_confidence",
        "composition_system",
        "scene_type",
        "quality_score_1to5",
        "preservation_quality",
        "damage_level",
        "is_reference_candidate",
        "is_clean_candidate",
        "image_width",
        "image_height",
    ]
    for c in required:
        if c not in ann.columns:
            ann[c] = ""

    base = ann.drop_duplicates("image_id")[required].copy()

    agg = ann.groupby("image_id").agg(
        num_boxes=("label", "size"),
        labels=("label_norm", list_join),
        roi_types=("roi_type", list_join),
        quality_types=("quality_type", list_join),
        num_face=("roi_type", lambda s: int((s == "face").sum())),
        num_head=("roi_type", lambda s: int((s == "head").sum())),
        num_figure=("roi_type", lambda s: int((s == "figure").sum())),
        num_hand=("roi_type", lambda s: int((s == "hand").sum())),
        num_cloth=("roi_type", lambda s: int((s == "cloth").sum())),
        num_outline=("quality_type", lambda s: int((s == "outline").sum())),
        num_partial=("quality_type", lambda s: int((s == "partial").sum())),
        num_normal=("quality_type", lambda s: int((s == "normal").sum())),
    ).reset_index()

    out = base.merge(agg, on="image_id", how="left")

    out["is_reference_candidate_norm"] = out["is_reference_candidate"].map(yes_no_uncertain)
    out["is_clean_candidate_norm"] = out["is_clean_candidate"].map(yes_no_uncertain)
    out["role_reference_candidate"] = out["is_reference_candidate_norm"].eq("yes")
    out["role_clean_full_candidate"] = out["is_clean_candidate_norm"].eq("yes")
    out["role_clean_roi_candidate"] = out["is_clean_candidate_norm"].isin(["yes", "roi_only"])

    return out


def make_warnings(images: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    warnings: List[Dict[str, object]] = []

    if images["image_id"].duplicated().any():
        dups = images.loc[images["image_id"].duplicated(keep=False), "image_id"].unique().tolist()
        warnings.append({"level": "error", "issue": "duplicated_image_id", "detail": ";".join(map(str, dups[:30]))})

    missing_split = images["split"].isna().sum() if "split" in images.columns else len(images)
    if missing_split:
        warnings.append({"level": "error", "issue": "images_without_split", "detail": int(missing_split)})

                                                                                      
    style_split = images.pivot_table(index="final_style_group", columns="split", values="image_id", aggfunc="count", fill_value=0)
    for style, row in style_split.iterrows():
        n_total = int(row.sum())
        if n_total >= 3:
            for sp in ["val", "test"]:
                if int(row.get(sp, 0)) == 0:
                    warnings.append({"level": "warning", "issue": "style_missing_holdout", "detail": f"{style} has 0 images in {sp}"})

                                                                     
    if "role_clean_full_candidate" in images.columns:
        clean_by_split = images.groupby("split")["role_clean_full_candidate"].sum()
        for sp in ["train", "val", "test"]:
            if int(clean_by_split.get(sp, 0)) == 0:
                warnings.append({"level": "warning", "issue": "no_full_clean_candidates", "detail": sp})

    return pd.DataFrame(warnings) if warnings else pd.DataFrame(columns=["level", "issue", "detail"])


def save_split_outputs(images: pd.DataFrame, ann: pd.DataFrame, output_dir: Path, summary: Dict[str, object]) -> None:
    ensure_dir(output_dir)

    images = images.copy()
    ann = ann.copy()

                                           
    split_map = images[["image_id", "split"]]
    ann = ann.merge(split_map, on="image_id", how="left")

    images.to_csv(output_dir / "split_images_v1.csv", index=False, encoding="utf-8-sig")
    ann.to_csv(output_dir / "annotations_master_with_split.csv", index=False, encoding="utf-8-sig")

    for sp in ["train", "val", "test"]:
        img_sp = images[images["split"] == sp].copy()
        ann_sp = ann[ann["split"] == sp].copy()

        img_sp.to_csv(output_dir / f"{sp}_images.csv", index=False, encoding="utf-8-sig")
        ann_sp.to_csv(output_dir / f"{sp}_annotations.csv", index=False, encoding="utf-8-sig")

        with open(output_dir / f"{sp}_image_ids.txt", "w", encoding="utf-8") as f:
            for image_id in img_sp["image_id"].astype(str).tolist():
                f.write(image_id + "\n")

                                                 
        img_sp[img_sp["role_reference_candidate"]].to_csv(
            output_dir / f"reference_candidates_{sp}.csv", index=False, encoding="utf-8-sig"
        )
        img_sp[img_sp["role_clean_full_candidate"]].to_csv(
            output_dir / f"clean_full_candidates_{sp}.csv", index=False, encoding="utf-8-sig"
        )
        img_sp[img_sp["role_clean_roi_candidate"]].to_csv(
            output_dir / f"clean_roi_candidates_{sp}.csv", index=False, encoding="utf-8-sig"
        )

                 
    images.pivot_table(index="final_style_group", columns="split", values="image_id", aggfunc="count", fill_value=0).to_csv(
        output_dir / "split_count_by_style.csv", encoding="utf-8-sig"
    )
    images.pivot_table(index="tomb_id", columns="split", values="image_id", aggfunc="count", fill_value=0).to_csv(
        output_dir / "split_count_by_tomb.csv", encoding="utf-8-sig"
    )
    ann.pivot_table(index="roi_type", columns="split", values="ann_id", aggfunc="count", fill_value=0).to_csv(
        output_dir / "split_count_by_roi_type.csv", encoding="utf-8-sig"
    )
    ann.pivot_table(
        index=["final_style_group", "roi_type", "quality_type"],
        columns="split",
        values="ann_id",
        aggfunc="count",
        fill_value=0,
    ).to_csv(output_dir / "split_count_by_style_roi_quality.csv", encoding="utf-8-sig")

                      
    role_rows = []
    for sp, g in images.groupby("split"):
        role_rows.append(
            {
                "split": sp,
                "num_images": int(len(g)),
                "reference_candidate_images": int(g["role_reference_candidate"].sum()),
                "clean_full_candidate_images": int(g["role_clean_full_candidate"].sum()),
                "clean_roi_candidate_images": int(g["role_clean_roi_candidate"].sum()),
                "outline_images": int((g["num_outline"] > 0).sum()),
                "face_images": int((g["num_face"] > 0).sum()),
                "head_images": int((g["num_head"] > 0).sum()),
            }
        )
    pd.DataFrame(role_rows).sort_values("split").to_csv(output_dir / "split_role_statistics.csv", index=False, encoding="utf-8-sig")

    warnings = make_warnings(images, ann)
    warnings.to_csv(output_dir / "split_warnings.csv", index=False, encoding="utf-8-sig")

    summary["num_warnings"] = int(len(warnings))
    summary["warnings"] = warnings.to_dict(orient="records")
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create official train/val/test splits for Tang tomb mural dataset.")
    parser.add_argument("--project_root", type=str, default=r"<PROJECT_ROOT>", help="Project root directory.")
    parser.add_argument(
        "--annotations_csv",
        type=str,
        default=None,
        help="Path to annotations_master.csv. Default: <project_root>/annotation_outputs/annotations_master.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <project_root>/splits",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--stratify_col", type=str, default="final_style_group", help="Image-level column used for stratified split.")
    parser.add_argument(
        "--manual_split_csv",
        type=str,
        default=None,
        help="Optional CSV with columns image_id,split. If provided, automatic split is skipped.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory. Existing files with the same names will be overwritten.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    annotations_csv = Path(args.annotations_csv) if args.annotations_csv else project_root / "annotation_outputs" / "annotations_master.csv"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "splits"

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum}")

    if not annotations_csv.exists():
        raise FileNotFoundError(f"annotations_csv not found: {annotations_csv}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}\n"
            f"Use --overwrite or choose a different --output_dir."
        )
    ensure_dir(output_dir)

    print(f"[INFO] Reading annotations: {annotations_csv}")
    ann = read_csv_smart(annotations_csv)
    print(f"[INFO] Annotation rows: {len(ann)}")

    if "image_id" not in ann.columns:
        raise ValueError("annotations_master.csv must contain image_id column.")
    if "ann_id" not in ann.columns:
        ann["ann_id"] = ann["image_id"].astype(str) + "__obj" + ann.groupby("image_id").cumcount().add(1).astype(str).str.zfill(4)

    images = build_image_table(ann)
    print(f"[INFO] Unique annotated images: {len(images)}")

    if args.stratify_col not in images.columns:
        raise ValueError(f"stratify_col not found in image table: {args.stratify_col}")

    if args.manual_split_csv:
        print(f"[INFO] Applying manual split: {args.manual_split_csv}")
        images = apply_manual_split(images, Path(args.manual_split_csv))
        split_mode = "manual"
    else:
        print(f"[INFO] Creating stratified image-level split by: {args.stratify_col}")
        images = stratified_image_split(
            images,
            stratify_col=args.stratify_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_mode = "stratified_image_level"

                                      
    images = images.sort_values(["split", "final_style_group", "tomb_id", "image_id"], na_position="last").reset_index(drop=True)

    split_counts_img = images["split"].value_counts().to_dict()
    print(f"[INFO] Image split counts: {split_counts_img}")

                                                
    if images["image_id"].duplicated().any():
        duplicated = images.loc[images["image_id"].duplicated(keep=False), "image_id"].unique().tolist()
        raise ValueError(f"Duplicated image_id after split: {duplicated[:20]}")
    if images["split"].isna().any():
        raise ValueError("Some images have no split assigned.")

    summary = {
        "script": "04_make_splits.py",
        "project_root": str(project_root),
        "annotations_csv": str(annotations_csv),
        "output_dir": str(output_dir),
        "split_mode": split_mode,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "stratify_col": args.stratify_col,
        "num_annotation_rows": int(len(ann)),
        "num_unique_images": int(len(images)),
        "image_split_counts": {k: int(v) for k, v in split_counts_img.items()},
        "style_split_counts": images.pivot_table(index="final_style_group", columns="split", values="image_id", aggfunc="count", fill_value=0).to_dict(),
        "notes": [
            "Split unit is image_id; all annotations from the same image stay in one split.",
            "is_real_damage_case and needs_damage_mask are ignored in this stage.",
            "Use clean_full_candidates_*.csv for full-canvas simulated-mask benchmark.",
            "Use clean_roi_candidates_*.csv for ROI-level clean candidate generation.",
            "Use reference_candidates_*.csv and annotations with use_for filters for reference bank construction.",
        ],
    }

    save_split_outputs(images, ann, output_dir, summary)

    print("\n[DONE] Official splits created.")
    print(f"[DONE] Output directory: {output_dir}")
    print("[DONE] Main files:")
    print(f"       {output_dir / 'split_images_v1.csv'}")
    print(f"       {output_dir / 'annotations_master_with_split.csv'}")
    print(f"       {output_dir / 'split_summary.json'}")
    print(f"       {output_dir / 'split_warnings.csv'}")


if __name__ == "__main__":
    main()
