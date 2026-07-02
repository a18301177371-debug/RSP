                     
                       

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ENCODER_REGISTRY: Dict[str, Dict[str, str]] = {
    "dinov2_small": {"model_name": "facebook/dinov2-small", "tag": "dinov2_small", "family": "dinov2"},
    "dinov2_base": {"model_name": "facebook/dinov2-base", "tag": "dinov2_base", "family": "dinov2"},
    "dinov3_vits": {"model_name": "facebook/dinov3-vits16-pretrain-lvd1689m", "tag": "dinov3_vits", "family": "dinov3"},
    "dinov3_vitb": {"model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m", "tag": "dinov3_vitb", "family": "dinov3"},
}

STYLE_ORDER = ["S0", "S1", "S2", "S3"]
STYLE_ALIAS = {
    "s uk": "S_UK", "s_uk": "S_UK", "suk": "S_UK", "unknown": "S_UK", "": "S_UK",
    "s0": "S0", "s1": "S1", "s2": "S2", "s3": "S3",
}


def norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def norm_style(x: object) -> str:
    s = norm_text(x)
    key = s.lower().replace("-", "_").strip()
    return STYLE_ALIAS.get(key, s if s else "S_UK")


def style_score(a: object, b: object) -> float:
    """Soft style compatibility score in [0, 1]."""
    sa, sb = norm_style(a), norm_style(b)
    if sa == sb:
        return 1.0
    if "S_UK" in {sa, sb}:
        return 0.45
                                                 
    if {sa, sb} in [{"S1", "S2"}, {"S2", "S3"}]:
        return 0.82
    if sa in STYLE_ORDER and sb in STYLE_ORDER:
        d = abs(STYLE_ORDER.index(sa) - STYLE_ORDER.index(sb))
        if d == 1:
            return 0.62
        if d == 2:
            return 0.35
        return 0.20
    return 0.30


def infer_gender_from_row(row: pd.Series) -> str:
    g = norm_text(row.get("gender", "")).lower()
    if g in {"man", "male"}:
        return "man"
    if g in {"woman", "female"}:
        return "woman"
                                                                    
    text = " ".join(norm_text(row.get(c, "")).lower() for c in ["label", "identity", "sample_id", "patch_id"])
    if any(t in text for t in ["female", "woman", "maid", "palacelady", "lady"]):
        return "woman"
    if any(t in text for t in ["male", " man", "_man", "eunuch", "official"]):
        return "man"
    return "unknown"


def gender_score(qrow: pd.Series, rrow: pd.Series) -> float:
    qg, rg = infer_gender_from_row(qrow), infer_gender_from_row(rrow)
    if qg == "unknown" or rg == "unknown":
        return 0.55
    return 1.0 if qg == rg else 0.10


def role_score(role: object) -> float:
    r = norm_text(role).lower()
    if r == "clean_reference":
        return 1.0
    if r == "partial_reference":
        return 0.68
    if r == "weak_structure":
        return 0.08
    return 0.40


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def get_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def load_encoder(model_name: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    print(f"[INFO] Loading encoder: {model_name} on {device}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return processor, model


def model_forward_embeddings(model, inputs, pooling: str = "auto") -> np.ndarray:
    outputs = model(**inputs)
    if pooling == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feats = outputs.pooler_output
    elif pooling == "mean":
        feats = outputs.last_hidden_state.mean(dim=1)
    elif pooling == "cls":
        feats = outputs.last_hidden_state[:, 0]
    else:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0]
    return feats.detach().float().cpu().numpy()


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def extract_query_embeddings(
    image_paths: Sequence[Path],
    model_name: str,
    device: str,
    batch_size: int,
    pooling: str = "auto",
) -> np.ndarray:
    import torch
    processor, model = load_encoder(model_name, device)
    feats_all: List[np.ndarray] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Embedding query ROI images"):
        batch_paths = image_paths[start:start + batch_size]
        images = []
        for p in batch_paths:
            if not p.exists():
                raise FileNotFoundError(f"Query ROI image not found: {p}")
            images.append(load_image_rgb(p))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = model_forward_embeddings(model, inputs, pooling=pooling)
        feats_all.append(feats)
    emb = np.concatenate(feats_all, axis=0).astype("float32")
    return l2_normalize(emb).astype("float32")


def resolve_patch_path(row: pd.Series, reference_bank_dir: Path) -> Optional[Path]:
    vals = []
    for col in ["patch_path", "patch_rel_path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            vals.append(str(row[col]).strip())
    candidates: List[Path] = []
    for v in vals:
        p = Path(v)
        candidates.append(p if p.is_absolute() else reference_bank_dir / p)
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] if candidates else None


def copy_reference_image(src: Path, dst: Path, overwrite: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def load_thumb(path: Optional[Path], size: int = 160) -> Image.Image:
    try:
        if path is None:
            raise FileNotFoundError("None path")
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), (245, 245, 245))
            canvas.paste(im, ((size - im.width)//2, (size - im.height)//2))
            return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (235, 235, 235))
        d = ImageDraw.Draw(canvas)
        d.text((8, size//2 - 8), "MISSING", fill=(0, 0, 0))
        return canvas


def draw_text_block(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, width: int, font) -> None:
    x, y = xy
    lines: List[str] = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        line = ""
        for token in re.split(r"(\s+|[_/\\.-])", raw):
            cand = line + token
            try:
                length = draw.textlength(cand, font=font)
            except Exception:
                length = len(cand) * 7
            if length <= width or not line:
                line = cand
            else:
                lines.append(line)
                line = token.lstrip()
        if line:
            lines.append(line)
    for line in lines[:5]:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += 14


def make_contact_sheet(
    query_row: pd.Series,
    ref_rows: List[pd.Series],
    score_rows: List[Dict[str, float]],
    reference_bank_dir: Path,
    output_path: Path,
    thumb_size: int = 160,
) -> None:
    n = 1 + len(ref_rows)
    label_h = 88
    margin = 10
    gap = 8
    w = margin*2 + n*thumb_size + (n-1)*gap
    h = margin*2 + thumb_size + label_h
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
        font_bold = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    query_path = Path(str(query_row.get("query_roi_path", "")))
    rows = [query_row] + ref_rows
    for i, row in enumerate(rows):
        x = margin + i * (thumb_size + gap)
        y = margin
        if i == 0:
            thumb = load_thumb(query_path, size=thumb_size)
            title = "QUERY"
            subtitle = f"{query_row.get('sample_id','')}\n{query_row.get('roi_type','')}|{query_row.get('final_style_group','')}|{infer_gender_from_row(query_row)}"
        else:
            p = resolve_patch_path(row, reference_bank_dir)
            thumb = load_thumb(p, size=thumb_size)
            sc = score_rows[i-1]
            title = f"R{i:02d} final={sc['final_score']:.3f}"
            subtitle = (
                f"vis={sc['visual_score']:.3f} style={sc['style_score']:.2f} gender={sc['gender_score']:.2f}\n"
                f"{row.get('roi_type','')}|{row.get('final_style_group','')}|{infer_gender_from_row(row)}\n"
                f"{row.get('patch_id','')}"
            )
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x+thumb_size-1, y+thumb_size-1], outline=(80, 80, 80), width=1)
        draw.text((x, y+thumb_size+4), title, fill=(0, 0, 0), font=font_bold)
        draw_text_block(draw, (x, y+thumb_size+20), subtitle, thumb_size, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_candidate_mask(
    meta: pd.DataFrame,
    candidate_patch_size: int,
    candidate_roles: Sequence[str],
    candidate_split: str,
) -> np.ndarray:
    mask = np.ones(len(meta), dtype=bool)
    if "patch_size" in meta.columns and candidate_patch_size > 0:
        mask &= meta["patch_size"].astype(int).values == int(candidate_patch_size)
    if candidate_roles:
        mask &= meta["crop_role"].astype(str).isin(set(candidate_roles)).values
    if candidate_split.lower() != "all" and "split" in meta.columns:
        mask &= meta["split"].astype(str).str.lower().values == candidate_split.lower()
    return mask


def select_candidates_for_query(
    meta: pd.DataFrame,
    query_row: pd.Series,
    base_mask: np.ndarray,
    roi_policy: str,
    min_candidates: int,
    exclude_same_image: bool,
    exclude_same_tomb: bool,
) -> np.ndarray:
    mask = base_mask.copy()
    if exclude_same_image and "image_id" in meta.columns:
        mask &= meta["image_id"].astype(str).values != str(query_row.get("image_id", ""))
    if exclude_same_tomb and "tomb_id" in meta.columns:
        mask &= meta["tomb_id"].astype(str).values != str(query_row.get("tomb_id", ""))

    q_roi = str(query_row.get("roi_type", ""))
    if roi_policy == "required" and "roi_type" in meta.columns:
        roi_mask = meta["roi_type"].astype(str).values == q_roi
        strict = mask & roi_mask
        if strict.sum() >= min_candidates:
            mask = strict
    return np.where(mask)[0]


def enforce_image_diversity(rescored: List[Tuple], meta: pd.DataFrame, top_k: int, max_refs_per_image: int) -> List[Tuple]:
    if max_refs_per_image <= 0:
        return rescored[:top_k]
    selected: List[Tuple] = []
    counts: Dict[str, int] = defaultdict(int)
    for item in rescored:
        idx = int(item[1])
        image_id = str(meta.iloc[idx].get("image_id", ""))
        if counts[image_id] < max_refs_per_image:
            selected.append(item)
            counts[image_id] += 1
        if len(selected) >= top_k:
            break
                                                          
    if len(selected) < top_k:
        chosen = {int(x[1]) for x in selected}
        for item in rescored:
            if int(item[1]) not in chosen:
                selected.append(item)
                if len(selected) >= top_k:
                    break
    return selected[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k references for official inpainting benchmark samples.")
    parser.add_argument("--project_root", type=Path, default=Path(r"<PROJECT_ROOT>"))
    parser.add_argument("--encoder_tag", type=str, default="dinov2_small")
    parser.add_argument("--benchmark_dir", type=Path, default=None)
    parser.add_argument("--retrieval_dir", type=Path, default=None)
    parser.add_argument("--reference_bank_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--references_dir", type=Path, default=None)
    parser.add_argument("--encoder", choices=list(ENCODER_REGISTRY.keys()), default=None, help="Defaults to encoder_tag when possible.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pooling", choices=["auto", "pooler", "cls", "mean"], default="auto")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=800)
    parser.add_argument("--candidate_patch_size", type=int, default=256)
    parser.add_argument("--candidate_roles", nargs="+", default=["clean_reference", "partial_reference"])
    parser.add_argument("--candidate_split", type=str, default="train")
    parser.add_argument("--roi_policy", choices=["required", "soft"], default="required")
    parser.add_argument("--min_candidates", type=int, default=10)
    parser.add_argument("--exclude_same_image", action="store_true", default=True)
    parser.add_argument("--allow_same_image", action="store_true")
    parser.add_argument("--exclude_same_tomb", action="store_true")
    parser.add_argument("--style_weight", type=float, default=0.10)
    parser.add_argument("--gender_weight", type=float, default=0.04)
    parser.add_argument("--role_weight", type=float, default=0.02)
    parser.add_argument("--roi_weight", type=float, default=0.04)
    parser.add_argument("--max_refs_per_image", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--make_contact_sheets", action="store_true", default=True)
    parser.add_argument("--max_contact_sheets", type=int, default=60)
    parser.add_argument("--copy_reference_images", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root
    benchmark_dir = args.benchmark_dir or (project_root / "restoration_benchmark")
    retrieval_dir = args.retrieval_dir or (project_root / "retrieval" / args.encoder_tag)
    reference_bank_dir = args.reference_bank_dir or (project_root / "reference_bank")
    output_dir = args.output_dir or (benchmark_dir / "retrieval" / args.encoder_tag)
    references_dir = args.references_dir or (benchmark_dir / "references" / args.encoder_tag)

    master_path = benchmark_dir / "metadata" / "inpaint_benchmark_master.csv"
    meta_path = retrieval_dir / "index_metadata.csv"
    emb_path = retrieval_dir / "embeddings.npy"
    if not master_path.exists():
        raise FileNotFoundError(f"Benchmark master CSV not found: {master_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Reference index metadata not found: {meta_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Reference embeddings not found: {emb_path}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)

    samples = read_csv_flexible(master_path).reset_index(drop=True)
    if args.max_samples is not None:
        samples = samples.head(int(args.max_samples)).copy()
    ref_meta = read_csv_flexible(meta_path).reset_index(drop=True)
    ref_emb = np.load(emb_path).astype("float32")
    if len(ref_meta) != ref_emb.shape[0]:
        raise ValueError(f"Reference metadata rows {len(ref_meta)} != embeddings rows {ref_emb.shape[0]}")
    if "index_id" not in ref_meta.columns:
        ref_meta.insert(0, "index_id", np.arange(len(ref_meta), dtype=np.int64))

    query_paths = [Path(str(p)) for p in samples["query_roi_path"].tolist()]
    encoder_key = args.encoder or args.encoder_tag
    if encoder_key not in ENCODER_REGISTRY:
                                                                      
        encoder_key = "dinov2_small" if args.encoder_tag == "dinov2_small" else args.encoder_tag
    if encoder_key not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder {encoder_key}. Choose one of {list(ENCODER_REGISTRY)}")
    device = get_device(args.device)
    query_emb = extract_query_embeddings(
        image_paths=query_paths,
        model_name=ENCODER_REGISTRY[encoder_key]["model_name"],
        device=device,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )
    np.save(output_dir / "query_embeddings.npy", query_emb)
    samples.to_csv(output_dir / "query_embedding_metadata.csv", index=False, encoding="utf-8-sig")

    base_mask = build_candidate_mask(ref_meta, args.candidate_patch_size, args.candidate_roles, args.candidate_split)
    print(f"[INFO] Benchmark samples: {len(samples)}")
    print(f"[INFO] Reference metadata rows: {len(ref_meta)}")
    print(f"[INFO] Base candidate refs: {int(base_mask.sum())}")

    exclude_same_image = args.exclude_same_image and not args.allow_same_image
    long_rows: List[Dict[str, object]] = []
    warn_rows: List[Dict[str, object]] = []
    wide_records: List[Dict[str, object]] = []

    for qi, qrow in tqdm(samples.iterrows(), total=len(samples), desc="Retrieving references for benchmark"):
        cand_idxs = select_candidates_for_query(
            ref_meta,
            qrow,
            base_mask,
            roi_policy=args.roi_policy,
            min_candidates=args.min_candidates,
            exclude_same_image=exclude_same_image,
            exclude_same_tomb=args.exclude_same_tomb,
        )
        if len(cand_idxs) == 0:
            warn_rows.append({"sample_id": qrow.get("sample_id", ""), "warning": "no_candidates"})
            continue
        visual = ref_emb[cand_idxs] @ query_emb[int(qi)]
        n_pool = min(args.candidate_k, len(cand_idxs))
        if n_pool <= 0:
            warn_rows.append({"sample_id": qrow.get("sample_id", ""), "warning": "empty_candidate_pool"})
            continue
        top_local = np.argpartition(-visual, np.arange(n_pool))[:n_pool]
        pool_idxs = cand_idxs[top_local]
        pool_visual = visual[top_local]

        q_roi = str(qrow.get("roi_type", ""))
        rescored: List[Tuple[float, int, float, float, float, float, float]] = []
        for idx, vis in zip(pool_idxs, pool_visual):
            r = ref_meta.iloc[int(idx)]
            ss = style_score(qrow.get("final_style_group", ""), r.get("final_style_group", ""))
            gs = gender_score(qrow, r)
            rs = role_score(r.get("crop_role", ""))
            roi_s = 1.0 if str(r.get("roi_type", "")) == q_roi else 0.0
            final = float(vis) + args.style_weight * ss + args.gender_weight * gs + args.role_weight * rs + args.roi_weight * roi_s
            rescored.append((final, int(idx), float(vis), ss, gs, rs, roi_s))
        rescored.sort(key=lambda x: x[0], reverse=True)
        rescored = enforce_image_diversity(rescored, ref_meta, args.top_k, args.max_refs_per_image)

        sample_id = str(qrow.get("sample_id", f"sample_{qi:06d}"))
        split = str(qrow.get("split", "unknown"))
        sample_ref_dir = references_dir / split / sample_id
        if args.copy_reference_images:
            if sample_ref_dir.exists() and args.overwrite:
                shutil.rmtree(sample_ref_dir)
            sample_ref_dir.mkdir(parents=True, exist_ok=True)

        wide = qrow.to_dict()
        wide["topk_count"] = len(rescored)
        result_rows: List[pd.Series] = []
        result_scores: List[Dict[str, float]] = []
        for rank, (final, idx, vis, ss, gs, rs, roi_s) in enumerate(rescored, start=1):
            r = ref_meta.iloc[idx]
            src_path = resolve_patch_path(r, reference_bank_dir)
            ref_copy_path = ""
            if args.copy_reference_images and src_path is not None and src_path.exists():
                ext = src_path.suffix.lower() or ".png"
                ref_copy = sample_ref_dir / f"ref_{rank:02d}{ext}"
                copy_reference_image(src_path, ref_copy, overwrite=True)
                ref_copy_path = str(ref_copy)
            score_dict = {
                "final_score": final,
                "visual_score": vis,
                "style_score": ss,
                "gender_score": gs,
                "role_score": rs,
                "roi_score": roi_s,
            }
            result_rows.append(r)
            result_scores.append(score_dict)
            long_rows.append({
                "sample_id": sample_id,
                "split": split,
                "query_image_id": qrow.get("image_id", ""),
                "query_tomb_id": qrow.get("tomb_id", ""),
                "query_style_group": qrow.get("final_style_group", ""),
                "query_roi_type": qrow.get("roi_type", ""),
                "query_gender_raw": qrow.get("gender", ""),
                "query_gender_norm": infer_gender_from_row(qrow),
                "query_label": qrow.get("label", ""),
                "rank": rank,
                "ref_index_id": idx,
                "ref_patch_id": r.get("patch_id", ""),
                "ref_image_id": r.get("image_id", ""),
                "ref_tomb_id": r.get("tomb_id", ""),
                "ref_tomb_name": r.get("tomb_name", ""),
                "ref_split": r.get("split", ""),
                "ref_style_group": r.get("final_style_group", ""),
                "ref_roi_type": r.get("roi_type", ""),
                "ref_quality_type": r.get("quality_type", ""),
                "ref_crop_role": r.get("crop_role", ""),
                "ref_gender_raw": r.get("gender", ""),
                "ref_gender_norm": infer_gender_from_row(r),
                "ref_label": r.get("label", ""),
                "visual_score": vis,
                "style_score": ss,
                "gender_score": gs,
                "role_score": rs,
                "roi_score": roi_s,
                "final_score": final,
                "ref_patch_path": r.get("patch_path", ""),
                "ref_patch_rel_path": r.get("patch_rel_path", ""),
                "ref_copy_path": ref_copy_path,
            })
            wide[f"ref_{rank:02d}_path"] = ref_copy_path or str(src_path or "")
            wide[f"ref_{rank:02d}_patch_id"] = r.get("patch_id", "")
            wide[f"ref_{rank:02d}_roi_type"] = r.get("roi_type", "")
            wide[f"ref_{rank:02d}_style_group"] = r.get("final_style_group", "")
            wide[f"ref_{rank:02d}_gender"] = infer_gender_from_row(r)
            wide[f"ref_{rank:02d}_visual_score"] = vis
            wide[f"ref_{rank:02d}_final_score"] = final
        wide_records.append(wide)

        if args.make_contact_sheets and len(result_rows) > 0 and qi < args.max_contact_sheets:
            safe_sample = re.sub(r"[^A-Za-z0-9_\-]+", "_", sample_id)[:120]
            sheet_path = output_dir / "contact_sheets" / f"{safe_sample}_top{args.top_k}.jpg"
            make_contact_sheet(qrow, result_rows, result_scores, reference_bank_dir, sheet_path)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_records)
    warn_df = pd.DataFrame(warn_rows)
    long_df.to_csv(output_dir / "inpaint_retrieval_topk_long.csv", index=False, encoding="utf-8-sig")
    wide_df.to_csv(output_dir / "inpaint_benchmark_with_references.csv", index=False, encoding="utf-8-sig")
    warn_df.to_csv(output_dir / "retrieval_warnings.csv", index=False, encoding="utf-8-sig")

    summary = {
        "encoder_tag": args.encoder_tag,
        "encoder_key": encoder_key,
        "benchmark_samples": int(len(samples)),
        "reference_metadata_rows": int(len(ref_meta)),
        "reference_embedding_shape": list(map(int, ref_emb.shape)),
        "query_embedding_shape": list(map(int, query_emb.shape)),
        "base_candidate_refs": int(base_mask.sum()),
        "top_k": int(args.top_k),
        "topk_rows": int(len(long_df)),
        "queries_without_reference": int(len(warn_df)),
        "candidate_split": args.candidate_split,
        "candidate_roles": args.candidate_roles,
        "candidate_patch_size": int(args.candidate_patch_size),
        "roi_policy": args.roi_policy,
        "exclude_same_image": bool(exclude_same_image),
        "exclude_same_tomb": bool(args.exclude_same_tomb),
        "max_refs_per_image": int(args.max_refs_per_image),
        "counts_query_roi_type": Counter(samples.get("roi_type", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_query_style_group": Counter(samples.get("final_style_group", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_ref_roi_type": Counter(long_df.get("ref_roi_type", pd.Series(dtype=str)).astype(str)).most_common() if not long_df.empty else [],
        "counts_ref_style_group": Counter(long_df.get("ref_style_group", pd.Series(dtype=str)).astype(str)).most_common() if not long_df.empty else [],
        "counts_ref_gender_norm": Counter(long_df.get("ref_gender_norm", pd.Series(dtype=str)).astype(str)).most_common() if not long_df.empty else [],
    }
    with open(output_dir / "retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE] 08 benchmark reference retrieval finished.")
    print(f"[DONE] Benchmark samples: {len(samples)}")
    print(f"[DONE] Top-k rows: {len(long_df)}")
    print(f"[DONE] Queries without reference: {len(warn_df)}")
    print(f"[DONE] Output dir: {output_dir}")
    print(f"[DONE] References copied to: {references_dir}")


if __name__ == "__main__":
    main()
