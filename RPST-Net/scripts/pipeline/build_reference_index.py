                     
                       

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ENCODER_REGISTRY: Dict[str, Dict[str, str]] = {
    "dinov2_small": {
        "model_name": "facebook/dinov2-small",
        "tag": "dinov2_small",
        "family": "dinov2",
    },
    "dinov2_base": {
        "model_name": "facebook/dinov2-base",
        "tag": "dinov2_base",
        "family": "dinov2",
    },
    "dinov3_vits": {
        "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "tag": "dinov3_vits16_lvd1689m",
        "family": "dinov3",
    },
    "dinov3_vitb": {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "tag": "dinov3_vitb16_lvd1689m",
        "family": "dinov3",
    },
}


def sanitize_token(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9_\-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def resolve_patch_path(row: pd.Series, reference_bank_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for col in ["patch_path", "patch_rel_path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            p = Path(str(row[col]).strip())
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(reference_bank_dir / p)
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] if candidates else None


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def get_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        return "cpu"
    return requested


def load_encoder(model_name: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    print(f"[INFO] Loading encoder: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    if device == "cuda":
                                                                                      
        pass
    return processor, model


def model_forward_embeddings(model, inputs, pooling: str = "auto"):
    import torch

    outputs = model(**inputs)
    if pooling == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feats = outputs.pooler_output
    elif pooling == "cls":
        feats = outputs.last_hidden_state[:, 0]
    elif pooling == "mean":
        feats = outputs.last_hidden_state.mean(dim=1)
    else:
                                                               
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0]
    return feats.detach().float().cpu().numpy()


def extract_embeddings(
    df: pd.DataFrame,
    image_paths: Sequence[Path],
    model_name: str,
    device: str,
    batch_size: int,
    pooling: str,
    num_workers_hint: int = 0,
) -> np.ndarray:
    import torch

    processor, model = load_encoder(model_name, device)
    embeddings: List[np.ndarray] = []
    n = len(image_paths)
    print(f"[INFO] Extracting embeddings for {n} patches on {device}; batch_size={batch_size}")

    for start in tqdm(range(0, n, batch_size), desc="Embedding patches"):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        ok_indices = []
        for j, p in enumerate(batch_paths):
            try:
                images.append(load_image_rgb(p))
                ok_indices.append(j)
            except Exception as e:
                raise RuntimeError(f"Failed to load image: {p}; error={e}") from e

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = model_forward_embeddings(model, inputs, pooling=pooling)
        embeddings.append(feats)

    emb = np.concatenate(embeddings, axis=0).astype("float32")
    emb = l2_normalize(emb).astype("float32")
    return emb


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def save_role_indexes(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    role_col: str = "crop_role",
) -> Dict[str, int]:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    if role_col not in df.columns:
        return counts
    for role, idxs in df.groupby(role_col).groups.items():
        role_token = sanitize_token(role)
        mapping = np.array(sorted(list(idxs)), dtype="int64")
        if len(mapping) == 0:
            continue
        sub_emb = embeddings[mapping]
        index = build_faiss_index(sub_emb)
        faiss.write_index(index, str(out_dir / f"index_{role_token}.faiss"))
        np.save(out_dir / f"mapping_{role_token}.npy", mapping)
        counts[role_token] = int(len(mapping))
    return counts


def save_group_indexes(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    group_col: str,
    prefix: str,
) -> Dict[str, int]:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    if group_col not in df.columns:
        return counts
    for val, idxs in df.groupby(group_col).groups.items():
        token = sanitize_token(val)
        mapping = np.array(sorted(list(idxs)), dtype="int64")
        if len(mapping) == 0:
            continue
        sub_emb = embeddings[mapping]
        index = build_faiss_index(sub_emb)
        faiss.write_index(index, str(out_dir / f"index_{prefix}_{token}.faiss"))
        np.save(out_dir / f"mapping_{prefix}_{token}.npy", mapping)
        counts[token] = int(len(mapping))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official DINO retrieval embeddings and FAISS indexes.")
    parser.add_argument("--project_root", type=Path, default=Path(r"<PROJECT_ROOT>"))
    parser.add_argument("--metadata_csv", type=Path, default=None)
    parser.add_argument("--reference_bank_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--encoder", choices=sorted(ENCODER_REGISTRY.keys()), default="dinov2_small")
    parser.add_argument("--model_name", type=str, default=None, help="Override HF model name.")
    parser.add_argument("--encoder_tag", type=str, default=None, help="Override output subdirectory tag.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "pooler", "cls", "mean"])
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only_roles", nargs="+", default=None, help="Optional crop_role filter, e.g. clean_reference partial_reference")
    args = parser.parse_args()

    project_root = args.project_root
    reference_bank_dir = args.reference_bank_dir or (project_root / "reference_bank")
    metadata_csv = args.metadata_csv or (reference_bank_dir / "metadata" / "roi_metadata.csv")

    enc_cfg = ENCODER_REGISTRY[args.encoder]
    model_name = args.model_name or enc_cfg["model_name"]
    encoder_tag = args.encoder_tag or enc_cfg["tag"]
    output_dir = args.output_dir or (project_root / "retrieval" / encoder_tag)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = read_csv_flexible(metadata_csv)
    original_rows = len(df)
    if args.only_roles:
        role_set = set(args.only_roles)
        df = df[df["crop_role"].astype(str).isin(role_set)].copy()
        print(f"[INFO] Filtered roles {sorted(role_set)}: {original_rows} -> {len(df)} rows")
    if args.max_patches is not None:
        df = df.head(args.max_patches).copy()
        print(f"[INFO] Using max_patches={args.max_patches}; rows={len(df)}")

    if df.empty:
        raise ValueError("No rows to embed after filtering.")

    image_paths = [resolve_patch_path(row, reference_bank_dir) for _, row in df.iterrows()]
    missing = [(i, str(p)) for i, p in enumerate(image_paths) if p is None or not p.exists()]
    if missing:
        miss_path = output_dir / "missing_patch_paths.csv"
        pd.DataFrame(
            [{"row_index": i, "patch_path_candidate": p} for i, p in missing]
        ).to_csv(miss_path, index=False, encoding="utf-8-sig")
        raise FileNotFoundError(f"Missing patch image files: {len(missing)}. See {miss_path}")

    device = get_device(args.device)
    start_time = time.time()
    embeddings = extract_embeddings(
        df=df,
        image_paths=image_paths,
        model_name=model_name,
        device=device,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )
    elapsed = time.time() - start_time

    import faiss

    index_all = build_faiss_index(embeddings)
    emb_path = output_dir / "embeddings.npy"
    index_path = output_dir / "index_all.faiss"
    metadata_out = output_dir / "index_metadata.csv"
    info_path = output_dir / "index_info.json"

    np.save(emb_path, embeddings)
    faiss.write_index(index_all, str(index_path))

    df_out = df.copy().reset_index(drop=True)
    df_out.insert(0, "index_id", np.arange(len(df_out), dtype=np.int64))
    df_out.to_csv(metadata_out, index=False, encoding="utf-8-sig")

                                                                 
    role_counts = save_role_indexes(embeddings, df_out, output_dir / "role_indexes", role_col="crop_role")
    roi_counts = save_group_indexes(embeddings, df_out, output_dir / "roi_indexes", group_col="roi_type", prefix="roi")
    style_counts = save_group_indexes(embeddings, df_out, output_dir / "style_indexes", group_col="final_style_group", prefix="style")

    norms = np.linalg.norm(embeddings, axis=1)
    pd.DataFrame({
        "index_id": np.arange(len(norms)),
        "embedding_norm": norms,
        "patch_id": df_out.get("patch_id", pd.Series([""] * len(df_out))).values,
    }).head(1000).to_csv(output_dir / "preview_embedding_norms.csv", index=False, encoding="utf-8-sig")

    info = {
        "encoder": args.encoder,
        "model_name": model_name,
        "encoder_tag": encoder_tag,
        "device": device,
        "pooling": args.pooling,
        "batch_size": args.batch_size,
        "metadata_csv": str(metadata_csv),
        "reference_bank_dir": str(reference_bank_dir),
        "output_dir": str(output_dir),
        "rows_embedded": int(len(df_out)),
        "embedding_shape": list(map(int, embeddings.shape)),
        "index_type": "IndexFlatIP over L2-normalized embeddings (cosine similarity)",
        "elapsed_seconds": round(elapsed, 3),
        "counts_by_crop_role": Counter(df_out.get("crop_role", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_by_roi_type": Counter(df_out.get("roi_type", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_by_quality_type": Counter(df_out.get("quality_type", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_by_style_group": Counter(df_out.get("final_style_group", pd.Series(dtype=str)).astype(str)).most_common(),
        "role_index_counts": role_counts,
        "roi_index_counts": roi_counts,
        "style_index_counts": style_counts,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n[DONE] Retrieval index built.")
    print(f"[DONE] Rows embedded: {len(df_out)}")
    print(f"[DONE] Embeddings: {embeddings.shape} -> {emb_path}")
    print(f"[DONE] FAISS index: {index_path}")
    print(f"[DONE] Metadata: {metadata_out}")
    print(f"[DONE] Info: {info_path}")


if __name__ == "__main__":
    main()
