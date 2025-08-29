"""
Transfection-based nuclei filtering.

Applies a robust, local background-corrected test per nucleus on a specified
protein/transfection channel and retains only nuclei called as transfected.

Triggering condition (by design):
- Apply only when input_source == 'nd2' AND nd2_selection_settings.transfection_channel_index is not None.

Minimal, dependency-light implementation:
- Rolling-ball style background removal via skimage.restoration.rolling_ball
- Per-nucleus robust Z vs local annular background
- Benjamini–Hochberg FDR control and minimum log2 fold-change threshold

The implementation is batch-friendly and avoids side effects on images. It only
returns a new labels mask retaining passing nuclei with sequential relabeling.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.restoration import rolling_ball


# Print the selected filtering method only once per process
_PRINTED_METHOD: bool = False


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values for a 1D array of p-values.
    Returns array of same shape with monotone non-decreasing adjusted q-values.
    """
    if p.size == 0:
        return p.copy()
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # Enforce monotonicity
    q_sorted = q[order]
    q_monotone = np.minimum.accumulate(q_sorted[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q_monotone, 0.0, 1.0)
    return out


def _prep_protein_channel(image: np.ndarray, protein_channel_index: int, bg_radius: int) -> np.ndarray:
    """Extract protein channel as float and apply rolling-ball background removal.

    Works with 2D or 3D (H,W,C). Returns float64 array same HxW.
    """
    if image is None:
        return None  # type: ignore
    if image.ndim == 3:
        # Support both HWC and CHW
        c = int(protein_channel_index)
        if 0 <= c < image.shape[0] and image.shape[0] <= 16:
            chan = image[c, ...]
        elif 0 <= c < image.shape[2]:
            chan = image[:, :, c]
        else:
            raise ValueError(
                f"Protein channel index {c} out of range for image with shape {image.shape} "
                "(supported layouts: CHW, HWC)"
            )
    elif image.ndim == 2:
        chan = image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")

    # Convert to float and background-correct (rolling ball expects float)
    chan = chan.astype(np.float64, copy=False)
    try:
        bg = rolling_ball(chan, radius=int(bg_radius))
        corr = chan - bg
        corr[corr < 0] = 0
    except Exception:
        # Fallback: no background correction
        corr = chan
    return corr


def _compute_global_background(
    I: np.ndarray,
    labels: np.ndarray,
    *,
    exclusion_radius: int = 8,
    percentile: int = 20,
) -> Tuple[float, float, int]:
    """Global background from non-nuclear pixels excluding a margin around ALL nuclei.
    Returns (bg_median, bg_sigma_robust, n_bg_pixels). Fails fast if no pixels available.
    """
    all_nuc = labels > 0
    if exclusion_radius > 0:
        margin = binary_dilation(all_nuc, footprint=disk(int(exclusion_radius)))
        bg_mask = ~margin
    else:
        bg_mask = ~all_nuc
    vals = I[bg_mask]
    if vals.size == 0:
        raise RuntimeError("ERR_GLOBAL_BG_EMPTY: no pixels available for global background")
    # Robust low-tail trimming to avoid bright structures in 'background'
    thr = np.percentile(vals, int(percentile))
    vals = vals[vals <= thr]
    if vals.size == 0:
        raise RuntimeError("ERR_GLOBAL_BG_TRIM_EMPTY: percentile trimming removed all background pixels")
    bg_med = float(np.median(vals))
    bg_sig = float(1.4826 * _mad(vals))
    return bg_med, max(bg_sig, 1e-6), int(vals.size)


def _background_stats_global(
    I: np.ndarray,
    labels: np.ndarray,
    nucleus_id: int,
    *,
    global_bg: Optional[Tuple[float, float, int]] = None,
) -> Optional[Tuple[float, float, float, int, int]]:
    """Global background stats for a single nucleus.

    Returns (mu_in, mu_bg, sigma_bg, bg_px, nuc_px) or None if invalid nucleus.
    """
    if global_bg is None:
        raise RuntimeError("ERR_GLOBAL_BG_NOT_PROVIDED: global background must be computed once per image")
    nuc = (labels == nucleus_id)
    nuc_px = int(nuc.sum())
    if nuc_px <= 0:
        return None
    mu_in = float(np.median(I[nuc]))
    mu_bg, sigma_bg, bg_px = global_bg
    return mu_in, float(mu_bg), float(sigma_bg), int(bg_px), nuc_px


def _background_stats_annulus(
    I: np.ndarray,
    labels: np.ndarray,
    nucleus_id: int,
    *,
    r_in: int = 3,
    r_out: int = 10,
    min_ring_pixels: int = 200,
    ring_high_clip_percentile: Optional[float] = None,
) -> Optional[Tuple[float, float, float, int, int]]:
    """Local annulus background stats for a single nucleus.

    Returns (mu_in, mu_bg, sigma_bg, ring_px, nuc_px) or None if insufficient pixels.
    """
    nuc = (labels == nucleus_id)
    nuc_px = int(nuc.sum())
    if nuc_px <= 0:
        return None
    mu_in = float(np.median(I[nuc]))

    all_nuclei = labels > 0
    se_in = disk(int(max(1, r_in)))
    se_out = disk(int(max(1, r_out)))
    A_in = binary_dilation(nuc, footprint=se_in)
    A_out = binary_dilation(nuc, footprint=se_out)
    ring = A_out & (~A_in) & (~all_nuclei)
    ring_px = int(ring.sum())
    if ring_px < max(1, int(min_ring_pixels)):
        return None
    ring_vals = I[ring]
    if ring_high_clip_percentile is not None:
        hi = np.percentile(ring_vals, float(ring_high_clip_percentile))
        ring_vals = ring_vals[ring_vals <= hi]
        if ring_vals.size == 0:
            return None
    mu_bg = float(np.median(ring_vals))
    sigma_bg = float(1.4826 * _mad(ring_vals))
    return mu_in, mu_bg, max(sigma_bg, 1e-6), ring_px, nuc_px


def _compute_background_stats(
    I: np.ndarray,
    labels: np.ndarray,
    nucleus_id: int,
    *,
    model: str = "annulus",
    r_in: int = 3,
    r_out: int = 10,
    min_ring_pixels: int = 200,
    ring_high_clip_percentile: Optional[float] = None,
    global_bg: Optional[Tuple[float, float, int]] = None,
) -> Optional[Tuple[float, float, float, int, int]]:
    """Return (mu_in, mu_bg, sigma_bg, bg_px, nuc_px) per nucleus, or None if insufficient pixels.
    Delegates to specialized helpers based on model.
    - model='annulus': uses local annular background
    - model='global': uses precomputed global background
    """
    model = str(model).lower()
    if model == "global":
        return _background_stats_global(I, labels, nucleus_id, global_bg=global_bg)
    # Default: annulus
    return _background_stats_annulus(
        I,
        labels,
        nucleus_id,
        r_in=r_in,
        r_out=r_out,
        min_ring_pixels=min_ring_pixels,
        ring_high_clip_percentile=ring_high_clip_percentile,
    )


def filter_labels_by_transfection(
    image: np.ndarray,
    labels: np.ndarray,
    protein_channel_index: int,
    *,
    background_radius: int = 50,
    method: str = "annulus",
    global_exclusion_radius: int = 8,
    global_background_percentile: int = 20,
    ring_high_clip_percentile: Optional[float] = None,
    r_in: int = 3,
    r_out: int = 10,
    q_target: float = 0.05,
    delta_min: float = float(np.log2(1.5)),
    min_nucleus_pixels: int = 50,
    min_ring_pixels: int = 200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a relabeled mask keeping only nuclei called as transfected.

    Parameters mirror the attached plan defaults but are intentionally minimal.
    Returns (filtered_labels, stats_dict) where stats_dict contains summary counts.
    """
    if labels is None or image is None:
        return labels, {"kept": 0, "total": 0}
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array of nucleus IDs")

    I = _prep_protein_channel(image, protein_channel_index, background_radius)
    all_ids = np.unique(labels)
    all_ids = all_ids[all_ids > 0]
    total = int(all_ids.size)
    if total == 0:
        return labels.copy(), {"kept": 0, "total": 0}

    # Precompute global background once if needed (method controls background model)
    bg_model = str(method or "annulus").lower()
    global _PRINTED_METHOD
    if not _PRINTED_METHOD:
        try:
            print(f"[transfection_filter] method={bg_model}")
        finally:
            _PRINTED_METHOD = True
    precomputed_global: Optional[Tuple[float, float, int]] = None
    if bg_model == "global":
        precomputed_global = _compute_global_background(
            I, labels, exclusion_radius=int(global_exclusion_radius), percentile=int(global_background_percentile)
        )

    eps = 1e-6 * float(I.max() - I.min() if np.isfinite(I).all() else 1.0)
    Z, dlog2, keep_ids, sizes_ok = [], [], [], []

    for k in all_ids:
        stats = _compute_background_stats(
            I,
            labels,
            int(k),
            model=bg_model,
            r_in=int(r_in),
            r_out=int(r_out),
            min_ring_pixels=int(min_ring_pixels),
            ring_high_clip_percentile=ring_high_clip_percentile,
            global_bg=precomputed_global,
        )
        if stats is None:
            continue
        mu_in, mu_bg, sd_bg, ring_px, nuc_px = stats
        if nuc_px < min_nucleus_pixels or ring_px < min_ring_pixels:
            continue
        z = (mu_in - mu_bg) / (sd_bg + eps)
        d = float(np.log2((mu_in + eps) / (mu_bg + eps)))
        Z.append(z)
        dlog2.append(d)
        keep_ids.append(int(k))
        sizes_ok.append((nuc_px, ring_px))

    if len(keep_ids) == 0:
        return np.zeros_like(labels, dtype=labels.dtype), {"kept": 0, "total": total}

    Z = np.asarray(Z, dtype=float)
    dlog2 = np.asarray(dlog2, dtype=float)
    # one-sided p = 1 - Phi(Z)
    try:
        # Lightweight normal CDF via math.erf
        import math
        p = 0.5 * (1.0 - (1.0 + np.vectorize(lambda x: math.erf(x / np.sqrt(2.0)))(Z)))
        p = np.clip(p, 0.0, 1.0)
    except Exception:
        # Fallback to scipy if available
        try:
            from scipy.stats import norm  # type: ignore
            p = 1.0 - norm.cdf(Z)
        except Exception:
            # Worst-case: no test → all pass only by effect size
            p = np.ones_like(Z)

    qvals = _bh_fdr(p)
    calls = (qvals <= float(q_target)) & (dlog2 >= float(delta_min))

    passing_ids = [kid for kid, c in zip(keep_ids, calls) if bool(c)]
    filtered = np.zeros_like(labels, dtype=labels.dtype)
    new_id = 1
    for pid in passing_ids:
        filtered[labels == pid] = new_id
        new_id += 1

    return filtered, {
        "kept": int(new_id - 1),
        "total": int(total),
        "tested": int(len(keep_ids)),
        "params": {
            "background_radius": int(background_radius),
            "method": str(bg_model),
            "global_exclusion_radius": int(global_exclusion_radius),
            "global_background_percentile": int(global_background_percentile),
            "ring_high_clip_percentile": (None if ring_high_clip_percentile is None else float(ring_high_clip_percentile)),
            "r_in": int(r_in),
            "r_out": int(r_out),
            "q_target": float(q_target),
            "delta_min": float(delta_min),
            "min_nucleus_pixels": int(min_nucleus_pixels),
            "min_ring_pixels": int(min_ring_pixels),
        },
    }


def filter_labels_by_transfection_batch(
    images: List[np.ndarray],
    labels_list: List[np.ndarray],
    protein_channel_index: int,
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Apply transfection filter across a batch of images/masks.

    settings may include keys of filter_labels_by_transfection parameters.
    Missing keys fall back to defaults.
    """
    settings = settings or {}
    filtered_list: List[np.ndarray] = []
    stats_list: List[Dict[str, Any]] = []
    for img, lab in zip(images, labels_list):
        if lab is None or img is None:
            filtered_list.append(lab)
            stats_list.append({"kept": 0, "total": 0})
            continue
        # Select background model only from 'method' with default to 'annulus' if unspecified
        _model = str(settings.get("method", "annulus"))
        flt, st = filter_labels_by_transfection(
            img,
            lab,
            protein_channel_index,
            background_radius=int(settings.get("background_radius", 50)),
            method=_model,
            global_exclusion_radius=int(settings.get("global_exclusion_radius", 8)),
            global_background_percentile=int(settings.get("global_background_percentile", 20)),
            ring_high_clip_percentile=settings.get("ring_high_clip_percentile", None),
            r_in=int(settings.get("r_in", 3)),
            r_out=int(settings.get("r_out", 10)),
            q_target=float(settings.get("q_target", 0.05)),
            delta_min=float(settings.get("delta_min", float(np.log2(1.5)))),
            min_nucleus_pixels=int(settings.get("min_nucleus_pixels", 50)),
            min_ring_pixels=int(settings.get("min_ring_pixels", 200)),
        )
        filtered_list.append(flt)
        stats_list.append(st)
    return filtered_list, stats_list
