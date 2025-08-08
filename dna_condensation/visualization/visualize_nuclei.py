"""
Visualize a panel of labeled nuclei with annotated metrics.

This module plots a parameterized number of nuclei sampled from a chosen group
and annotates each with selected per-nucleus metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import find_contours, regionprops

sns.set_style("white")


def _extract_nucleus_patch(image: np.ndarray, label_mask: np.ndarray, label_id: int, pad: int = 5):
    """Return cropped image patch and mask for a given nucleus label.
    image: 2D or 3D (H, W[, C]) numpy array (display channel already selected if 3D).
    label_mask: 2D labeled mask with nucleus IDs.
    label_id: integer label to extract.
    pad: pixels of padding around the bounding box.
    """
    if image.ndim == 3:
        # assume last dim is channels; visualize single channel was already selected by caller
        img2d = image
    else:
        img2d = image

    # Binary mask for this label
    m = (label_mask == label_id)
    if not np.any(m):
        return None, None

    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + 1 + pad)

    patch_img = img2d[y0:y1, x0:x1]
    patch_mask = m[y0:y1, x0:x1]
    return patch_img, patch_mask


def create_nuclei_panel(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    image_names: Sequence[str],
    features_df: pd.DataFrame,
    n_nuclei: int = 12,
    metrics: Optional[List[str]] = None,
    group: Optional[str] = None,
    group_column: str = "condition",
    colormap: str = "magma",
    channel_index: Optional[int] = None,
    save_path: Optional[Path] = None,
    random_state: Optional[int] = 42,
):
    """Create and optionally save a panel of nuclei with metric annotations.

    - images: list of collapsed/preprocessed images used for analysis
    - masks: list of labeled masks (same order as images)
    - image_names: list of ND2 filenames (must match features_df['image_name'])
    - features_df: dataframe with per-nucleus rows; needs 'image_name', 'nucleus_id' and metrics
    - n_nuclei: number of nuclei to plot
    - metrics: list of metric column names to annotate (defaults to homogeneity mean and intensity entropy)
    - group: optional group name to filter nuclei by (None means random from all)
    - group_column: name of the group column in features_df
    - colormap: matplotlib colormap for intensity visualization
    - channel_index: if images are 3D (H, W, C), which channel to display; if None and 3D, use last channel
    - save_path: where to save the panel image; if None, does not save
    - random_state: RNG seed for reproducible sampling
    """
    if metrics is None:
        metrics = ["glcm_homogeneity_mean", "intensity_entropy"]

    df = features_df.copy()
    if group:
        df = df[df[group_column] == group]

    if len(df) == 0:
        raise ValueError("No nuclei available for the requested selection.")

    # Random sample requested count
    rs = np.random.RandomState(random_state)
    sample_df = df.sample(n=min(n_nuclei, len(df)), random_state=rs)

    # Build figure grid
    n = len(sample_df)
    n_cols = min(6, max(3, int(np.ceil(np.sqrt(n)))))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.6))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Map image_name to index for quick lookup
    name_to_idx = {str(name): idx for idx, name in enumerate(image_names)}

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img_name = str(row.get("image_name"))
        label_id = int(row.get("nucleus_id", -1))
        if img_name not in name_to_idx or label_id <= 0:
            ax.axis("off")
            continue

        img_idx = name_to_idx[img_name]
        img = images[img_idx]
        msk = masks[img_idx]

        # Select display channel if needed
        if img is None or msk is None:
            ax.axis("off")
            continue
        disp = img
        if disp.ndim == 3:
            ch = channel_index if channel_index is not None else (disp.shape[2] - 1)
            disp = disp[:, :, ch]

        # Extract patch around nucleus
        patch_img, patch_mask = _extract_nucleus_patch(disp, msk, label_id, pad=6)
        if patch_img is None:
            ax.axis("off")
            continue

        # Normalize patch for better visualization
        p = patch_img.astype(np.float32)
        if np.nanmax(p) > np.nanmin(p):
            p = (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p) + 1e-8)

        # Mask background for focus
        masked = np.where(patch_mask, p, np.nan)

        im = ax.imshow(masked, cmap=colormap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw contour
        try:
            contours = find_contours(patch_mask.astype(float), 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color="white", linewidth=0.8)
        except Exception:
            pass

        # Title with group and ID
        grp = row.get(group_column, "?")
        ax.set_title(f"{grp} • ID {label_id}", fontsize=9)

        # Metrics annotation under image
        lines = []
        for metric in metrics:
            if metric in row and pd.notnull(row[metric]):
                try:
                    val = float(row[metric])
                    lines.append(f"{metric}: {val:.3f}")
                except Exception:
                    lines.append(f"{metric}: n/a")
            else:
                lines.append(f"{metric}: n/a")
        ax.text(0.5, -0.14, "\n".join(lines), ha="center", va="top", fontsize=8, transform=ax.transAxes)

    # Turn off any unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved nuclei panel to {save_path}")

    return fig
