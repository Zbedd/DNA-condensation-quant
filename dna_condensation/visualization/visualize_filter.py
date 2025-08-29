#!/usr/bin/env python3
"""
Filter Visualization Script

Compares segmentation results with transfection filter OFF vs ON, side-by-side,
and shows a composite of Nuclear (Red) and LSD1/Protein (Green) channels.

Layout per image (3 columns):
- Left:  Composite RGB (Nuclear→Red, LSD1→Green)
- Middle: Segmentation (filter = False)
- Right:  Segmentation (filter = True)

Fail-fast checks:
- input_source must be 'nd2'
- nd2_selection_settings.transfection_channel_index must exist and not be null
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.pipeline.batch_processor import main as run_batch_processor
from dna_condensation.pipeline.config import config as shared_config, Config
from dna_condensation.core.config_validator import ND2SelectionValidator


class FilterVisualizer:
    """Visualize composite + segmentation with transfection filter OFF vs ON."""

    def __init__(self, config_file=None, selection_override=None):
        # Prefer shared config; allow explicit file override if provided
        if config_file:
            self.config = Config(config_file)
        else:
            self.config = shared_config

        # Visualization settings (reuse same image_count config as visualize_images.py)
        viz_config = self.config.get("image_visualization", {}) or {}
        self.n_images = viz_config.get("n_images", 6)
        self.overlay_alpha = viz_config.get("overlay_alpha", 0.3)
        self.contour_color = viz_config.get("contour_color", "red")
        self.contour_width = viz_config.get("contour_width", 1.0)
        self.figure_size = viz_config.get("figure_size", [15, 10])
        self.colormap = viz_config.get("colormap", "gray")
        self.label_colormap = viz_config.get("label_colormap", "tab20")
        self.show_overlay = viz_config.get("show_segmentation_overlay", True)

        # ND2 selection handling
        self.selection_config = selection_override if selection_override is not None else self.config.get("nd2_selection_settings")
        self._validate_and_adjust_selection_config()

        # Source and ND2 cfg
        self.input_source = str(self.config.get("input_source", "unknown")).lower()
        self.nd2_cfg = self.config.get("nd2_selection_settings", {}) or {}

        # Fail fast: ensure ND2 and transfection channel present
        self._fail_fast_nd2_requirements()

        # Output directory
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fail_fast_nd2_requirements(self):
        prot_idx = self.nd2_cfg.get("transfection_channel_index", None)
        if self.input_source != "nd2" or prot_idx is None:
            # Explicit error per repo ground rules
            found_src = self.input_source or "<unset>"
            raise RuntimeError(
                "ERR_VIS_FILTER_ND2_REQUIRED: ND2 input with a configured transfection channel is required\n"
                f"Found: input_source={found_src}, transfection_channel_index={prot_idx}\n"
                "Action: Set input_source: nd2 and nd2_selection_settings.transfection_channel_index: <channel_index> in pipeline config."
            )

    def _validate_and_adjust_selection_config(self):
        # Limit and validate selection config for visualization
        if self.selection_config is None:
            self.selection_config = {
                "count": self.n_images,
                "seed": 42,
            }
            print(f"No ND2 selection config found. Using visualization default: {self.n_images} images")
            return

        validator = ND2SelectionValidator()
        is_valid, errors = validator.validate_selection_config(self.selection_config)
        if not is_valid:
            print("ND2 selection configuration errors:")
            for e in errors:
                print(f"  - {e}")
            print("Using default visualization settings instead.")
            self.selection_config = {
                "count": self.n_images,
                "seed": 42,
            }
            return

        # Cap count for visualization
        cfg_count = self.selection_config.get("count")
        if cfg_count is None or cfg_count > self.n_images:
            self.selection_config = dict(self.selection_config)
            self.selection_config["count"] = self.n_images
            print(f"Limiting display to {self.n_images} images for visualization")

        print(validator.get_validation_summary(self.selection_config))

    def _generate_timestamped_filename(self, base_name, extension="png"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"nd2_{base_name}_{ts}.{extension}"

    def _get_save_path(self, save_path, default_name):
        if save_path is None:
            filename = self._generate_timestamped_filename(default_name)
            return self.output_dir / filename
        return Path(save_path)

    def _normalize_image_for_display(self, image, channel_index=None):
        """Normalize a single 2D plane to [0,1]; select channel for 3D HxWxC arrays."""
        if image is None:
            return None
        if image.ndim == 3 and channel_index is not None:
            img = image[:, :, channel_index]
        else:
            img = image
        img = img.astype(np.float32, copy=False)
        vmax = float(np.max(img)) if img.size else 0.0
        vmin = float(np.min(img)) if img.size else 0.0
        if vmax <= vmin:
            return np.zeros_like(img, dtype=np.float32)
        return (img - vmin) / (vmax - vmin)

    def _add_segmentation_overlay(self, ax, mask, color="red", alpha=0.3, contour_width=1.0, show_overlay=True, show_contours=True):
        """Add filled overlay (binary or per-label color) and optional contours."""
        if mask is None:
            return None, None

        overlay_artist = None
        contour_artist = None

        if show_overlay:
            used_alpha = min(1.0, float(alpha) * 2.0)
            # Decide if labeled
            try:
                mask_max = int(np.max(mask))
            except Exception:
                mask_max = 0
            is_integer = np.issubdtype(mask.dtype, np.integer)
            is_labeled = is_integer and mask_max > 1

            if is_labeled:
                cmap = plt.get_cmap(self.label_colormap)
                n_colors = getattr(cmap, "N", 256)
                lut = np.zeros((mask_max + 1, 4), dtype=float)
                for lbl in range(1, mask_max + 1):
                    rgba = list(cmap((lbl % n_colors) / max(1, n_colors - 1)))
                    rgba[3] = used_alpha
                    lut[lbl] = rgba
                colored = lut[mask.astype(int)]
                overlay_artist = ax.imshow(colored, alpha=1.0)
            else:
                colored = np.zeros((*mask.shape, 4), dtype=float)
                seg = mask > 0
                if color == "red":
                    colored[seg] = [1, 0, 0, used_alpha]
                elif color == "blue":
                    colored[seg] = [0, 0, 1, used_alpha]
                elif color == "green":
                    colored[seg] = [0, 1, 0, used_alpha]
                elif color == "yellow":
                    colored[seg] = [1, 1, 0, used_alpha]
                else:
                    colored[seg] = [1, 0, 0, used_alpha]
                overlay_artist = ax.imshow(colored, alpha=1.0)

        if show_contours:
            try:
                boundaries = find_boundaries(mask > 0, mode="outer")
                contour_artist = ax.contour(boundaries, levels=[0.5], colors=color, linewidths=contour_width)
            except Exception:
                try:
                    cs = []
                    contours = find_contours(mask.astype(float), 0.5)
                    for c in contours:
                        line = ax.plot(c[:, 1], c[:, 0], color=color, linewidth=contour_width)
                        cs.extend(line)
                    contour_artist = cs
                except Exception:
                    pass

        return overlay_artist, contour_artist

    def _run_pipeline_with_transfection(self, force_filter: bool):
        """Run pipeline once with transfection filter forced OFF or ON, returning the result dict."""
        original_nd2 = self.config.get("nd2_selection_settings")
        # Prepare two overrides:
        # - keep same selection (count/seed), flip only transfection_channel_index
        nd2_override = dict(self.selection_config or {})
        prot_idx = (self.config.get("nd2_selection_settings") or {}).get("transfection_channel_index", None)

        try:
            # Attach required ND2 keys for batch_processor if selection override is partial
            if "transfection_channel_index" not in nd2_override:
                nd2_override["transfection_channel_index"] = prot_idx
            # Flip according to force
            nd2_override["transfection_channel_index"] = prot_idx if force_filter else None

            self.config.set("nd2_selection_settings", nd2_override)
            result = run_batch_processor(return_images=True, skip_validation=True)
            if result is None:
                raise RuntimeError("ERR_VIS_FILTER_NO_RESULT: pipeline returned no data in forced run")
            return result
        finally:
            # Restore original ND2 config
            if original_nd2 is not None:
                self.config.set("nd2_selection_settings", original_nd2)
            else:
                self.config.pop("nd2_selection_settings", None)

    def visualize(self, save_path=None, auto_save=True):
        """Render Composite | Seg(filter=False) | Seg(filter=True) with interactive toggles."""
        # Two consistent runs on the same ND2 selection
        off_res = self._run_pipeline_with_transfection(force_filter=False)
        on_res = self._run_pipeline_with_transfection(force_filter=True)

        # Validate same selection order (fail fast if mismatch)
        off_names = off_res.get("image_names", [])
        on_names = on_res.get("image_names", [])
        if len(off_names) != len(on_names) or any(a != b for a, b in zip(off_names, on_names)):
            raise RuntimeError(
                "ERR_VIS_FILTER_SELECTION_MISMATCH: image selections differ between runs\n"
                f"Off-run names: {off_names}\nOn-run names:  {on_names}\n"
                "Action: ensure both runs use identical nd2_selection_settings (count, seed, filters)."
            )

        raw_images = off_res["raw_images"]
        final_images = off_res["final_images"]  # base for both segmentation columns
        masks_off = off_res["masks"]
        masks_on = on_res["masks"]
        image_names = off_res["image_names"]
        nuc_channel = int(off_res["channel_index"])
        prot_channel = int((self.config.get("nd2_selection_settings") or {}).get("transfection_channel_index"))

        n_display = min(self.n_images, len(raw_images))
        if n_display == 0:
            raise RuntimeError("ERR_VIS_FILTER_NO_IMAGES: zero images available for visualization")

        # 3 columns: Composite | Seg(F) | Seg(T)
        n_rows = n_display
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Collect artists for toggling
        overlay_artists = []  # [row][col] -> overlay artist or None
        contour_artists = []  # [row][col] -> contour artist(s) or None

        for row in range(n_display):
            row_overlays = []
            row_contours = []

            # Column 0: Composite (Nuc→R, Prot→G)
            ax0 = axes[row, 0]
            ax0.set_xticks([])
            ax0.set_yticks([])
            base = raw_images[row]
            if base is None or base.ndim != 3 or prot_channel >= base.shape[2]:
                raise RuntimeError(
                    "ERR_VIS_FILTER_IMAGE_SHAPE: expected raw image with channels-last (H,W,C) and valid indices\n"
                    f"Found shape: {None if base is None else base.shape}, nuc={nuc_channel}, prot={prot_channel}"
                )
            nuc_img = self._normalize_image_for_display(base, nuc_channel)
            prot_img = self._normalize_image_for_display(base, prot_channel)
            rgb = np.zeros((nuc_img.shape[0], nuc_img.shape[1], 3), dtype=np.float32)
            rgb[..., 0] = nuc_img  # Red
            rgb[..., 1] = prot_img  # Green
            ax0.imshow(rgb)
            if row == 0:
                ax0.set_title("Composite (Nuc→R, LSD1→G)", fontsize=12, fontweight="bold")
            # Name label on left
            img_name = image_names[row] if row < len(image_names) else f"Image {row+1}"
            if len(img_name) > 25:
                img_name = img_name[:22] + "..."
            ax0.set_ylabel(img_name, fontsize=10, rotation=0, ha="right", va="center")
            row_overlays.append(None)
            row_contours.append(None)

            # Column 1: Segmentation (filter=False)
            ax1 = axes[row, 1]
            ax1.set_xticks([])
            ax1.set_yticks([])
            base1 = final_images[row]
            img1 = self._normalize_image_for_display(base1, nuc_channel)
            ax1.imshow(img1, cmap=self.colormap)
            o1, c1 = self._add_segmentation_overlay(
                ax1,
                masks_off[row] if row < len(masks_off) else None,
                color=self.contour_color,
                alpha=self.overlay_alpha,
                contour_width=self.contour_width,
                show_overlay=self.show_overlay,
                show_contours=True,
            )
            if row == 0:
                ax1.set_title("Segmentation (filter=False)", fontsize=12, fontweight="bold")
            row_overlays.append(o1)
            row_contours.append(c1)

            # Column 2: Segmentation (filter=True)
            ax2 = axes[row, 2]
            ax2.set_xticks([])
            ax2.set_yticks([])
            base2 = final_images[row]  # same base
            img2 = self._normalize_image_for_display(base2, nuc_channel)
            ax2.imshow(img2, cmap=self.colormap)
            o2, c2 = self._add_segmentation_overlay(
                ax2,
                masks_on[row] if row < len(masks_on) else None,
                color=self.contour_color,
                alpha=self.overlay_alpha,
                contour_width=self.contour_width,
                show_overlay=self.show_overlay,
                show_contours=True,
            )
            if row == 0:
                ax2.set_title("Segmentation (filter=True)", fontsize=12, fontweight="bold")
            row_overlays.append(o2)
            row_contours.append(c2)

            overlay_artists.append(row_overlays)
            contour_artists.append(row_contours)

        self._add_interactive_controls(fig, overlay_artists, contour_artists)

        plt.tight_layout()
        plt.subplots_adjust(left=0.15)

        if auto_save:
            out = self._get_save_path(None, "filter_comparison.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved visualization to: {out}")

        plt.show()
        return fig

    def _add_interactive_controls(self, fig, overlay_artists, contour_artists):
        """Add interactive CheckButtons for 'Fill Overlay' and 'Contours'."""
        ax_chk = plt.axes([0.01, 0.65, 0.14, 0.18])
        labels = ["Fill Overlay", "Contours"]
        visibility = [self.show_overlay, True]
        chk = CheckButtons(ax_chk, labels, visibility)

        def toggle(label):
            if label == "Fill Overlay":
                for row in overlay_artists:
                    for ov in row:
                        if ov is not None:
                            ov.set_visible(not ov.get_visible())
            elif label == "Contours":
                for row in contour_artists:
                    for ct in row:
                        if ct is None:
                            continue
                        if hasattr(ct, "collections"):
                            for coll in ct.collections:
                                coll.set_visible(not coll.get_visible())
                        elif hasattr(ct, "__iter__"):
                            for line in ct:
                                if hasattr(line, "get_visible"):
                                    line.set_visible(not line.get_visible())
                        else:
                            try:
                                ct.set_visible(not ct.get_visible())
                            except Exception:
                                pass
            plt.draw()

        chk.on_clicked(toggle)
        fig._chk = chk  # prevent GC


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize segmentation with transfection filter OFF vs ON")
    parser.add_argument("--n-images", type=int, default=None, help="Number of images to display (defaults to image_visualization.n_images)")
    parser.add_argument("--save", type=str, help="Optional output path (default: auto-timestamped in visualization/output)")
    parser.add_argument("--no-save", action="store_true", help="Do not save the figure to disk")
    parser.add_argument("--config", type=str, default=None, help="Optional explicit config file path")
    args = parser.parse_args()

    viz = FilterVisualizer(config_file=args.config)

    if args.n_images is not None:
        viz.n_images = int(args.n_images)

    fig = viz.visualize(save_path=args.save, auto_save=not args.no_save)
    return fig


if __name__ == "__main__":
    main()