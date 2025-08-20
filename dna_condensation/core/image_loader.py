from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
import io
import zipfile
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from nd2reader import ND2Reader
from pathlib import Path
from dna_condensation.pipeline.config import config
from dna_condensation.core.file_selector import ND2FileSelector

# Optional heavy imports inside functions to avoid hard deps at import time

# Define global cache directory outside of Git tracking
BBBC022_CACHE_DIR = project_root / ".bbbc022_cache"

def get_nd2_objects(path: Optional[str] = None, selection_config: Optional[Dict[str, Any]] = None) -> List[ND2Reader]:
    """
    Return a list of ND2Reader objects for all .nd2 files in the specified path,
    with optional file selection and filtering.
    
    Args:
        path: Directory path containing .nd2 files. If None, uses config raw_nd2_path.
        selection_config: Optional configuration for file selection and filtering.
                         If None, uses config nd2_selection_settings or no filtering.
        
    Returns:
        List of ND2Reader objects for each selected .nd2 file.
        
    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError: If no .nd2 files are found in the path.
    """
    # Use provided path or fall back to config
    if path is None:
        path = config.get('raw_nd2_path')
    
    if not path:
        raise ValueError("No path provided and no raw_nd2_path configured")
    
    # Convert to Path object for easier handling
    nd2_path = Path(path)
    
    # Check if path exists
    if not nd2_path.exists():
        raise FileNotFoundError(f"Path does not exist: {nd2_path}")
    
    # Find all .nd2 files in the directory
    nd2_files = list(nd2_path.glob("*.nd2"))
    
    if not nd2_files:
        raise ValueError(f"No .nd2 files found in: {nd2_path}")
    
    print(f"Found {len(nd2_files)} ND2 files in directory")
    
    # Apply file selection if configured
    if selection_config is None:
        # Use selection config from main config if available
        selection_config = config.get('nd2_selection_settings')
    
    if selection_config:
        # Apply file selection
        file_selector = ND2FileSelector(selection_config)
        selected_files = file_selector.select_files(nd2_files)
        
        # Print selection summary
        print(file_selector.get_selection_summary(nd2_files, selected_files))
        
        nd2_files = selected_files
    else:
        print("No file selection applied - using all available files")
    
    # Create ND2Reader objects for each selected file
    nd2_objects = []
    for nd2_file in nd2_files:
        try:
            nd2_reader = ND2Reader(str(nd2_file))
            nd2_objects.append(nd2_reader)
            print(f"Loaded: {nd2_file.name}")
        except Exception as e:
            print(f"Warning: Could not load {nd2_file.name}: {e}")
    
    if not nd2_objects:
        raise ValueError("No valid .nd2 files could be loaded")
    
    print(f"Successfully loaded {len(nd2_objects)} ND2 files for processing")
    return nd2_objects


def load_bbbc022_images(
    count: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    use_cache: bool = True,
    group_mapping: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Load BBBC022 images by selecting specific image file names using the canonical metadata CSV.

        Contract:
        - Select groups by compound
            • Treatment: compound column contains any configured treatment term (case-insensitive substring).
            • Control: compound column is blank (after stripping quotes/whitespace).
        - Enforce even count and return half control and half treatment deterministically (no randomness).
        - Read BBBC022_v1_image.csv (cached) and load images by filename from plate/channel zips.

    Returns: (images, metadata)
      - images: list of numpy arrays
      - metadata: list of dicts with keys: image_name, plate, well, compound, condition
    """
    import pandas as pd
    import requests

    # Cache location
    if output_dir is None and use_cache:
        cache_dir = BBBC022_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(cache_dir)
    elif output_dir is None:
        output_dir = str(BBBC022_CACHE_DIR)

    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Metadata CSV
    csv_path = cache_dir / "BBBC022_v1_image.csv"
    url = "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv"
    if not csv_path.exists():
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"ERR_BBBC_METADATA_FETCH_FAILED: HTTP {resp.status_code}")
        csv_path.write_bytes(resp.content)

    # Read with robustness to columns quoting
    df = pd.read_csv(
        csv_path,
        engine="python",
        encoding="utf-8",
        sep=",",
        quotechar='"',
        quoting=3,
        on_bad_lines='skip'
    )
    df.columns = df.columns.str.strip().str.strip('"')

    well_col = 'Image_Metadata_CPD_WELL_POSITION'
    compound_col = 'Image_Metadata_SOURCE_COMPOUND_NAME'
    plate_col = 'Image_Metadata_PlateID'
    name_col_candidates = [
        'Image_FileName_Hoechst',
        'Image_FileName_DNA',
        'Image_FileName_OrigHoechst',
    ]
    missing = [c for c in (well_col, compound_col, plate_col) if c not in df.columns]
    if missing:
        raise RuntimeError(
            "ERR_MISSING_GROUPING_METADATA: required columns not found in BBBC022 metadata\n"
            f"Required: {well_col}, {compound_col}, {plate_col}\n"
            f"Missing:  {', '.join(missing)}"
        )
    name_col = None
    for c in name_col_candidates:
        if c in df.columns:
            name_col = c
            break
    if not name_col:
        raise RuntimeError(
            "ERR_MISSING_IMAGE_NAME_COLUMN: could not find an image filename column in metadata\n"
            f"Tried: {name_col_candidates}"
        )

    # Map metadata filename column to plate/channel zip index (w1..w5)
    channel_map = {
        # BBBC022 convention: w1 = Hoechst (DNA)
        'Image_FileName_OrigHoechst': 1,
        'Image_FileName_Hoechst': 1,
        'Image_FileName_DNA': 1,
        # Other channels
        'Image_FileName_OrigER': 2,
        'Image_FileName_OrigMito': 3,
        'Image_FileName_OrigPh_golgi': 4,
        'Image_FileName_OrigSyto': 5,
    }
    w_index = channel_map.get(name_col, 1)

    # Helper to detect blank compound
    def _is_blank(val):
        if val is None:
            return True
        try:
            if isinstance(val, float) and np.isnan(val):
                return True
        except Exception:
            pass
        if isinstance(val, str):
            s = val.strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            return s == ""
        return False

    if count % 2 != 0:
        raise ValueError(
            "ERR_COUNT_NOT_EVEN: even count required for balanced sampling\n"
            f"Requested: {count}\n"
            "Action: set an even 'count' in bbbc022_settings"
        )
    half = count // 2
    
    # Determine groups by compound
    bbbc_cfg = config.get("bbbc022_settings", {})
    treatment_terms = bbbc_cfg.get("treatment_compounds", []) or []
    if not treatment_terms:
        raise RuntimeError(
            "ERR_MISSING_TREATMENT_TERMS: config.bbbc022_settings.treatment_compounds must list treatment identifiers"
        )
    comp_series = df[compound_col].astype(str)
    treat_mask = np.zeros(len(df), dtype=bool)
    for term in treatment_terms:
        if isinstance(term, str) and term.strip():
            treat_mask |= comp_series.str.contains(term, case=False, na=False).to_numpy()
    ctrl_mask = df[compound_col].apply(_is_blank).to_numpy()
    df_treat = df[treat_mask].copy()
    df_ctrl = df[ctrl_mask].copy()

    # Deterministic selection: sort by image filename column, then take first N
    df_ctrl_sorted = df_ctrl.sort_values(by=name_col, kind='stable')
    df_treat_sorted = df_treat.sort_values(by=name_col, kind='stable')

    if len(df_ctrl_sorted) < half or len(df_treat_sorted) < half:
        raise ValueError(
            "ERR_INSUFFICIENT_GROUP_SAMPLES: cannot satisfy balanced selection from compound-based groups\n"
            f"Requested per-group: {half}\n"
            f"Available - treatment: {len(df_treat_sorted)}, control: {len(df_ctrl_sorted)}\n"
            "Action: reduce count or update bbbc022_settings.treatment_compounds"
        )

    sel_ctrl = df_ctrl_sorted.head(half).assign(__g='control')
    sel_treat = df_treat_sorted.head(half).assign(__g='treatment')
    sel = pd.concat([sel_treat, sel_ctrl], ignore_index=True)

    # Determine which plate/channel zips are needed for the selected channel only
    plates_needed = sorted({str(row[plate_col]) for _, row in sel.iterrows()})
    print(plates_needed)
    print(f"Using channel w{w_index} based on column {name_col}")

    print(f"Determining zip files needed for plates: {plates_needed}")

    zip_urls = []
    for plate in plates_needed:
        zip_urls.append(f"https://www.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_{plate}w{w_index}.zip")

    # Download needed zips if missing
    local_zips: List[Path] = []
    for u in zip_urls:
        name = u.split("/")[-1]
        p = raw_dir / name
        if not p.exists():
            r = requests.get(u, timeout=120, allow_redirects=True)
            if r.status_code != 200:
                # Non-200 likely means this plate/channel doesn't exist; skip it silently
                continue
            p.write_bytes(r.content)
            print(f"Downloaded {name}")
        local_zips.append(p)

    if not local_zips:
        raise RuntimeError("ERR_BBBC_ZIPS_UNAVAILABLE: no plate/channel zip files could be downloaded or found in cache")

    # Create index of zip members for fast lookup
    zip_members: Dict[str, Tuple[Path, zipfile.ZipInfo]] = {}
    for zp in local_zips:
        with zipfile.ZipFile(zp, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fname = Path(info.filename).name
                if fname not in zip_members:
                    zip_members[fname] = (zp, info)

    # Load selected images by filename
    images: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    for _, row in sel.iterrows():
        fname = str(row[name_col]).strip()
        if fname.startswith('"') and fname.endswith('"'):
            fname = fname[1:-1]
        if fname not in zip_members:
            raise RuntimeError(
                "ERR_MISSING_ZIP_MEMBERS: selected filename not found in any BBBC022 zip\n"
                f"Missing: {fname}"
            )
        zp, info = zip_members[fname]
        with zipfile.ZipFile(zp, 'r') as zf:
            with zf.open(info) as fh:
                data = fh.read()
        try:
            from skimage.io import imread
        except Exception as e:
            raise ImportError("scikit-image is required to read BBBC022 image files (pip install scikit-image)") from e
        img_bytes = io.BytesIO(data)
        img = imread(img_bytes)
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32)
        images.append(img)
        md = {
            'image_name': fname,
            'plate': str(row[plate_col]),
            'well': str(row[well_col]).strip('"').upper(),
            'compound': None if _is_blank(row[compound_col]) else str(row[compound_col]),
            'condition': row.get('__g', 'unknown'),
        }
        metadata.append(md)

    if len(images) != count:
        raise ValueError(
            "ERR_IMAGE_COUNT_MISMATCH: did not load requested number of images\n"
            f"Requested: {count}, Loaded: {len(images)}"
        )

    print(f"✓ Loaded {len(images)} BBBC022 images by filename")
    return images, metadata
