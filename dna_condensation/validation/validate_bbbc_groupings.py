from pathlib import Path
import sys

# Ensure project root is on path (mirrors other modules)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from dna_condensation.pipeline.config import config
from dna_condensation.core.image_loader import load_bbbc022_images, BBBC022_CACHE_DIR


def validate_bbbc_groupings():
	"""
	Minimal validator for BBBC022 group selection.

	Steps:
	  1) Use config.bbbc022_settings.count (rounded down to even) to get planned image names via load_bbbc022_images(..., return_image_names_only=True).
	  2) Read cached BBBC022_v1_image.csv and extract relevant metadata for those names.
	  3) Build an np.array with columns: [image_name, plate, well, compound, condition].
	  4) Assertions:
		 a) control rows have null/blank compound.
		 b) treatment rows have compound matching any configured treatment term.
		 c) group sizes are roughly equal (abs(diff) <= 1).

	Returns:
	  np.ndarray of shape (N, 5) with dtype=object.
	"""

	# --- Load planned image names without downloading zips ---
	bcfg = config.get("bbbc022_settings", {}) or {}
	requested_count = int(bcfg.get("count", 20))
	# Loader enforces even counts; adjust here if odd
	count = requested_count if requested_count % 2 == 0 else requested_count - 1
	if count <= 0:
		raise ValueError("ERR_INVALID_COUNT: bbbc022_settings.count must be >= 2")

	names = load_bbbc022_images(
		count=count,
		output_dir=str(BBBC022_CACHE_DIR),
		use_cache=True,
		return_image_names_only=True,
	)

	if not isinstance(names, list) or not names:
		raise RuntimeError("ERR_NO_NAMES_RETURNED: loader did not return any image names")

	# --- Read metadata CSV (downloaded earlier by loader if missing) ---
	csv_path = Path(BBBC022_CACHE_DIR) / "BBBC022_v1_image.csv"
	if not csv_path.exists():
		raise FileNotFoundError(
			f"ERR_MISSING_METADATA_CSV: expected metadata at {csv_path} (run loader once to cache it)"
		)

	df = pd.read_csv(
		csv_path,
		engine="python",
		encoding="utf-8",
		sep=",",
		quotechar='"',
		quoting=3,
		on_bad_lines='skip',
	)
	df.columns = df.columns.str.strip().str.strip('"')

	# Identify filename column (match image_loader logic)
	name_col_candidates = [
		'Image_FileName_Hoechst',
		'Image_FileName_DNA',
		'Image_FileName_OrigHoechst',
	]
	well_col = 'Image_Metadata_CPD_WELL_POSITION'
	compound_col = 'Image_Metadata_SOURCE_COMPOUND_NAME'
	plate_col = 'Image_Metadata_PlateID'

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

	# Helper for blank detection and consistent filename cleaning
	def _clean_name(s: str) -> str:
		s = str(s).strip()
		if s.startswith('"') and s.endswith('"'):
			s = s[1:-1]
		return s

	def _is_blank(val) -> bool:
		if val is None:
			return True
		try:
			if isinstance(val, float) and np.isnan(val):
				return True
		except Exception:
			pass
		if isinstance(val, str):
			t = val.strip()
			if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
				t = t[1:-1].strip()
			return t == ""
		return False

	# Build a lookup from cleaned filename to first matching row
	df = df.copy()
	df["__clean_name"] = df[name_col].map(_clean_name)
	meta_by_name = {row["__clean_name"]: row for _, row in df.iterrows()}

	# Treatment terms from config
	treatment_terms = [t for t in (bcfg.get("treatment_compounds", []) or []) if isinstance(t, str) and t.strip()]
	if not treatment_terms:
		raise RuntimeError(
			"ERR_MISSING_TREATMENT_TERMS: config.bbbc022_settings.treatment_compounds must list treatment identifiers"
		)

	def _is_treatment(compound: str) -> bool:
		if compound is None:
			return False
		s = str(compound)
		return any(term.lower() in s.lower() for term in treatment_terms)

	# Assemble rows and conditions
	rows = []
	for nm in names:
		row = meta_by_name.get(_clean_name(nm))
		if row is None:
			raise RuntimeError(
				"ERR_SELECTED_NAME_NOT_IN_METADATA: planned filename missing from metadata CSV\n"
				f"Missing: {nm}"
			)
		compound = None if _is_blank(row.get(compound_col)) else str(row.get(compound_col))
		condition = 'control' if compound is None else ('treatment' if _is_treatment(compound) else 'other')
		rows.append([
			_clean_name(nm),
			str(row.get(plate_col)),
			str(row.get(well_col)).strip('"').upper(),
			compound,
			condition,
		])

	arr = np.array(rows, dtype=object)

	# --- Tests ---
	# a) control rows have null/blank compound
	ctrl_mask = arr[:, 4] == 'control'
	if ctrl_mask.any():
		ctrl_compounds = arr[ctrl_mask, 3]
		assert all(c is None or str(c).strip() == '' for c in ctrl_compounds), (
			"ERR_CONTROL_COMPOUND_NONNULL: one or more control rows have non-null compound"
		)

	# b) treatment rows have one of specified compounds
	treat_mask = arr[:, 4] == 'treatment'
	if treat_mask.any():
		treat_compounds = arr[treat_mask, 3]
		assert all(_is_treatment(c) for c in treat_compounds), (
			"ERR_TREATMENT_COMPOUND_MISMATCH: one or more treatment rows lack a configured treatment compound"
		)

	# c) group sizes roughly equal (abs diff <= 1)
	n_ctrl = int(ctrl_mask.sum())
	n_treat = int(treat_mask.sum())
	assert abs(n_ctrl - n_treat) <= 1, (
		f"ERR_GROUP_IMBALANCE: control={n_ctrl}, treatment={n_treat} (allowed diff <= 1)"
	)

	print(f"Validation passed: control={n_ctrl}, treatment={n_treat}, total={len(arr)}")
	return arr


if __name__ == "__main__":
	result = validate_bbbc_groupings()
	# Show a small preview by group
	print("First 5 treatment rows:")
	treat_rows = result[result[:, 4] == 'treatment'][:5]
	for r in treat_rows:
		print(r)

	print("First 5 control rows:")
	ctrl_rows = result[result[:, 4] == 'control'][:5]
	for r in ctrl_rows:
		print(r)
 