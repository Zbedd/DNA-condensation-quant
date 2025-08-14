"""
File selection utilities for ND2 files with flexible filtering and balanced sampling.

This module provides the ND2FileSelector class for intelligent selection of ND2 files
based on content filters, pattern matching, and balanced random sampling strategies.
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class ND2FileSelector:
    """
    Handles selection and filtering of ND2 files based on configuration.
    
    Implements a multi-stage pipeline:
    1. File discovery and metadata parsing
    2. Content-based filtering (OR logic for group filters)
    3. Pattern-based filtering (include/exclude)
    4. Balanced random sampling with specified count
    
    The selector always uses balanced random sampling: if count is specified,
    it tries to get equal representation across detected groups, with random
    selection within each group (seeded for reproducibility).
    """
    
    def __init__(self, selection_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the file selector with configuration.
        
        Args:
            selection_config: Configuration dictionary with selection parameters.
                            If None, no filtering is applied (select all files).
        """
        self.config = selection_config or {}
        self.seed = self.config.get('seed', 42)
        
    def select_files(self, nd2_file_paths: List[Path]) -> List[Path]:
        """
        Main orchestration method for file selection pipeline.
        
        Args:
            nd2_file_paths: List of Path objects for all available ND2 files
            
        Returns:
            List of Path objects for selected ND2 files
        """
        if not self.config:
            # No configuration means select all files
            return nd2_file_paths
            
        print(f"Starting file selection from {len(nd2_file_paths)} available ND2 files")
        
        # Stage 1: Parse metadata from filenames
        files_with_metadata = []
        for file_path in nd2_file_paths:
            metadata = self._parse_filename_metadata(file_path.name)
            files_with_metadata.append((file_path, metadata))
        
        # Stage 2: Apply content filters (OR logic)
        filtered_files = self._apply_content_filters(files_with_metadata)
        print(f"After content filtering: {len(filtered_files)} files")
        
        # Stage 3: Apply pattern filters
        file_paths_only = [file_path for file_path, _ in filtered_files]
        pattern_filtered = self._apply_pattern_filters(file_paths_only)
        print(f"After pattern filtering: {len(pattern_filtered)} files")
        
        # Stage 4: Apply count limit with balanced random sampling
        final_selection = self._apply_balanced_count_limit(pattern_filtered)
        print(f"Final selection: {len(final_selection)} files")
        
        return final_selection
    
    def _parse_filename_metadata(self, filename: str) -> Dict[str, str]:
        """
        Extract ID and name from ND2 filename using the expected pattern.
        
        Expected pattern: '48hr_[LSD1_ID]_[LSD1_name]_[additional_info].nd2'
        Example: '48hr_dk16_wtLSD1_well1_20x001.nd2'
        
        Args:
            filename: The ND2 filename to parse
            
        Returns:
            Dictionary with 'id', 'name', and 'full_name' keys
        """
        # Regex pattern to extract ID and name from filename
        # Pattern: 48hr_([^_]+)_([^_]+)_.*\.nd2
        pattern = r'48hr_([^_]+)_([^_]+)_.*\.nd2'
        match = re.match(pattern, filename)
        
        if match:
            lsd1_id = match.group(1)
            lsd1_name = match.group(2)
            return {
                'id': lsd1_id,
                'name': lsd1_name,
                'full_name': filename
            }
        else:
            # Fallback for files that don't match expected pattern
            return {
                'id': '',
                'name': '',
                'full_name': filename
            }
    
    def _apply_content_filters(self, files_with_metadata: List[Tuple[Path, Dict[str, str]]]) -> List[Tuple[Path, Dict[str, str]]]:
        """
        Filter files based on ID and name content using OR logic.
        
        OR Logic: A file is included if:
        - It contains ANY of the specified IDs, OR
        - It contains ANY of the specified names
        - If both ids and names are specified, file matches if it matches either condition
        
        Args:
            files_with_metadata: List of (file_path, metadata_dict) tuples
            
        Returns:
            Filtered list of (file_path, metadata_dict) tuples
        """
        group_filters = self.config.get('group_filters', {})
        if not group_filters:
            return files_with_metadata
            
        target_ids = group_filters.get('ids')
        target_names = group_filters.get('names')
        
        if not target_ids and not target_names:
            return files_with_metadata
            
        filtered_files = []
        
        for file_path, metadata in files_with_metadata:
            file_id = metadata.get('id', '')
            file_name = metadata.get('name', '')
            
            # Check if file matches any ID filter (if specified)
            id_match = False
            if target_ids:
                id_match = any(target_id in file_id for target_id in target_ids)
            
            # Check if file matches any name filter (if specified)
            name_match = False
            if target_names:
                name_match = any(target_name in file_name for target_name in target_names)
            
            # OR logic: include if matches any ID OR any name
            if target_ids and target_names:
                # Both specified: include if matches either condition
                if id_match or name_match:
                    filtered_files.append((file_path, metadata))
            elif target_ids:
                # Only IDs specified: include if matches any ID
                if id_match:
                    filtered_files.append((file_path, metadata))
            elif target_names:
                # Only names specified: include if matches any name
                if name_match:
                    filtered_files.append((file_path, metadata))
        
        return filtered_files
    
    def _apply_pattern_filters(self, file_paths: List[Path]) -> List[Path]:
        """
        Filter files based on include/exclude patterns applied to full filename.
        
        Args:
            file_paths: List of Path objects to filter
            
        Returns:
            Filtered list of Path objects
        """
        pattern_filters = self.config.get('file_pattern_filters', {})
        if not pattern_filters:
            return file_paths
            
        include_patterns = pattern_filters.get('include_patterns')
        exclude_patterns = pattern_filters.get('exclude_patterns')
        
        filtered_files = file_paths
        
        # Apply include patterns (file must match ALL include patterns)
        if include_patterns:
            new_filtered = []
            for file_path in filtered_files:
                filename = file_path.name
                if all(pattern in filename for pattern in include_patterns):
                    new_filtered.append(file_path)
            filtered_files = new_filtered
        
        # Apply exclude patterns (file must NOT match ANY exclude pattern)
        if exclude_patterns:
            new_filtered = []
            for file_path in filtered_files:
                filename = file_path.name
                if not any(pattern in filename for pattern in exclude_patterns):
                    new_filtered.append(file_path)
            filtered_files = new_filtered
        
        return filtered_files
    
    def _apply_balanced_count_limit(self, file_paths: List[Path]) -> List[Path]:
        """
        Apply count limit using balanced random sampling across groups.
        
        This method:
        1. Groups files by their ID+name combination
        2. Calculates how many files to select from each group (balanced)
        3. Randomly selects files within each group (seeded)
        
        Args:
            file_paths: List of Path objects to sample from
            
        Returns:
            Balanced random sample of Path objects
        """
        count = self.config.get('count')
        if count is None or count >= len(file_paths):
            # No count limit or count exceeds available files
            return file_paths
            
        # Group files by their metadata (ID + name combination)
        groups = defaultdict(list)
        for file_path in file_paths:
            metadata = self._parse_filename_metadata(file_path.name)
            group_key = f"{metadata.get('id', '')}_{metadata.get('name', '')}"
            groups[group_key].append(file_path)
        
        # Calculate balanced sampling
        num_groups = len(groups)
        if num_groups == 0:
            return []
        
        # Base files per group (integer division)
        base_per_group = count // num_groups
        # Remaining files to distribute
        remainder = count % num_groups
        
        # Set up random number generator with seed
        rng = random.Random(self.seed)
        
        selected_files = []
        group_names = list(groups.keys())
        
        # Randomly determine which groups get the extra files
        groups_for_extra = rng.sample(group_names, remainder) if remainder > 0 else []
        
        for group_name in group_names:
            group_files = groups[group_name]
            
            # Determine how many files to select from this group
            files_from_group = base_per_group
            if group_name in groups_for_extra:
                files_from_group += 1
            
            # Randomly select files from this group
            if files_from_group >= len(group_files):
                # Take all files from this group
                selected_files.extend(group_files)
            else:
                # Randomly sample from this group
                sampled = rng.sample(group_files, files_from_group)
                selected_files.extend(sampled)
        
        print(f"Balanced sampling: {len(groups)} groups, {base_per_group} base per group, {remainder} extra files")
        for group_name, group_files in groups.items():
            files_selected = len([f for f in selected_files if f in group_files])
            print(f"  Group {group_name}: {files_selected}/{len(group_files)} files selected")
        
        return selected_files
    
    def get_selection_summary(self, original_files: List[Path], selected_files: List[Path]) -> str:
        """
        Generate a human-readable summary of the selection process.
        
        Args:
            original_files: Original list of files before selection
            selected_files: Final list of selected files
            
        Returns:
            String summary of the selection process
        """
        summary = f"ND2 File Selection Summary:\n"
        summary += f"  Original files: {len(original_files)}\n"
        summary += f"  Selected files: {len(selected_files)}\n"
        
        if self.config:
            group_filters = self.config.get('group_filters', {})
            if group_filters.get('ids'):
                summary += f"  ID filters: {group_filters['ids']}\n"
            if group_filters.get('names'):
                summary += f"  Name filters: {group_filters['names']}\n"
                
            pattern_filters = self.config.get('file_pattern_filters', {})
            if pattern_filters:
                if pattern_filters.get('include_patterns'):
                    summary += f"  Include patterns: {pattern_filters['include_patterns']}\n"
                if pattern_filters.get('exclude_patterns'):
                    summary += f"  Exclude patterns: {pattern_filters['exclude_patterns']}\n"
            
            count = self.config.get('count')
            if count:
                summary += f"  Count limit: {count}\n"
                summary += f"  Sampling: Balanced random (seed: {self.seed})\n"
        else:
            summary += "  No filters applied (all files selected)\n"
        
        return summary
