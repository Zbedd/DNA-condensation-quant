"""
Configuration validator for ND2 selection settings.

This module provides validation for the nd2_selection_settings configuration
to ensure all parameters are valid before processing begins.
"""

import re
from typing import Dict, List, Tuple, Any, Optional


class ND2SelectionValidator:
    """
    Validates ND2 selection configuration parameters.
    
    Provides comprehensive validation of all nd2_selection_settings parameters
    with helpful error messages for debugging configuration issues.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.valid_sampling_strategies = ["sequential", "random", "balanced"]
        
    def validate_selection_config(self, config: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate complete ND2 selection configuration.
        
        Args:
            config: The nd2_selection_settings dictionary from config.yaml
            
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        if config is None:
            # No config means no filtering - this is valid
            return True, []
            
        errors = []
        
        # Validate count parameter
        is_valid, error_msg = self._validate_count(config.get('count'))
        if not is_valid:
            errors.append(error_msg)
            
        # Validate group filters
        group_filters = config.get('group_filters', {})
        is_valid, filter_errors = self._validate_group_filters(group_filters)
        if not is_valid:
            errors.extend(filter_errors)
            
        # Validate pattern filters
        pattern_filters = config.get('file_pattern_filters', {})
        is_valid, pattern_errors = self._validate_pattern_filters(pattern_filters)
        if not is_valid:
            errors.extend(pattern_errors)
            
        # Validate seed
        is_valid, error_msg = self._validate_seed(config.get('seed'))
        if not is_valid:
            errors.append(error_msg)
            
        return len(errors) == 0, errors
    
    def _validate_count(self, count: Any) -> Tuple[bool, str]:
        """
        Validate count parameter.
        
        Args:
            count: The count value to validate
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if count is None:
            return True, ""
            
        if not isinstance(count, int):
            return False, f"count must be an integer or null, got {type(count).__name__}: {count}"
            
        if count <= 0:
            return False, f"count must be positive, got: {count}"
            
        return True, ""
    
    def _validate_group_filters(self, group_filters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate group filters (ids and names).
        
        Args:
            group_filters: Dictionary containing 'ids' and 'names' keys
            
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors = []
        
        if not isinstance(group_filters, dict):
            return False, [f"group_filters must be a dictionary, got {type(group_filters).__name__}"]
            
        # Validate ids
        ids = group_filters.get('ids')
        if ids is not None:
            if not isinstance(ids, list):
                errors.append(f"group_filters.ids must be a list or null, got {type(ids).__name__}: {ids}")
            else:
                for i, id_val in enumerate(ids):
                    if not isinstance(id_val, str):
                        errors.append(f"group_filters.ids[{i}] must be a string, got {type(id_val).__name__}: {id_val}")
                    elif not id_val.strip():
                        errors.append(f"group_filters.ids[{i}] cannot be empty or whitespace")
        
        # Validate names
        names = group_filters.get('names')
        if names is not None:
            if not isinstance(names, list):
                errors.append(f"group_filters.names must be a list or null, got {type(names).__name__}: {names}")
            else:
                for i, name_val in enumerate(names):
                    if not isinstance(name_val, str):
                        errors.append(f"group_filters.names[{i}] must be a string, got {type(name_val).__name__}: {name_val}")
                    elif not name_val.strip():
                        errors.append(f"group_filters.names[{i}] cannot be empty or whitespace")
        
        return len(errors) == 0, errors
    
    def _validate_pattern_filters(self, pattern_filters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate pattern filters (include_patterns and exclude_patterns).
        
        Args:
            pattern_filters: Dictionary containing pattern filter settings
            
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors = []
        
        if not isinstance(pattern_filters, dict):
            return False, [f"file_pattern_filters must be a dictionary, got {type(pattern_filters).__name__}"]
        
        # Validate include_patterns
        include_patterns = pattern_filters.get('include_patterns')
        if include_patterns is not None:
            is_valid, error_msgs = self._validate_pattern_list(include_patterns, "include_patterns")
            if not is_valid:
                errors.extend(error_msgs)
        
        # Validate exclude_patterns
        exclude_patterns = pattern_filters.get('exclude_patterns')
        if exclude_patterns is not None:
            is_valid, error_msgs = self._validate_pattern_list(exclude_patterns, "exclude_patterns")
            if not is_valid:
                errors.extend(error_msgs)
        
        return len(errors) == 0, errors
    
    def _validate_pattern_list(self, patterns: Any, field_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a list of regex patterns.
        
        Args:
            patterns: The pattern list to validate
            field_name: Name of the field for error messages
            
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors = []
        
        if not isinstance(patterns, list):
            return False, [f"{field_name} must be a list or null, got {type(patterns).__name__}: {patterns}"]
        
        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, str):
                errors.append(f"{field_name}[{i}] must be a string, got {type(pattern).__name__}: {pattern}")
            elif not pattern.strip():
                errors.append(f"{field_name}[{i}] cannot be empty or whitespace")
            else:
                # Test if it's a valid regex pattern
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"{field_name}[{i}] is not a valid regex pattern '{pattern}': {e}")
        
        return len(errors) == 0, errors
    
    def _validate_seed(self, seed: Any) -> Tuple[bool, str]:
        """
        Validate seed parameter.
        
        Args:
            seed: The seed value to validate
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if seed is None:
            return False, "seed cannot be null - required for reproducible balanced random sampling"
            
        if not isinstance(seed, int):
            return False, f"seed must be an integer, got {type(seed).__name__}: {seed}"
            
        return True, ""
    
    def get_validation_summary(self, config: Optional[Dict[str, Any]]) -> str:
        """
        Generate a human-readable validation summary.
        
        Args:
            config: The nd2_selection_settings configuration
            
        Returns:
            String summary of validation results
        """
        is_valid, errors = self.validate_selection_config(config)
        
        if config is None:
            return "ND2 Selection Config: No selection settings (will process all files)"
            
        summary = "ND2 Selection Config Validation:\n"
        
        if is_valid:
            summary += "  ✅ Configuration is valid\n"
            
            # Summarize settings
            count = config.get('count')
            if count:
                summary += f"  📊 Count limit: {count} files\n"
            else:
                summary += "  📊 Count limit: None (process all files)\n"
                
            group_filters = config.get('group_filters', {})
            ids = group_filters.get('ids')
            names = group_filters.get('names')
            
            if ids or names:
                summary += "  🔍 Group filters:\n"
                if ids:
                    summary += f"    - IDs: {ids}\n"
                if names:
                    summary += f"    - Names: {names}\n"
                summary += "    - Logic: OR (file matches ANY ID OR ANY name)\n"
            else:
                summary += "  🔍 Group filters: None\n"
                
            pattern_filters = config.get('file_pattern_filters', {})
            include_patterns = pattern_filters.get('include_patterns')
            exclude_patterns = pattern_filters.get('exclude_patterns')
            
            if include_patterns or exclude_patterns:
                summary += "  📝 Pattern filters:\n"
                if include_patterns:
                    summary += f"    - Include: {include_patterns}\n"
                if exclude_patterns:
                    summary += f"    - Exclude: {exclude_patterns}\n"
            else:
                summary += "  📝 Pattern filters: None\n"
                
            seed = config.get('seed', 42)
            summary += f"  🎲 Random seed: {seed}\n"
            summary += "  ⚖️ Sampling: Balanced random across groups\n"
            
        else:
            summary += "  ❌ Configuration has errors:\n"
            for error in errors:
                summary += f"    - {error}\n"
                
        return summary
