# DNA Condensation Analysis - Project Structure

## Overview
This document describes the organization of the DNA condensation analysis project.

## Directory Structure

```
DNA-condensation-quant/
├── dna_condensation/           # Main analysis package
│   ├── core/                   # Core analysis modules
│   ├── validation/             # Validation framework (BBBC022)
│   │   ├── output/            # Validation results and outputs
│   │   ├── bbbc022_validation.py
│   │   └── README.md
│   └── analysis_archive/       # Historical analysis results
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── docs/                       # Documentation files
├── dev_scripts/               # Development utilities
└── README.md                  # Main project documentation
```

## Key Components

### Validation Framework (`dna_condensation/validation/`)
- **Purpose**: Benchmark validation using BBBC022 dataset
- **Status**: Fully functional with significant biological validation (p=0.0479, Cohen's d=0.78)
- **Outputs**: Statistical comparisons, visualizations, and processed data

### Analysis Archive (`dna_condensation/analysis_archive/`)
- **Purpose**: Historical analysis results and development iterations
- **Contents**: Timestamped analysis runs and debugging outputs
- **Status**: Archived for reference

### Documentation (`docs/`)
- **Purpose**: Project documentation and technical summaries
- **Contents**: Implementation summaries, feature documentation, and project structure

## Usage
- Main analysis: Use modules in `dna_condensation/core/`
- Validation: Run scripts in `dna_condensation/validation/`
- Testing: Execute tests from `tests/` directory
- Examples: Reference implementations in `examples/`

## Maintenance
- Keep validation framework up to date with new benchmark datasets
- Archive old analysis results to maintain clean project structure
- Update documentation when adding new features or modules
