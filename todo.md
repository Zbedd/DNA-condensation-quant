

Check the visualize-images output. Some labels are not filled, and homogeneity values will be skewed. Add a more robust fill {cm:2025-08-16}
Big blobs of nuclei are being maintained. These need to be filtered out using a percent of median approach {cm:2025-08-16}
Validate that batch_processor is working {cm:2025-08-23}

Revisit nuclei filtering as needed {cm:2025-08-23}
Figure out why nuclear density = 1 {cm:2025-08-18}
Important: write a script to validate bbbc image grouping {cm:2025-08-23}
Improve filtering. I had to set min_size relative to medium to >100, indicating most cells are noise. {cm:2025-08-23}
Add CCP {cm:2025-08-24}
Add CI {cm:2025-08-24}
Add a summary plot (minimum CCP, CV, CI) {cm:2025-08-24}
Get CI calculated correctly
Fix key_metrics