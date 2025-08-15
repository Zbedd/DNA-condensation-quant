

Check the visualize-images output. Some labels are not filled, and homogeneity values will be skewed. Add a more robust fill
Big blobs of nuclei are being maintained. These need to be filtered out using a percent of median approach
Validate that batch_processor is working 
Use equivalent_diameter/2 to get the max radius of nuclei. Make sure bg_ball_radius is bigger than this value.