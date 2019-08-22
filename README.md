# Overview

For this project, I was tasked with identifying the 5 best zip codes for investment by a real estate firm. I focused on a single region, Washington, DC, for my analysis. I built functions to generalize my work so it can be applied to any subset of zip codes. My metrics for determining the "best" zip codes for investment were the returns on investment for 1 year and 2 years. 

<img src='images\dc_map.png'>

# Methodology 

I will briefly outline the methodology that I followed for this project. The same methodology can be used with the provided functions for analysis on any desired region. 

* Load dataset and isolate desired region
* Format and clean data
* Exploratory data analysis and visualizations
* Optimize model parameters
    * Use Forward Chaining Nested Cross Validation on each model
* Validate best model for each zipcode
* Visualize forecasts
* Calculate 1 year and 2 year ROI
* Select and visualize best zip codes
* Summary and future work

# Results

<img src='images\best_roi_results.png'>

<img src='images\DC_best_zipcodes.png'>
