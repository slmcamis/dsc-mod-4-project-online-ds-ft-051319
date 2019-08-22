# Overview

For this project, I was tasked with identifying the 5 best zip codes for investment by a real estate firm. I focused on a single region, Washington, DC, for my analysis. I built functions to generalize my work so it can be applied to any subset of zip codes. My metrics for determining the "best" zip codes for investment were the returns on investment for 1 year and 2 years. 

<img src='images\dc_map.png'>

# Methodology 

<a href='DC_Zipcode_Analysis.ipynb'>DC Zipcode Analysis Notebook</a>
<a href=https://docs.google.com/presentation/d/1-73HQuUa8xQMfJbmJ0aaUQO5ueFg79XBtxcC8s-Kokc/edit?usp=sharing>Presentation</a>

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

# Results and Recommendations

<img src='images\best_roi_results.png'>

<img src='images\DC_best_zipcodes.png'>

* Invest in Southeast and Northeast quadrants of DC
* Expand analysis region to include entire DMV area
* Connect with local government concerning gentrification

# Future Work

* Continue to optimize models for more accurate predictions
* Explore other evaluation metrics
* Test functions on other regions


