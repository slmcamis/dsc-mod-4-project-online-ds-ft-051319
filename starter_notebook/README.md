
# Module 4 -  Final Project Specifications

## Introduction

In this lesson, we'll review all the guidelines and specifications for the final project for Module 4.

## Objectives

* Understand all required aspects of the Final Project for Module 4
* Understand all required deliverables
* Understand what constitutes a successful project

### Final Project Summary

Another module down--you're absolutely crushing it! For this project, you'll get to flex your **_Time-Series_** muscles!

<img src='https://raw.githubusercontent.com/learn-co-curriculum/dsc-mod-4-project/master/images/timegif.gif'>

For this module's final project, we're going to put your newfound **_Time Series Analysis_** skills to the test. You will be forecasting real estate prices of various zipcodes using data from [Zillow](https://www.zillow.com/research/data/). However, this won't be as straightforward as just running a time-series analysis--you're going to have to make some data-driven decisions and think critically along the way!

### The Project

For this project, you will be acting as a consultant for a fictional real-estate investment firm. The firm has asked you what seems like a simple question:

> what are the top 5 best zipcodes for us to invest in?

This may seem like a simple question at first glance, but there's more than a little ambiguity here that you'll have to think through in order to provide a solid recommendation. Should your recommendation be focused on profit margins only? What about risk? What sort of time horizon are you predicting against?  Your recommendation will need to detail your rationale and answer any sort of lingering questions like these in order to demonstrate how you define "best".

As mentioned previously, the data you'll be working with comes from the [Zillow Research Page](https://www.zillow.com/research/data/). However, there are many options on that page, and making sure you have exactly what you need can be a bit confusing. For simplicity's sake, we have already provided the dataset for you in this repo--you will find it in the file `zillow_data.csv`.

## The Deliverables

The goal of this project is to have you complete a very common real-world task in regard to Time-Series Modeling. However, real world problems often come with a significant degree of ambiguity, which requires you to use your knowledge of statistics and data science to think critically about and answer. While the main task in this project is Time-Series Modeling, that isn't the overall goal--it is important to understand that Time-Series Modeling is a tool in your toolbox, and the forecasts it provides you are what you'll use to answer important questions.

In short, to pass this project, demonstrating the quality and thoughtfulness of your overall recommendation is at least as important as successfully building a Time-Series model!

In order to successfully complete this project, you must have:

* A well-documented **_Jupyter Notebook_** containing any code you've written for this project (use the notebook in this repo, `mod_4_starter_notebook.ipynb`)
* A **_[Blog post](https://github.com/learn-co-curriculum/dsc-welcome-blogging)_**.
* An **_'Executive Summary' PowerPoint Presentation_** that explains your rationale and methodology for determining the best zipcodes for investment.


### Jupyter Notebook Must-Haves

For this project, you will be provided with a jupyter notebook containing some starter code. If you inspect the zillow dataset file, you'll notice that the datetimes for each sale are the actual column names--this is a format you probably haven't seen before. To ensure that you're not blocked by preprocessing, we've provided some helper functions to help simplify getting the data into the correct format. You're not required to use this notebook or keep it in its current format, but we strongly recommend you consider making use of the helper functions so you can spend your time working on the parts of the project that matter.

#### Organization/Code Cleanliness

The notebook should be well organized, easy to follow, and code is modularized and commented where appropriate.

* Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code. All functions have docstrings that act as professional-quality documentation.
* The notebook is written to technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.
* Data visualizations you create should be clearly labeled and contextualized--that is, they fit with the surrounding code or problems you're trying to solve. No dropping data visualizations randomly around your notebook without any context!

#### Findings

Your notebook should briefly mention the metrics you have defined as "best", so that any readers understand what technical metrics you are trying to optimize for (for instance, risk vs profitability, ROI yield, etc.). You do **not** need to explain or defend your your choices in the notebook--the blog post and executive summary presentation are both better suited to that sort of content. However, the notebook should provide enough context about your definition for "best investment" so that they understand what the code you are writing is trying to solve.

#### Visualizations

Time-Series Analysis is an area of data science that lends itself well to intuitive data visualizations. Whereas we may not be able to visualize the best choice in a classification or clustering problem with a high-dimensional dataset, that isn't an issue with Time Series data. As such, **_any findings worth mentioning in this problem are probably also worth visualizing_**. Your notebook should make use of data visualizations as appropriate to make your findings obvious to any readers.

Also, remember that if a visualization is worth creating, then it's also worth taking the extra few minutes to make sure that it is easily understandable and well-formatted. When creating visualizations, make sure that they have:

* A title
* Clearly labeled X and Y axes, with appropriate scale for each
* A legend, when necessary
* No overlapping text that makes it hard to read
* An intelligent use of color--multiple lines should have different colors and/or symbols to make them easily differentiable to the eye
* An appropriate amount of information--avoid creating graphs that are "too busy"--for instance, don't create a line graph with 25 different lines on it

<center><img src='images/bad-graph-1.png' height=100% width=100%>
There's just too much going on in this graph for it to be readable--don't make the same mistake! (<a href='http://genywealth.com/wp-content/uploads/2010/03/line-graph.php_.png'>Source</a>)</center>

### Blog Post Must-Haves

Refer back to the [Blogging Guidelines](https://github.com/learn-co-curriculum/dsc-welcome-blogging) for the technical requirements and blog ideas.


### Executive Summary Must-Haves

Your presentation should:

Contain between 5-10 professional quality slides detailing:

* A high-level overview of your methodology and findings, including the 5 zipcodes you recommend investing in
* A brief explanation of what metrics you defined as "best" in order complete this project

As always, this prresentation should also:

* Take no more than 5 minutes to present
* Avoid technical jargon and explain results in a clear, actionable way for non-technical audiences.




# Mod 4 Project - Starter Notebook

This notebook has been provided to you so that you can make use of the following starter code to help with the trickier parts of preprocessing the Zillow dataset. 

The notebook contains a rough outline the general order you'll likely want to take in this project. You'll notice that most of the areas are left blank. This is so that it's more obvious exactly when you should make use of the starter code provided for preprocessing. 

**_NOTE:_** The number of empty cells are not meant to infer how much or how little code should be involved in any given step--we've just provided a few for your convenience. Add, delete, and change things around in this notebook as needed!

# Some Notes Before Starting

This project will be one of the more challenging projects you complete in this program. This is because working with Time Series data is a bit different than working with regular datasets. In order to make this a bit less frustrating and help you understand what you need to do (and when you need to do it), we'll quickly review the dataset formats that you'll encounter in this project. 

## Wide Format vs Long Format

If you take a look at the format of the data in `zillow_data.csv`, you'll notice that the actual Time Series values are stored as separate columns. Here's a sample: 

<img src='~/../images/df_head.png'>

You'll notice that the first seven columns look like any other dataset you're used to working with. However, column 8 refers to the median housing sales values for April 1996, column 9 for May 1996, and so on. This This is called **_Wide Format_**, and it makes the dataframe intuitive and easy to read. However, there are problems with this format when it comes to actually learning from the data, because the data only makes sense if you know the name of the column that the data can be found it. Since column names are metadata, our algorithms will miss out on what dates each value is for. This means that before we pass this data to our ARIMA model, we'll need to reshape our dataset to **_Long Format_**. Reshaped into long format, the dataframe above would now look like:

<img src='~/../images/melted1.png'>

There are now many more rows in this dataset--one for each unique time and zipcode combination in the data! Once our dataset is in this format, we'll be able to train an ARIMA model on it. The method used to convert from Wide to Long is `pd.melt()`, and it is common to refer to our dataset as 'melted' after the transition to denote that it is in long format. 

# Helper Functions Provided

Melting a dataset can be tricky if you've never done it before, so you'll see that we have provided a sample function, `melt_data()`, to help you with this step below. Also provided is:

* `get_datetimes()`, a function to deal with converting the column values for datetimes as a pandas series of datetime objects
* Some good parameters for matplotlib to help make your visualizations more readable. 

Good luck!


# Step 1: Load the Data/Filtering for Chosen Zipcodes

# Step 2: Data Preprocessing


```python
def get_datetimes(df):
    return pd.to_datetime(df.columns.values[1:], format='%Y-%m')
```

# Step 3: EDA and Visualization


```python
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

# NOTE: if you visualizations are too cluttered to read, try calling 'plt.gcf().autofmt_xdate()'!
```

# Step 4: Reshape from Wide to Long Format


```python
def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName', 'City', 'State', 'Metro', 'CountyName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})
```

# Step 5: ARIMA Modeling

# Step 6: Interpreting Results
