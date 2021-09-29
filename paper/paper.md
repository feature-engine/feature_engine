---
title: 'Feature-engine: A Python package for feature engineering for machine learning'
tags:
  - python
  - feature engineering
  - feature selection
  - machine learning
  - data science
authors:
  - name: Soledad Galli
    affiliation: 1
affiliations:
 - name: Train in Data
   index: 1
date: 6 August 2021
bibliography: paper.bib
---

# Summary

Feature-engine is an open source Python library with the most exhaustive battery of 
transformations to engineer and select features for use in machine learning. Feature-engine 
supports several techniques to impute missing data, encode categorical variables, transform 
variables mathematically, perform discretization, remove or censor outliers, and combine 
variables into new features. Feature-engine also hosts an array of algorithms for feature 
selection.

The primary goal of Feature-engine is to make commonly used data transformation procedures 
accessible to researchers and data scientists, focusing on creating user-friendly and 
intuitive classes, compatible with existing machine learning libraries, like Scikit-learn 
[@sklearn] and Pandas [@pandas].

Many feature transformation techniques learn parameters from data, like the values for 
imputation or the mappings for encoding. Feature-engine classes learn these parameters 
from the data and store them in their attributes to transform future data. Feature-engine’s 
transformers preserve Scikit-learn’s functionality with the methods fit() and transform() 
to learn parameters from and then transform data. Feature-engine's transformers can be 
incorporated into a Scikit-learn Pipeline to streamline data transformation and facilitate 
model deployment, by allowing the serialization of the entire pipeline in one pickle.

When pre-processing a dataset different feature transformations are applied to different 
variable groups. Feature-engine classes allow the user to select which variables to transform 
within each class, therefore, while taking the entire dataframe as input, only the indicated 
variables are modified. Data pre-processing and feature engineering are commonly done 
together with data exploration. Feature-engine transformers return dataframes as output, 
thus, users can continue to leverage the power of Pandas for data analysis and visualization 
after transforming the data set.

In summary, Feature-engine supports a large variety of commonly used data transformation 
techniques [@data_prep; @boxcox; @yeojohnson; @kdd_2009_competition; 
@beatingkaggle; @micci_mean_encoder], as well as techniques that were developed 
in data science competitions [@niculescu09_kdd], including those for feature selection 
[@miller09_kdd]. Thus, Feature-engine builds upon and extends the capabilities of 
Python's current scientific computing stack and makes accessible transformations that 
are otherwise not easy to find, understand or code, to data scientist and data 
practitioners.



# Statement of need

Data scientists spend an enormous amount of time on data pre-processing and transformation 
ahead of training machine learning models [@domingos]. While some feature engineering 
processes can be domain-specific, a large variety of transformations are commonly applied 
across datasets. For example, data scientists need to impute or remove missing values or 
transform categories into numbers, to train machine learning models using Scikit-learn, 
the main library for machine learning. Yet, depending on the nature of the variable and 
the characteristics of the machine learning model, they may need to use different techniques. 

Feature-engine gathers the most frequently used data pre-processing techniques, as well as 
bespoke techniques developed in data science competitions, in a library, from which users can pick 
and choose the transformation that they need, and use it just like they would use any other 
Scikit-learn class. As a result, users are spared of manually creating a lot of code, which 
is often repetitive, as the same procedures are applied to different datasets. In addition, 
Feature-engine classes are written to production standards, which ensures classes return 
the expected result, and maximizes reproducibility between research and production 
environments through version control.

In the last few years, a number of open source Python libraries that support feature 
engineering techniques have emerged, highlighting the importance of making feature 
engineering and creation accessible and, as much as possible, automated. Among these, 
Featuretools [@kanter2015deep] creates features from temporal and relational datasets, 
tsfresh [@christ_tsfresh] extracts features from time series, Category encoders 
[@category_encoders] supports a comprehensive list of methods to encode categorical 
variables, and Scikit-learn [@sklearn] implements a number of data transformation 
techniques, with the caveat that the transformations are applied to the entire dataset, 
and the output are NumPy arrays. Feature-engine extends the capabilities of the current 
Python’s scientific computing stack by allowing the application of the transformations 
to subsets of variables in the dataset, returning dataframes for data exploration, and 
supporting transformations not currently available in other libraries, like those for 
outlier censoring or removal, besides additional techniques for discretization and 
feature selection that were developed by data scientist working in the industry or data 
science competitions.


# Acknowledgements

I would like to acknowledge all of the contributors and users of Feature-engine, who helped 
with valuable feedback, bug fixes, and additional functionality to further improve the library. 
A special thanks to Christopher Samiullah for continuous support on code quality and 
architecture. A list of  Feature-engine contributors is available at 
https://github.com/feature-engine/feature_engine/graphs/contributors.

# References