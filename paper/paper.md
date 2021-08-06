---
title: 'Feature-engine: A Python package for feature engineering for machine learning'
tags:
  - Python
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

Feature-engine is an open source Python library with an exhaustive battery of 
transformers to engineer and select features for use in machine learning models. 
Feature-engine simplifies and streamlines the implementation of end-to-end feature
engineering pipelines, by allowing the selection of feature subsets within its 
transformers, and returning dataframes for easy data exploration and visualization. 
Feature-engine’s transformers preserve Scikit-learn functionality with the methods fit()
and transform() to learn parameters from and then transform data. Feature-engine's 
transformers can be incorporated into a Scikit-learn Pipeline to streamline data 
transformation and model deployment.

The primary goal of Feature-engine is to make commonly used data transformation 
procedures accessible to researchers and data scientists, focusing on creating user-
friendly and intuitive classes, compatible with existing machine learning libraries, 
such as Scikit-learn (Pedregosa et al., 2011) and Pandas (citation).


# Statement of need

Data scientists spend an enormous amount of time on data pre-processing and 
transformation ahead of training machine learning models. As a result they produce a lot
of code, which is often repetitive as the same procedures are applied to different 
datasets. These poses challenges at the time of model deployment, as code needs to be
often re-written to production standards, which is not only time consuming, but also
increases the opportunities for lack of reproducibility.

Feature-engine brings together the most frequently used data pre-processing techniques 
and transformations in a battery of classes that users can use off-the-shelf, to 
transform their data and select their features, just like they would use any other
Scikit-learn transformer class. Like this, Feature-engine decreases the amount of code
users need to develop from scratch, reduces deployment timelines and maximises 
reproducibility.


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

I would like to acknowledge all of the contributors and users of Feature-engine, who 
helped with valuable feedback, bug fixes, and additional functionality to further 
improve the library. A special thanks to Christopher Samiullah for continuous support
on code quality and architecture. A comprehensive list of all contributors to 
Feature-engine is available at 
https://github.com/feature-engine/feature_engine/graphs/contributors.

# References