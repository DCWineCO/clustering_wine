# Predicting Wine Quality Based on Physiochemical Properties

##### *A CodeUp Data Science Project by DeAdrian Hill and Corey Baughman*
##### April 19th, 2023
***

### Abstract
**Project Description:** Can the perceived quality of wine be predicted by it's physiochemical properties? This project uses a dataset of Portuguese vinho verde wines that were rated on a ten point quality scale by three experts. Each wine also was tested to measure 11 different physiochemical properties. The aim of this project is to assess whether perceived quality of wines can be predicted by a model derived from these properties. Wines are divided into general categories of white and red, both from variants of the same grape.

**Goal:** To explore physiochemical drivers of perceived quality in wines and to create an ML model which will predict quality more accurately than baseline chance.

**Plan of Attack:**
To achieve this goal we worked through the following steps in the data science pipeline. These steps also function as a Table of Contents to this document.
 
[1. Plan and Investigate Background Information](#Intro:-Background-Information)

[2. Formulate Initial Questions/Hypotheses](#Initial-Questions)

[3. Acquire and Prepare Data](#Data-Acquisition-and-Preparation)

[4. Exploratory Data Analysis](#Exploratory-Data-Analysis)

[5. Preprocessing](#Pre-processing)

[6. Modeling](#Modeling)

[7. Model Evaluation](#Model-Evaluation)

[8. Conclusions](#Conclusions)

[9. Next Steps](#Next-Steps)

[Appendix](#Appendix:-Instructions-to-Reproduce-Work)

***

### Intro: Background Information

This project is based on research work done by Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis in a 2009 paper called *Wine Quality*. Citation and link to abstract are provided in the Appendix. Their basic methodology was as follows:
> two datasets were created, using red and white wine samples.
The inputs include objective tests (e.g. PH values) and the output is based on sensory data
(median of at least 3 evaluations made by wine experts). Each expert graded the wine quality
between 0 (very bad) and 10 (very excellent)... The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables
are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
These datasets can be viewed as classification or regression tasks.
The classes are ordered and not balanced (e.g. there are munch more normal wines than
excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
or poor wines. Also, we are not sure if all input variables are relevant. So
it could be interesting to test feature selection methods.
Number of Instances: red wine - 1599; white wine - 4898.
Number of Attributes: 11 + output attribute
Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
feature selection.

#### Data Dictionary 

| Attribute | Description of Attribute |
| :---------| :------------------------ |
| fixed acidity | most acids involved with wine or fixed or nonvolatile (do not evaporate readily) |
| volatile acidity | the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste |
| citric acid | found in small quantities, citric acid can add 'freshness' and flavor to wines |
| residual sugar | the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet |
| chlorides | the amount of salt in the wine |
| free sulfur dioxide | the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine |
| total sulfur dioxide | amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine |
| density | the density of water is close to that of water depending on the percent alcohol and sugar content |
| pH | describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale |
| sulphates | a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant |
| alcohol | the percent alcohol content of the wine |
| red | the type of wine |
| white | the type of wine |

***

### Initial Questions

1. Does alcohol effect wine quality?
1. Does density effect wine quality?
1. Do chlorides effect wine quality?
1. Is there a difference in quality for red or white wine?
1. Is there a correlation between volatile acidity and quality?
1. Is there a linear correlation between residual free sulphur and quality?



***

### Data Acquisition and Preparation

- goals
- tools
- methods
- findings

***

### Exploratory Data Analysis

1. negative correlations:

    a. fixed acidity
    
    b. volatile acidity
    
    c. chlorides
    
    d. total sulfur dioxide
    
    e. density
    
2. positive correlations:
    
    a. citric acid
    
    b. free sulfur dioxide
    
    c. alcohol
    
    d. is red
    
3. no correlation:
    
    a. residual sugar
    
    b. pH
    
    c. sulphates
    
4. Clusters:

    We spent some effort examining different combinations of variables for useful clusters to aid in regression or classification, but we did not find any that beat our non-cluster models.
    
***
### Pre-processing

1. split target from features for all datasets

1. scaled X_train, X_validate, X_test

***

### Modeling

1. We chose to treat this as a regression problem

1. Regression algorithms used:

    a. OLS regressor
    
    b. Tweedie Regressor
    
    c. Polynomial Features
    
    d. LassoLars 

1. Findings: Our best model was a Polynomial Regression on all features that decreased errors by 23.35% over baseline

***

### Model Evaluation

1. established a baseline by testing mean and median quality values as predictions. We found that mean had a slightly lower RMSE and settled on that as a baseline model.

1. We used RMSE as our evaluation metric supplemented by plotting overlaid histograms of predicted and actual values to understand where the model performed well and where it didn't.

***

### Conclusions

1. We achieved our goals of finding drivers of wine quality as well as creating a model that outperforms baseline predictions of quality.

1. However, our model did not perform well at predicting high quality wines.

1. We found that almost all features were needed to get the best model.

***

### Next Steps

1. add class balancing to model
2. experiment with outlier detection modeling
3. explore using a classification model
***

### Appendix A: Instructions to Reproduce Work

1. Download project repo here:
https://github.com/DCWineCO/clustering_wine

2. Open and Run 'cluster_final_report.ipynb'

3. Necessary modules are included in the repo and should not need additional work to run as long as the whole repo is stored in the same directory.

### Appendix B: Links to Original Research

    - summary: https://data.world/food/wine-quality/workspace/project-summary?agentid=food&datasetid=wine-quality
    - research abstract: https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub

### Appendix C: Sources

This dataset is public available for research. The details are described in [Cortez et al., 2009].
Please include this citation if you plan to use this database:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
[Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
[bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

Title: Wine Quality
Sources
Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009


