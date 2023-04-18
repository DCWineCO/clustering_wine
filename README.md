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

Blah blah

***

### Data Acquisition and Preparation

- goals
- tools
- methods
- findings

***

### Exploratory Data Analysis

- goals
- tools
- methods
- findings

**Univariate Analysis**

- blah blah

**Bivariate Analysis**
 
- blah blah

**Multivariate Analysis**

- blah blah

***
### Pre-processing

- goals
- tools
- methods
- findings

***

### Modeling

- goals
- tools
- methods
- findings

***

### Model Evaluation

- goals
- tools
- methods
- findings

***

### Conclusions

- goals
- tools
- methods
- findings

***

### Next Steps

- goals
- tools
- methods
- findings

***

### Appendix: Instructions to Reproduce Work

1. Download project repo here:
https://github.com/DCWineCO/clustering_wine

1. Project links:

    - summary: https://data.world/food/wine-quality/workspace/project-summary?agentid=food&datasetid=wine-quality
    - research abstract: https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub

1. Sources:

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


