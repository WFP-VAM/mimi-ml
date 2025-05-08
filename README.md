# Bridging data gaps: predicting inadequate micronutrient intake with machine learning 

## Table of contents  
1. [Citing](#Citing)
2. [Abstract](#Abstract)
3. [Methodological approach](#Method)
4. [Data](#Data)
5. [Machine-learning](#ML)
6. [References](#References)

## Citing
<a name="Citing"/>

If you use our approach or code in this repository, please cite our paper:

Vasiliki Voukelatou, Kevin Tang, Ilaria Lauzana, Manita Jangid, Giulia Martini, Saskia de Pee, Frances Knight, and Duccio Piovani. "Bridging data gaps: predicting inadequate micronutrient intake with machine learning." bioRxiv (2025): 2025-04.
 <br/>
(https://www.biorxiv.org/content/10.1101/2025.04.08.647715v2.full.pdf)

`@article{voukelatou2025bridging, `<br/>
  `title={Bridging data gaps: predicting inadequate micronutrient intake with machine learning},`<br/>
  `author={Voukelatou, Vasiliki and Tang, Kevin and Lauzana, Ilaria and Jangid, Manita and Martini, Giulia and de Pee, Saskia and Knight, Frances and Piovani, Duccio},`<br/>
 `journal={bioRxiv},`<br/>
  `pages={2025--04},`<br/>
  `year={2025},`<br/>
  `publisher={Cold Spring Harbor Laboratory}`<br/>
}`

## Abstract
<a name="Abstract"></a>

Identifying populations at risk of inadequate micronutrient intake is useful for governments and humanitarian organizations in low- and middle-income countries to make informed and timely decisions on nutrition relevant policies and programmes. We propose a machine-learning methodological approach using secondary data on household dietary diversity, socioeconomic status, and climate indicators to predict the risk of inadequate micronutrient intake in Ethiopia and in Nigeria. We identify key predictive features common to both countries, and  we demonstrate the model's transferability from one country to another to predict risk of inadequate micronutrient intake in contexts where nationally representative primary data are unavailable.

## Methodological approach
<a name="Method"/></a>

We extract quantitative food consumption data from household level surveys to estimate the target variables. From the same surveys, we derive features related to food group diversity
and socioeconomic status. We also extract climate features from the WFP’s Seasonal Explorer platform. We then train machine-learning models using these variables to predict the risk of inadequate micronutrient intake (See Figure 1).

<img width="600" alt="results_models" src="https://github.com/user-attachments/assets/fd1ee7cb-7b88-4bad-a92a-2e0706ab9575">

<sup>Figure 1. Overview of the machine-learning methodological aprroach</sup>

## Data
<a name="Data"/></a>

**Target**<br/>
To generate the target variables, we use data from Household Consumption and Expenditure Surveys (HCES) collected as part of the Living Standards Measurement Study (LSMS) in both Ethiopia and Nigeria. For Ethiopia, we use the fourth wave of the Ethiopia Socioeconomic Survey, a nationally representative survey which collected information from 6770 households between May to September 2019 [1]. For Nigeria, we use the Nigerian Living Standards Survey, a nationally representative survey which collected information from 22,587 households between September 2018 to September 2019 [2]. The generated targets can be found in [`data/ethiopia_nigeria_targets.csv`](./data/ethiopia_nigeria_targets.csv)

**Features**<br/>
We generate a set of 25 food group diversity, socioeconomic, and climate features. 
To generate **food group diversity-related features**, and **socioeconomic-related features** we use data extracted from different modules of the same surveys used to generate the targets, specifically the HCES data collected as part of the LSMS 2018/19 for Ethiopia [1] and Nigeria [2]. To generate the **climate-related features** we extract data from WFP’s Seasonal Explorer platform [3], originally derived from satellite data. The features can be found in [`data/all_features.csv`](./data/all_features.csv), merged for both countries, and in [`data/eth/eth_features.csv`](./data/eth/eth_features.csv) and [`data/nga/nga_features.csv`](./data/nga/nga_features.csv), seperately for Ethiopia and Nigeria, respectively.

## Machine-learning
<a name="ML"/></a>

**Predicting risk of inadequate micronutrient intake with machine-learning** at a country level:<br/>The corresponding code can be found in `best_model_within_eth.ipynb` and `best_model_within_nga.ipynb` (for Ethiopia and Nigeria, respectively). <br>
**Cross-country models** and **application in data-constrained contexts**: <br/>The corresponding code can be found in `cross_country_running_notebook_eth_to_nga.ipynb` and `cross_country_running_notebook_nga_to_eth.ipynb.ipynb` (for the model trained in Ethiopia and the model trained in Nigeria, respectively). <br>
(metadata used for the analysis, training and visualisation are attached in the folder)

## References</a>
<a name="References"/>
- [1] Central Statistics Agency of Ethiopia. Ethiopia socioeconomic survey (ESS4) 2018-2019. Public Use Dataset. Ref: ETH_2018_ESS_v03. Downloaded from https://microdata.worldbank.org/index.php/catalog/3823/study-description on December 2024.<br/>
- [2] Nigeria National Bureau of Statistics. Nigeria Living Standards Survey (NLSS) 2018-2019. Dataset downloaded from https://microdata.worldbank.org/index.php/catalog/3827/study-description on December 2024.<br/>
- [3] World Food Programme. Wfp seasonal explorer. https://tinyurl.com/4vpp5mz9 (2022). (Online; accessed February 2023)
