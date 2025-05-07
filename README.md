## Bridging data gaps: predicting inadequate micronutrient intake with machine learning 

## Table of contents  
1. [Citing](#Citing)
2. [Abstract](#Abstract)
3. [Data] (#Data)

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


<a name="Abstract"/>

## Abstract

Identifying populations at risk of inadequate micronutrient intake is useful for governments and humanitarian organizations in low- and middle-income countries to make informed and timely decisions on nutrition relevant policies and programmes. We propose a machine-learning methodological approach using secondary data on household dietary diversity, socioeconomic status, and climate indicators to predict the risk of inadequate micronutrient intake in Ethiopia and in Nigeria. We identify key predictive features common to both countries, and  we demonstrate the model's transferability from one country to another to predict risk of inadequate micronutrient intake in contexts where nationally representative primary data are unavailable.

## Data 

**Target**
To generate the target variables, we use data from Household Consumption and Expenditure Surveys (HCES) collected as part of the Living Standards Measurement Study (LSMS) in both Ethiopia and Nigeria. For Ethiopia, we use the fourth wave of the Ethiopia Socioeconomic Survey, a nationally representative survey which collected information from 6770 households between May to September 2019 \cite{eth_ess}. For Nigeria, we use the Nigerian Living Standards Survey, a nationally representative survey which collected information from 22,587 households between September 2018 to September 2019 \cite{nig_nlss}. These data can be found in the folder named 

**Features**
We generate a set of 25 food group diversity, socioeconomic, and climate features. 
To generate **food group diversity-related features**, we use data from a different module of the same survey used to generate the targets, specifically the HCES data collected as part of the LSMS 2018/19 for Ethiopia and Nigeria \cite{eth_ess, nig_nlss}.


# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
