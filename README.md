# DataStrategyTeam17

Our team is composed by Noa Azran, Clara Descos, Robin Bricage, Maria Moret Torrella, and Hubert de Parseval.

## Content

This repository is used to present the results of the BCG Gamma AI course challenge. 
The goal is to predict prove the value of an investment in advanced data analytics capabilities for ClientCo.
This repository contains our preprocessing and clustering pipelines and our models. For storage reasons, no data 
is included.

## Approach description

We started by analyzing the data to decide the use case to push the most, and then we dove into the definition
of a churning client. We then set up to preprocess the dataset and build a clustering pipeline to add features 
that might be useful for the a churn prediction model.

## Project usage

In order to use the repository a data folder should be added manually. You can use the command 
pip install -r requirements.txt to install all the requirements. The src folder contains all the scripts 
used in the notebooks from the main folder. We decided to separate the main.py and 
the clustering main.py for more clarity as it is not yet used in the model.

## Future improvements

Geographical and client data would enable to interpret the clustering part, thus unlocking valuable insights. It 
also likely would increase the model accuracy. However, churn prediction is vulnerable to heavy biases due to 
the data selection and to the difficulty of defining churn for ClientCo's business. On a more theoretical level, 
a consistent churn prediction model would need to be trained on counterfactual data to really be accurate, which 
in real life is scarce.
