*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Heart Failure Prediction using Microsoft Azure

*TODO:* Write a short introduction to your project.

In this project, we demonstrate how to use the Azure ML Python SDK to train a model to predict mortality by heart failure using Azure AutoML and Hyperdrive services. After training, we are going to deploy the best model and evaluate the model endpoint by consuming it.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

To set this project, we require access to Azure ML Studio. The application flow for the project design is as follows:
1. Create an Azure ML workspace with a compute instance.
2. Create an Azure ML compute cluster.
3. Upload the Heart Failure prediction dataset to Azure ML Studio from this repository.
4. Import the notebooks and scripts attached in this repository to the Notebooks section in Azure ML Studio.
5. All instructions to run the cells are detailed in the notebooks.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

In this project, we analyze the [Heart Failure Prediction](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) dataset containing the medical records of 299 heart failure patients collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015. The patients, who were aged 40 years and above, comprise of 105 women and 194 men who have all previously had heart failures.

The dataset contains 13 features, which report clinical, body, and lifestyle information and is use as the training data for predicting heart failure risks. This results in prediction models, which if accurate, can potentially be used to help hospitals in assessing the severity of patients with cardiovascular diseases. 

Additional information about this dataset can be found in the original dataset curators [publication](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181001).

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

The task here is to predict mortality due to heart failure. Heart failure is a common event caused by Cardiovascular diseases (CVDs), and  it occurs when the heart cannot pump enough blood to meet the needs of the body. The main reason behind heart failure include diabetes, high blood pressure, or other heart conditions or diseases.

The objective of the task is to train a binary classification model that predict the target column DEATH_EVENT, which indicates if a heart failure patient will survive or not before the end of the follow-up period, based on the information provided by the other 11 features (predictors). The time feature was dropped before training since we cannot get a time value for new patients after deployment. The predictors variables are:

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## Citation

Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020) [Article](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec13).
