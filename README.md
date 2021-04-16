*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Heart Failure Prediction using Microsoft Azure

*TODO:* Write a short introduction to your project.

In this project, we demonstrate how to use the Azure ML Python SDK to train a model to predict mortality due to heart failure using Azure AutoML and Hyperdrive services. After training, we are going to deploy the best model and evaluate the model endpoint by consuming it.

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

The task here is to predict mortality due to heart failure. Heart failure is a common event caused by Cardiovascular diseases (CVDs), and  it occurs when the heart cannot pump enough blood to meet the needs of the body. The main reasons behind heart failure include diabetes, high blood pressure, or other heart conditions or diseases. By applying machine learning procedure to this analysis, we will have a predictive model that can potentially impact clinical practice, becoming a new supporting tool for physicians when assessing the increased risk of mortality among heart failure patients.

The objective of the task is to train a binary classification model that predict the target column <b>DEATH_EVENT</b>, which indicates if a heart failure patient will survive or not before the end of the follow-up period. This is based on the information provided by the 11 clinical features (or risk factors). The <b>time</b> feature was dropped before training since we cannot get a time value for new patients after deployment. The predictors variables are as follows:

1. Age: age of patient (years)
2. Anaemia: Decrease of red blood cells or hemoglobin. It has a value of 1 or 0 with 1 being the patient does have this condition
3. Creatinine Phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
4. Diabetes:  Is a 1 or 0 - whether the patient suffers from diabetes or not 
5. Ejection Fraction: Percentage of blood leaving the heart at each contraction (percentage)
6. High Blood Pressure: Is a 1 or 0 - If the patient has hypertension 
7. Platelets: Platelets in the blood (kiloplatelets/mL) 
8. Serum Creatinine: Level of serum creatinine in the blood (mg/dL)
9. Serum Sodium: Level of serum sodium in the blood (mEq/L)
10. Sex: Woman or man (binary)
11. Smoking: If the patient smokes or not
12. Time: Follow-up period (days)

Target variable - Death Event: If the patient died during the follow-up period
Death Event = 1 for dead patients and Death Event = 0 for survived patients

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The data for this project can be accessed in our workspace through the following steps:

* Download the data from [UCI Machine learning repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) or the [uploaded dataset](https://github.com/PeacePeters/Heart-Failure-Prediction-using-AzureML/blob/main/heart_failure.csv) in this GitHub repository

* Register the dataset either using AzureML SDK or AzureML Studio using a weburl or from local files.

* For this project, we registered the dataset in our workspace using a weburl in Azure SDK, retrieving the data from the csv file using <b>TabularDatasetFactory</b> Class .

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

We have used following configuration for AutoML.
```ruby
automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}

automl_config = AutoMLConfig(compute_target=compute_target,
                             task="classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",
                             n_cross_validations=5,
                             debug_log="automl_errors.log",
                             **automl_settings
                            )
```

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
