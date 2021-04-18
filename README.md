# Heart Failure Prediction using AzureML

In this project, we demonstrate how to use the Azure ML Python SDK to train a model to predict mortality due to heart failure using Azure AutoML and Hyperdrive services. After training, we are going to deploy the best model and evaluate the model endpoint by consuming it. 

This trained and deployed predictive model can potentially impact clinical practice, becoming a new supporting tool for physicians when assessing the increased risk of mortality among heart failure patients.

## Table of Contents
1. [Project Set Up and Installation](#Project-Set-Up-and-Installation)
2. [Dataset](#Dataset)
   * [Overview](#Overview)
   * [Task](#Task)
   * [Access](#Access)
3. [Automated ML](#Automated-ML)
   * [Results](#Results)
   * [Improvements for AutoML](#Improvements-for-AutoML)
4. [Hyperparameter Tuning](#Hyperparameter-Tuning)
   * [Results](#Results)
   * [Improvements for Hyperparameter Tuning](#Improvements-for-Hyperparameter-Tuning)
5. [Automated ML and Hyperparameter Tuning Comparison](#Automated-ML-and-Hyperparameter-Tuning-Comparison)
6. [Model Deployment](#Model-Deployment)
7. [Screen Recording](#Screen-Recording)
8. [Future Improvements](#Future-Improvements)
9. [Standout Suggestions](#Standout-Suggestions)
10. [Citation](#Citation)

## Project Set Up and Installation

To set this project, we require access to Azure ML Studio. The application flow for the project design is as follows:
1. Create an Azure ML workspace with a compute instance.
2. Create an Azure ML compute cluster.
3. Upload the Heart Failure prediction dataset to Azure ML Studio from this repository.
4. Import the notebooks and scripts attached in this repository to the Notebooks section in Azure ML Studio.
5. All instructions to run the cells are detailed in the notebooks.

## Dataset

### Overview

The Heart [Heart Failure Prediction](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) dataset is used for assessing the severity of patients with heart failure. It contains the medical records of 299 heart failure patients collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015. The patients, who are aged 40 years and above, comprise of 105 women and 194 men who have all previously had heart failures.

The dataset contains 13 features, which report clinical, body, and lifestyle information and is use as the training data for predicting heart failure risks. Regarding the dataset imbalance, the survived patients (death event = 0) are 203, while the dead patients (death event = 1) are 96. 

Additional information about this dataset can be found in the original dataset curators [publication](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181001).

### Task

The task here is to predict mortality due to heart failure. Heart failure is a common event caused by Cardiovascular diseases (CVDs), and  it occurs when the heart cannot pump enough blood to meet the needs of the body. The main reasons behind heart failure include diabetes, high blood pressure, or other heart conditions or diseases. By applying machine learning procedure to this analysis, we will have a predictive model that assesses the severity of patients with heart failure.

The objective of the task is to train a binary classification model that predict the target column <b>DEATH_EVENT</b>, which indicates if a heart failure patient will survive or not before the end of the follow-up period. This is based on the information provided by the 11 clinical features (or risk factors). The <b>time</b> feature is dropped before training since we cannot get a time value for new patients after deployment. The predictors variables are as follows:

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

The data for this project can be accessed in our workspace through the following steps:

* Download the data from [UCI Machine learning repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) or the [uploaded dataset](https://github.com/PeacePeters/Heart-Failure-Prediction-using-AzureML/blob/main/heart_failure.csv) in this GitHub repository

* Register the dataset either using AzureML SDK or AzureML Studio using a weburl or from local files.

* For this project, we registered the dataset in our workspace using a weburl in Azure SDK and retrieve the data from the csv file using the <b>TabularDatasetFactory</b> Class.

## Automated ML

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

As shown in above code snippet, the AutoML settings are: 

* The <i>task</i> for this machine learning problem is classification
* The <i>primary_metric</i> used is AUC weighted, which is more appropriate than accuracy since the dataset is moderately imbalanced (67.89% negative elements and 32.11% positive elements). 
* <i>n_cross_validation</i> of 5 folds rather than 3 is used which gives a better performance. 
* An <i>experiment_timeout_minutes</i> of 30 is specified to constrain usage.
* The <i>max_concurrent_iterations</i> to be executed in parallel during training is set to 5 so the process is completed faster.

### Results

The Best model is ```VotingEnsemble``` with an AUC value of ```0.9226163713547434```

Model hyper-parameters used for VotingEnsemble are shown below:

The parameters for the model VotingEnsemble are described in the table below:

```Minmaxscaler```
Parameters | Values |
 | ------------- | -------------
copy | True
feature_range | (0, 1)

randomforestclassifier
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': 'balanced',
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 0.2,
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.01,
 'min_samples_split': 0.2442105263157895,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': 1,
 'oob_score': True,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}

### Improvements for AutoML

1. Increase experiment timeout to allow for model experimentation.
2. Remove some features from our dataset that are collinear or not important in making the decision.

AutoML Widget![image](https://user-images.githubusercontent.com/68206315/115097539-dec3a400-9f22-11eb-8845-d65192d2f11c.png)

Best Model![image](https://user-images.githubusercontent.com/68206315/115098189-78408500-9f26-11eb-9a26-ee8180c1a89e.png)

![image](https://user-images.githubusercontent.com/68206315/115098550-82fc1980-9f28-11eb-90af-c522fe768ccf.png)

Best_model_run![image](https://user-images.githubusercontent.com/68206315/115098755-a8d5ee00-9f29-11eb-8034-fa1edec121fb.png)


## Hyperparameter Tuning

We use the SKLearn inbuilt Support Vector Machines (SVMs) for classification since it is capable of generating non-linear decision boundaries, and can achieve high accuracies. It is also more robust to outliers than Logistic Regression. This algorithm is used with the Azure ML HyperDrive service for hyperparameter tuning.

The hyperparameters tuned were the inverse regularization strength -C and the and kernel type -kernel with the search space defined for C as ```[0.5,1.0]``` and kernel as ```[linear,rbf,poly,sigmoid]```. We used Random Parameter Sampling method to sample over discrete kernel types and returns a C value whose logarithm is uniformly distributed. Random sampling can serve as a benchmark for refining the search space to improve results.

Parameter search space and Hyperdrive configuration.

```ruby
param_sampling = RandomParameterSampling( {
        "--kernel": choice('linear', 'rbf', 'poly', 'sigmoid'),
        "--C": choice(0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.7, 1.0, 1.3, 1.7,  2.0),
        "--C": loguniform(0.5, 1.0)
})


hyperdrive_run_config = HyperDriveConfig(run_config=estimator,
                                         hyperparameter_sampling=param_sampling,
                                         policy=early_termination_policy,
                                         primary_metric_name='AUC_weighted',
                                         primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                         max_total_runs=20,
                                         max_concurrent_runs=5)
```

We applied a <b>bandit</b> early termination policy to evaluate our benchmark metric (AUC_weighted). The policy is chosen based on slack factor, avoids premature termination of first 5 runs, and then subsequently terminates runs whose primary metric fall outside of the top 10%. This helps to stop the training process after it starts degrading the AUC_weighted with increased iteration count, thereby improving computational efficiency.

### Results

The SVM model achieved an AUC value of ```0.8166666666666667``` with the following parameters:

Hyperparameter | Value |
 | ------------- | -------------
Regularization Strength (C) | 1.862243408964539
Kernel | sigmoid

### Improvements for Hyperparameter Tuning

* We could improve this model by performing more feature engineering during data preparation phase.
* Adding more hyperparameters to be tuned can increase the model performance.
* Increasing max total runs to try a lot more combinations of hyperparameters, though this could have an impact on cost and training duration. 

Hyperdrive Run Widget provides information about logs recorded in the Run![image](https://user-images.githubusercontent.com/68206315/115134102-50741e80-a005-11eb-9260-1d1db8053fd3.png)

![image](https://user-images.githubusercontent.com/68206315/115134152-a779f380-a005-11eb-8cc5-f9738f7ec43a.png)

Hyperdrive experiment in Completed state with AUC value for each iteration![image](https://user-images.githubusercontent.com/68206315/115134142-903b0600-a005-11eb-9f74-35097de4e92b.png)

Best model: After the successfuly run of the experiment, we have the best model with kernel type as Signoid and C value of 1.86![image](https://user-images.githubusercontent.com/68206315/115134297-e6f50f80-a006-11eb-8f5c-5397adbc5ee5.png)

## Automated ML and Hyperparameter Tuning Comparison

Key | AutoML | Hyperdrive 
 | ------------- | ------------- | ------------- 
AUC_weighed | VotingEnsemble with 0.9226 | SVM with 0.8167
Duration | 42.13 minutes | 88.73 minutes

As shown in diagram, the VotingEnsemble model of AutoML performed better with an AUC value of 0.9226 compared to 0.8167 in Support Vector Machines through HyperDrive. So we will deploy the AutoML model.

## Model Deployment

The following steps are required to deploy a model using Azure SDK:

1. Register the dataset using SDK
2. Find the best model using Automl
3. Use the environment of automl's best_run or create a custom environment
4. Use the score.py file generated when the model is trained for deployment and evaluation. The scoring script describes the input data the model endpoint accepts. 
5. Deploy the model using any of the deployment choices - ACI, AKS or local. For our project, we deploy the model as webservice using Azure Container Instance with ```cpu_cores = 1```, ```memory_gb = 1``` and application insights enabled.
6. For inferencing, pass the sample test data in json format to model endpoint to test the webservice. This will be processed by the score.py file to make successful rest api call.

#### Deployed model

Successful model deployment using ACI (Azure Container Instance) and enable Application Insights![image](https://user-images.githubusercontent.com/68206315/115101958-ff4e2700-9f3f-11eb-9b73-2352690097a3.png)

Sample input data to query the endpoint

```ruby
data = {
    "data":
    [
        {
            'Age':75,
            'anaemia':0,
            'creatinine_phosphokinase':582,
            'diabetes':0,
            'ejection_fraction':20,
            'high_blood_pressure':1,
            'platelets':265000,
            'serum_creatinine':1.9,
            'serum_sodium':130,
            'sex':1,
            'smoking':0
        }
    ]
}
```

Sampled input data![image](https://user-images.githubusercontent.com/68206315/115102101-d1b5ad80-9f40-11eb-96fd-7920d3e71764.png)

Response from webservice: When we make an API call to our endpoint with sample data, we will see the inference output of the model.![image](https://user-images.githubusercontent.com/68206315/115102221-599bb780-9f41-11eb-89cd-3337f166c749.png)

## Screen Recording

https://youtu.be/kQ1-Q0g_2BE

## Future Improvements

1. A better performing AutoML model can be detected if the experiment timeout is increased.

2. Addressing the dataset imbalance by applying Synthetic Minority Oversampling Technique (SMOTE) can improve the performance of Hyperdrive model.

3. Converting the model into platform supported formats such as ONNX, TFLITE etc. will help optimize inference or model scoring and achieve scalability.

## Standout Suggestions

Enabled application insights during model deployment in order to log useful data about the requests being sent to the webservice.

## Citation

Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020) [Article](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec13).
