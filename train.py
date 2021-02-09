from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://raw.githubusercontent.com/PeacePeters/Deploy-the-best-model-using-AzureML/main/heart_failure.csv"

data_url = 'https://raw.githubusercontent.com/PeacePeters/Deploy-the-best-model-using-AzureML/main/heart_failure.csv'
ds = TabularDatasetFactory.from_delimited_files(path=data_url)
    
# Get the experiment run context
run = Run.get_context()


def clean_data(data)
    df = data.to_pandas_dataframe().dropna()

    # df["UnderCon"] = 0
    # df.loc[((df["anaemia"] == 1) | (df["diabetes"] == 1) | df["high_blood_pressure"] == 1), "UnderCon"] = 1
    # df.drop(["anaemia", "diabetes", "high_blood_pressure"], axis = 1, inplace = True)
    
    # corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
    x_df = df.drop(["DEATH_EVENT", "time"], axis=1)
    # x_df = df.drop(["creatinine_phosphokinase", "platelets", "sex", "smoking", "time", "DEATH_EVENT"], axis=1)
    # x_df = df.drop(["anaemia", "creatinine_phosphokinase", "diabetes", "high_blood_pressure", "platelets", "sex", "smoking", "time", "DEATH_EVENT"], axis=1)
   
    y_df = df["DEATH_EVENT"]
    
    return x_df,y_df


#def train_classifier(clf, X_train, Y_train):
    #start = time()
    #model=clf.fit(X_train, Y_train)
    #end = time()
    #print ("Trained model in {:.4f} seconds".format(end - start))
    #return model


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    #parser.add_argument('--kernel', type=str, default='linear', help='Kernel type to be used in the algorithm')
    #parser.add_argument('--coef0', type=int, default=0, help="Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    #run.log("Kernel:", np.str(args.kernel))
    #run.log("coef0:", np.int(args.coef0))
    run.log("Max iterations:", np.int(args.max_iter))

    # loading the dataset
    x, y = clean_data(ds)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # sc=StandardScaler()
    # x_train=sc.fit_transform(x_train)

    # Training a SVM classifier
    #clf = SVC(C=args.C,coef0=args.coef0)
    #clf = SVC(C=args.C,kernel=args.kernel)
    #model = train_classifier(clf, x_train, y_train)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    AUC_weighted = model.score(x_test, y_test)
    #AUC_weighted = model.score(sc.transform(x_test), y_test)
    run.log("AUC_weighted", np.float(AUC_weighted))

    # Save the trained model in the outputs folder
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/hyperdrive_model.joblib')

  
if __name__ == '__main__':    
    main()
