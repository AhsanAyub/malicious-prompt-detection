#!/usr/bin/env python3

__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Majumdar, Subho"]
__email__ = "md.ahsanayub@outlook.com"
__status__ = "Prototype"

import pandas as pd
#from ast import literal_eval

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def process_dataset(filename: str):
    dataset = pd.read_pickle(filename)
    dataset = dataset.dropna()
    test_indices = pd.read_csv("dataset/test_indices.csv")
    test_indices = test_indices["id"].tolist()
    
    train = dataset[~dataset["id"].isin(test_indices)]
    test = dataset[dataset["id"].isin(test_indices)]
    del dataset
    print("train and test split completed")

    Y_train = train.iloc[:, -1].values
    Y_train = train.iloc[:, -1]
    Y_train.to_pickle("dataset/openai_Y_train.pkl",protocol=3)
    train["text_embedding"] = train.text_embedding.apply(lambda x: eval(str(x)))
    X_train = train.text_embedding.apply(pd.Series)
    X_train.to_pickle("dataset/openai_X_train.pkl",protocol=4)
    X_train = X_train[:,:].values
    del train
    print("training set completed")
    
    Y_test = test.iloc[:, -1].values
    Y_test = test.iloc[:, -1]
    Y_test.to_pickle("dataset/openai_Y_test.pkl",protocol=3)
    test["text_embedding"] = test.text_embedding.apply(lambda x: eval(str(x)))
    X_test = test.text_embedding.apply(pd.Series)
    X_test.to_pickle("dataset/openai_X_test.pkl",protocol=4)
    X_test = X_test[:,:].values
    del test
    print("testing set completed")

    return X_train, X_test, Y_train, Y_test


# Driver program
if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = process_dataset("openai_master.pkl")

    X_train = pd.read_pickle("dataset/openai_X_train.pkl")
    print(X_train.shape)
    X_train = X_train.iloc[:,:].values
    X_test = pd.read_pickle("dataset/openai_X_test.pkl")
    X_test = X_test.iloc[:,:].values
    Y_train = pd.read_pickle("dataset/openai_Y_train.pkl")
    print(Y_train.shape)
    Y_train = Y_train.iloc[:].values
    print(len(Y_train))
    Y_test = pd.read_pickle("dataset/openai_Y_test.pkl")
    Y_test = Y_test.iloc[:].values

    # Fitting the classifier into training set
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)
    
    # Breakdown of statistical measure based on classes
    Y_pred = classifier.predict(X_test)
    print(classification_report(Y_test, Y_pred, digits=4))
    
    # Compute the model's performance
    df = pd.DataFrame(list(zip(Y_test,Y_pred)), columns =['Y_test', 'Y_pred'])
    df.to_csv("results/openai_logistic_regression.csv", index=False)

    # Fitting the classifier into training set
    classifier = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=0)
    classifier.fit(X_train, Y_train)
    
    # Breakdown of statistical measure based on classes
    Y_pred = classifier.predict(X_test)
    print(classification_report(Y_test, Y_pred, digits=4))
    
    # Compute the model's performance
    df = pd.DataFrame(list(zip(Y_test,Y_pred)), columns =['Y_test', 'Y_pred'])
    df.to_csv("results/openai_random_forest.csv", index=False)

    # Fitting the classifier into training set
    classifier = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    classifier.fit(X_train, Y_train)
    
    # Breakdown of statistical measure based on classes
    Y_pred = classifier.predict(X_test)
    print(classification_report(Y_test, Y_pred, digits=4))
    
    # Compute the model's performance
    df = pd.DataFrame(list(zip(Y_test,Y_pred)), columns =['Y_test', 'Y_pred'])
    df.to_csv("results/openai_xgb.csv", index=False)