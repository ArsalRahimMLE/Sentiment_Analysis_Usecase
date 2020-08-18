from ..Helping_modules import database, model
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import pandas as pd


def get_model_metrics():
    """
    Calcualate model metrics such as recall, precision etc
    :return: dictionary as response
    """
    # 01 fetch data #
    x_test = pd.read_csv("staticfiles/X_test_data.csv")
    y_test = pd.read_csv("staticfiles/y_test_data.csv")
    # 02 load model #
    rf = model.fetch_model_rf()
    # 03 run model on testing data #
    ypred = rf.predict(x_test)
    # 04 calculate model metrics #
    accuracy = model.calculate_accuracy(ypred, y_test)
    precision = model.calculate_precision(ypred, y_test)
    recall = model.calculate_recall(ypred, y_test)
    # 05 response to request #
    response = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    return response


def getting_sentiment():
#fetching data from database#
    positive_count = 0
    negative_count = 0

    df = database.fetch_data()
    total = len(df)
    for i in df.Sentiment:
        if i == 1:
            positive_count = positive_count + 1
        else:
            negative_count = negative_count+1
    pos_percentage = (positive_count/total)*100
    neg_percentage = (negative_count/total)*100
    response = {'Positive Sentiments': positive_count, 'Negative Sentiments': negative_count,'Positive':pos_percentage, 'Negative':neg_percentage}
    return response

def display_reviews():
    df = database.fetch_data()
    review_sent = df[['Review Text','Sentiment']]
    first = review_sent.loc[1][0]
    second = review_sent.loc[2][0]
    third = review_sent.loc[15][0]
    fourth = review_sent.loc[17][0]
    fifth = review_sent.loc[5][0]
    response = (first, second, third, fourth, fifth)
    return response


def get_model_comparison():
    """
        Calcualate  accuracy of different models i.e. Logistic Regression, Naive Bayes, Random forest and SVM
        :return: dictionary as response
        """
    # 01 fetch data #
    x_test = pd.read_csv("staticfiles/X_test_data.csv")
    y_test = pd.read_csv("staticfiles/y_test_data.csv")
    # 02 load models #
    lr = model.fetch_model()
    nb = model.fetch_model_nb()
    rf = model.fetch_model_rf()
    svm = model.fetch_model_svm()
    # 03 run models on testing data #
    ypred = lr.predict(x_test)
    ypred_nb = nb.predict(x_test)
    ypred_rf = rf.predict(x_test)
    ypred_svm = svm.predict(x_test)
    # 04 calculating accuracies of different models #
    accuracy = model.calculate_accuracy(ypred, y_test)
    accuracy_nb = model.calculate_accuracy(ypred_nb, y_test)
    accuracy_rf = model.calculate_accuracy(ypred_rf, y_test)
    accuracy_svm = model.calculate_accuracy(ypred_svm, y_test)
    # 05 calculating precision of different models
    precision = model.calculate_precision(ypred, y_test)
    precision_nb = model.calculate_precision(ypred_nb, y_test)
    precision_rf = model.calculate_precision(ypred_rf, y_test)
    precision_svm = model.calculate_precision(ypred_svm, y_test)
    # 06 Calculating Recall for different models
    recall = model.calculate_recall(ypred, y_test)
    recall_nb = model.calculate_recall(ypred_nb, y_test)
    recall_rf = model.calculate_recall(ypred_rf, y_test)
    recall_svm = model.calculate_recall(ypred_svm, y_test)
    # 07 confusion matrix
    cm = model.display_confusion_matrix(ypred, y_test)
    cm_nb = model.display_confusion_matrix(ypred_nb, y_test)
    cm_rf = model.display_confusion_matrix(ypred_rf, y_test)
    cm_svm = model.display_confusion_matrix(ypred_svm, y_test)
    # 08 response to request #
    accuracies = {'Logistic Regression': accuracy, 'Naive Bayes': accuracy_nb,
                 'Random Forest': accuracy_rf, 'SVM ': accuracy_svm}
    precisions = {'Logistic Regression': precision, 'Naive Bayes': precision_nb,
                  'Random Forest': precision_rf, 'SVM ': precision_svm}
    recalls = {'Logistic Regression': recall, 'Naive Bayes': recall_nb,
                  'Random Forest': recall_rf, 'SVM = ': recall_svm}
    confusion_matrices = {'Logistic Regression': cm.tolist(), 'Naive Bayes': cm_nb.tolist(),
               'Random Forest': cm_rf.tolist(), 'SVM = ': cm_svm.tolist()}

    response = {'Accuracy': accuracies, 'Precision': precisions, 'Recall': recalls,'Confusion_matrix': confusion_matrices}
    return response
