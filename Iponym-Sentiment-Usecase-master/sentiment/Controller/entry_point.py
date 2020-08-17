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
    rf = model.fetch_model(model_type='random forest')
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
    first = review_sent.loc[1]
    second = review_sent.loc[2]
    third = review_sent.loc[15]
    fourth = review_sent.loc[17]
    fifth = review_sent.loc[5]
    response = (first,second,third,fourth,fifth)
    return response