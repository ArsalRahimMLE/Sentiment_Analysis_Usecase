from sklearn.metrics import accuracy_score, recall_score, precision_score
import pickle
from sklearn.linear_model import LogisticRegression


def fetch_model(model_type='logistic regression'):
    """
    load model from pickle file
    :return:
    """
    model = pickle.load(open("staticfiles/model.pkl", "rb"))
    return model


def calculate_accuracy(ypred, y_test):
    """
    Calculate accuracy of model
    :param ypred:
    :param y_test:
    :return:
    """
    accuracy = accuracy_score(ypred, y_test)
    return accuracy


def calculate_precision(ypred, y_test):
    """
    Calculate accuracy of model
    :param ypred:
    :param y_test:
    :return:
    """
    precision = precision_score(ypred, y_test)
    return precision


def calculate_recall(ypred, y_test):
    """
    Calculate recall of model
    :param ypred:
    :param y_test:
    :return:
    """
    recall = recall_score(ypred, y_test)
    return recall


