from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Datasets/Glass.txt', sep='	').astype(float)

X = data.iloc[:, 0:9]
y = data.iloc[:, 9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# t={21, 31, 41, 51}

def AdaBoost(X, y, T=51, learning_rate=0.1):  # T----> #learner
    # Initialization of variables
    M = len(y)  # sample
    model_list, y_hat_list, model_error_list, model_weight_list, sample_weight_list = [], [], [], [], []

    # Initialize the sample weights
    sample_weight = np.ones(M) / M
    sample_weight_list.append(sample_weight.copy())

    # For t = 1 to T
    for t in range(T):  # weak learner
        # Fit a classifier
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y, sample_weight=sample_weight)
        y_predict = model.predict(X)

        # Mis classifications
        incorrect = (y_predict != y)

        # model error
        model_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        #  model weights
        model_weight = learning_rate * np.log((1. - model_error) / model_error)

        # Boost sample weights
        sample_weight *= np.exp(model_weight * incorrect * ((sample_weight > 0) | (model_weight < 0)))

        # Save iteration values
        model_list.append(model)
        y_hat_list.append(y_predict.copy())
        model_error_list.append(model_error.copy())
        model_weight_list.append(model_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Convert to np array for convenience
    model_list = np.asarray(model_list)
    y_hat_list = np.asarray(y_hat_list)
    model_error_list = np.asarray(model_error_list)
    model_weight_list = np.asarray(model_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # Predictions
    preds = (np.array([np.sign((y_hat_list[:, point] * model_weight_list).sum()) for point in range(M)]))
    print('Accuracy = ', (preds == y).sum() / M)

    return model_list, model_weight_list, sample_weight_list


model_list, model_weight_list, sample_weight_list = AdaBoost(X, y, T=10, learning_rate=0.1)


def AdaBoost_classify(x, model, model_weights):  # classification  X that fitted AdaBoost

    y_hat = np.asarray([(e.predict(x)).T * w for e, w in zip(model, model_weights)]) / model_weights.sum()
    return np.sign(y_hat.sum(axis=0))


model_list, model_weight_list, sample_weight_list = AdaBoost(X, y, T=10, learning_rate=1)


# noisy=================================


def make_noisy(X, y, Rn, Rc):
    # make copy from input
    X = X.copy()

    # number of training data and number of features
    n_samples, n_features = X.shape

    # number of features that will selected
    dv = np.floor(n_features * Rn).astype(np.int)

    # selected features
    features = np.random.choice([n for n in range(0, n_features)], dv)

    # for each feature
    for feature in features:
        # selected feature
        f_i = X[:, feature]

        # find range of min and max of a feature
        x = np.arange(np.min(f_i), np.max(f_i) + 1)

        # a little bit more/less than max/min
        xU, xL = x + 0.5, x - 0.5

        # number of cases
        Nc = np.floor(n_samples * Rc).astype(np.int)

        # cases
        cases = np.random.uniform(0, n_samples, Nc).astype(np.int)

        # set new values for selected cases
        X[cases, feature] = np.random.choice(x, Nc)
        model = AdaBoost(X, y, T=10, learning_rate=1)
        # Test Model
        predicted = model.predict(y)

    # Accuracy
    accuracy = accuracy_score(y_true=y, y_pred=predicted)

    print("accuracy with noise:", accuracy)
    return x
