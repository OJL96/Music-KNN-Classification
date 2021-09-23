"""
I, Omar LInes, have read and understood the School's Academic Integrity Policy,
as well as guidance relating to this module, and confirm that this submission
complies with the policy.

The content of this file is my own original work, with any significant material
copied or adapted from other sources clearly indicated and attributed.
"""


import numpy as np

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import kendalltau

# keeps rng consistent; aids in hyperparameter tunning and debugging.
np.random.seed(123)

""" FUNCTIONS """


def feature_selection(features, label_vector):

    correlation_vector = []
    irrelvent_features_index = []

    for i in range(features.shape[1]):
        # using Kendall Tau's correlation
        corr, _ = kendalltau(features[:, i], label_vector)
        correlation_vector.append(corr)

        # range set to pick up features with no correlation.
        if corr < 0.10 and corr < -0.10:
            irrelvent_features_index.append(i)

    return irrelvent_features_index


# calculates the Euclidean Distance between two vectors.
def euclidean_distance(train_vector, pred):

    distance = 0
    for i in range(len(train_vector)):
        distance += (train_vector[i] - pred[i])**2

    # square root without math libary: no change in execution time.
    # i felt that importing the math libary for one line was a waste.
    # personal preference
    return distance**0.5


# Voting system with the Euclidean Distance function integrated.
def get_neighbors(train, test_row, num_neighbors):

    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append([train_row, dist])

    # distances sorted from smallest to largest.
    distances.sort(key=lambda s: s[1])

    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


# find label that has the most votes
def knn_pred(train, test_row, num_neighbors):

    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]

    unique, counts = np.unique(output_values, return_counts=True)

    # if vote is a tie, class with closest distance is chosen.
    predication = unique[np.argmax(counts)]
    return predication


# main function for knn
def k_nearest_neighbors(train, test, num_neighbors=3):

    predictions = []
    for row in test:
        output = knn_pred(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)


# performance measure of KNN: accuracy metric.
def performance_measure(pred, actual):

    score = []
    [score.append(1) for i in range(len(pred)) if pred[i] == actual[i]]
    score = sum(score) / len(pred)
    return score


# plot hyperparameter of k value results.
def plot_tuning_results(max_k, k_scores):

    colors = cm.rainbow(np.linspace(0, 1, max_k))
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot([t+1 for t in range(max_k)], k_scores, color="black")

    temp = 0
    for t in range(max_k):
        ax.scatter(t+1,
                   k_scores[t],
                   color=colors[t],
                   label=("K-value {}".format(t+1)), s=80)

        if temp < k_scores[t]:
            optimal_k = t+1
            temp = k_scores[t]

    ax.scatter(optimal_k,
               max(k_scores),
               s=1000,
               linewidths=3,
               facecolors='none',
               edgecolors='black')
    plt.title("Hyperparameter Tunning: K parameter")
    plt.text(optimal_k,
             max(k_scores)+0.75,
             "Best K: {} \nAccuracy: {:.2f}%".format(optimal_k, max(k_scores)))
    ax.set_xlabel("K-value (N)")
    ax.set_ylabel("Accuracy Score (%)")
    plt.ylim(min(k_scores)-2.5, max(k_scores)+2.5)
    ax.legend(loc="lower left", ncol=5)


# hyparameter tuning function.
def hyperparameter_tuning(train, X_test, y_test, max_k):

    k_scores = []
    # setting values to find optimal k
    parameter_range = range(1, max_k+1)

    for i in parameter_range:
        result = k_nearest_neighbors(train, X_test, i)

        accuracy = performance_measure(result, y_test)
        k_scores.append(accuracy*100)
        print("Testing K value: {}".format(i))

    optimal_k_value = np.argmax(k_scores)+1
    print("Optimal K: {}".format(optimal_k_value))

    # visualise performance based on each k value tested.
    plot_tuning_results(max_k, k_scores)

    # return best k value, to be used for fixed validation and cross validation
    return optimal_k_value


# train test split. i did it manually as its simple
def train_test_split(dataset):

    # we randomise the dataset before splitting. So each kfold is different.
    np.random.shuffle(dataset)
    # 0.8 value = 80% split.
    X_train = np.array(dataset[0:int(len(dataset)*0.8), -1])
    y_train = np.asarray(dataset[0:int(len(dataset)*0.8), -1]).reshape(-1, 1)
    X_test = np.array(dataset[len(X_train):, :-1])
    y_test = np.asarray(dataset[len(X_train):, -1]).reshape(-1, 1)

    train = np.concatenate([X_train, y_train], axis=1)

    return train, X_test, y_test


def fixed_validation(dataset):

    # splitting data manually: 80% train, 20% test
    train, X_test, y_test = train_test_split(dataset)

    optimal_k = hyperparameter_tuning(train, X_test, y_test, 10)
    start_time = time.time()
    pred = k_nearest_neighbors(train, X_test, optimal_k)
    accuracy = performance_measure(pred, y_test)

    print("Fixed Validation Score: {:.2f}%".format(accuracy))
    print("KNN completion time: {:.2f}s\n".format(time.time() - start_time))

    # imported sklearn to save time
    from sklearn.metrics import classification_report
    print(classification_report(y_test, pred))

    # runs confusion matrix function for displaying it nicely
    confusion_matrix(y_test, pred)

    return optimal_k


# cross validaiton function to test the KNN's generalising of unseen data.
def cross_validation(dataset, optimal_k):

    k_folds = 10
    knn_scores = []
    for i in range(k_folds):

        train, X_test, y_test = train_test_split(dataset)

        start_time = time.time()
        pred = k_nearest_neighbors(train, X_test, optimal_k)
        accuracy = performance_measure(pred, y_test)
        knn_scores.append(accuracy*100)

        print("KNN (fold: {}) Score: {:.2f}%".format(i+1, accuracy*100))
        print("Completion time: {:.2f}s \n".format(time.time() - start_time))

    print("k_fold average: {}%".format(sum(knn_scores) / k_folds))

    plot_knn_results(10, knn_scores)


# plots confusion matrix with sklearn library and seaborn library.
def confusion_matrix(y_test, pred):
    # imported sklearn and pandas to save time
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import seaborn as sb
    import os
    labels = []
    [labels.append(i) for i in os.listdir(os.getcwd()+"\\genres\\")]

    data = confusion_matrix(y_test, pred)
    df_cm = pd.DataFrame(data, columns=labels, index=labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(18, 12))
    heatMap = sb.heatmap(df_cm,
                         cmap="Blues",
                         annot=True,
                         annot_kws={"size": 18})
    heatMap.get_figure()


# plots each fold's performance score.
def plot_knn_results(k_folds, k_fold_scores):

    colors = cm.rainbow(np.linspace(0, 1, k_folds))
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot([t+1 for t in range(k_folds)], k_fold_scores, color="black")
    ax.axhline(y=(sum(k_fold_scores) / len(k_fold_scores)),
               color='g', linestyle="--", label="Mean")

    for t in range(k_folds):
        ax.scatter(t+1,
                   k_fold_scores[t],
                   color=colors[t],
                   label="K-fold {}".format(t+1), s=80)

    plt.title("K-fold Evaluation")
    ax.set_xlabel("K-fold Value (N)")
    ax.set_ylabel("Accuracy Score (%)")
    plt.ylim(min(k_fold_scores)-2.5, max(k_fold_scores)+2.5)
    ax.legend(loc="lower left", ncol=5)


""" MAIN BODY """
if __name__ == '__main__':

    # Features extracted
    X = np.load("Extracted Features.npy")

    # Encoded labels
    Y = np.load("Encoded Labels.npy").reshape(-1, 1)

    # normalize feature matrix.
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # Since feature selection has no positive impact, i let user decided
    # whether to run feature selection function.
    user_input = input("Enable feature selection? Y/n>").upper()
    if user_input == 'Y':
        # removing irrelevant features
        features_to_delete = feature_selection(X, Y)
        X = np.delete(X, features_to_delete, 1)

    # column-wise concating feature matrix and label vector
    dataset = np.concatenate([X, Y], axis=1)

    # fixed validation used for assess parameters. Returns optimal parameter.
    optimal_k = fixed_validation(dataset)

    # run cross validation with parameters from fixed validation.
    cross_validation(dataset, optimal_k)

else:
    # only runs pre_processing.py when im not running from the main.py
    from pre_processing import run_pre_processing
    run_pre_processing()
