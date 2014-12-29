##### More elaborate code to get results for classification #####
##### Includes plotting diagnostics #####

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from RichertUtils import *      #need to import the plot utility functions from Richert

def measure(clf_class, parameters, name, X, Y, data_size=None, plot=True, verbose=False):

    if data_size is None:
        X = X    
        Y = Y
    else:
        X = X[:data_size]
        Y = Y[:data_size]

    cv = KFold(n=len(X), n_folds=3) #Indices=True 

    train_errors = []
    test_errors = []

    scores = []
    roc_scores = []
    fprs, tprs = [], []

    pr_scores = []
    precisions, recalls, thresholds = [], [], []

    for train, test in cv:
        X_train, y_train = X.iloc[train, :].values, Y.iloc[train].values
        X_test, y_test = X.iloc[test, :].values, Y.iloc[test].values

        clf = clf_class(**parameters)

        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        proba = clf.predict_proba(X_test)

        label_idx = 1
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, label_idx])
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba[:, label_idx])

        roc_scores.append(auc(fpr, tpr))
        fprs.append(fpr)
        tprs.append(tpr)

        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)
        if verbose == True:
            print(classification_report(y_test, proba[:, label_idx] > 
                                        0.63, target_names=['not accepted', 'accepted'])) #what is this 0.63?

    # get medium clone
    scores_to_sort = pr_scores  # roc_scores
    medium = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
    
    #if plot:
        #plot_roc(roc_scores[medium], name, fprs[medium], tprs[medium])
        #plot_pr(pr_scores[medium], name, precisions[medium], recalls[medium])

    avg_scores_summary = []
    
    summary = (name,
               np.mean(scores), np.std(scores),
               np.mean(roc_scores), np.std(roc_scores),
               np.mean(pr_scores), np.std(pr_scores))
    
    print(summary)
    avg_scores_summary.append(summary)
    precisions = precisions[medium]
    recalls = recalls[medium]
    thresholds = np.hstack(([0], thresholds[medium]))
    idx80 = precisions >= 0.8
    print("P=%.2f R=%.2f thresh=%.2f" % (precisions[idx80][0], recalls[
          idx80][0], thresholds[idx80][0]))

    return np.mean(train_errors), np.mean(test_errors)

def bias_variance_analysis(clf_class, parameters, name, X, Y, data_sizes, verbose=False):
    
    train_errors = []
    test_errors = []

    for data_size in data_sizes:
        train_error, test_error = measure(
            clf_class, parameters, name, X, Y, data_size, verbose)
        train_errors.append(train_error)
        test_errors.append(test_error)
    return train_errors, test_errors

def k_complexity_analysis(clf_class, parameters, X, Y, ks, data_size, verbose=False):
    
    train_errors = []
    test_errors = []

    for k in ks:
        parameters['n_neighbors'] = k
        train_error, test_error = measure(
            clf_class, parameters, "%dNN" % k, X, Y, data_size)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_k_complexity(ks, train_errors, test_errors)
