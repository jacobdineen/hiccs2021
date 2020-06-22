#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference:

#from __future__ import print_function
#from utils.heaton_utils import *

import itertools as it
import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt  # for plotting
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.display import HTML, display, display_html
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split)



def load_locally(file_name, dev_type = 'stochastic'):
    file = open(file_name,'rb')
    temp = pickle.load(file)
    if file_name == './data/gridsearch.pkl':

        temp.model_paths = {5000: [('RFC',
                                f'./models/5000/RFC.pkl'),
                                ('GBC', './models/5000/GBC.pkl'),
                                ('LR', './models/5000/LR.pkl')],
                                10000: [('RFC',
                                './models/10000/RFC.pkl'),
                                ('GBC', './models/10000/GBC.pkl'),
                                ('LR', './models/10000/LR.pkl')],
                                20000: [('RFC',
                                './models/20000/RFC.pkl'),
                                ('GBC', './models/20000/GBC.pkl'),
                                ('LR', './models/20000/LR.pkl')],
                                40000: [('RFC',
                                './models/40000/RFC.pkl'),
                                ('GBC', './models/40000/GBC.pkl'),
                                ('LR', './models/40000/LR.pkl')],
                                80000: [('RFC',
                                './models/80000/RFC.pkl'),
                                ('GBC', './models/80000/GBC.pkl'),
                                ('LR', './models/80000/LR.pkl')]}
    
    if file_name == './data_stochastic/gridsearch.pkl':
        temp.model_paths = {20000: [('RFC',
                                f'./models_stochastic/20000-0/RFC.pkl'),
                                ('GBC', './models_stochastic/20000-0/GBC.pkl'),
                                ('LR', './models_stochastic/20000-0/LR.pkl')],
                                20001: [('RFC',
                                './models_stochastic/20000-1/RFC.pkl'),
                                ('GBC', './models_stochastic/20000-1/GBC.pkl'),
                                ('LR', './models_stochastic/20000-1/LR.pkl')],
                                20002: [('RFC',
                                './models_stochastic/20000-2/RFC.pkl'),
                                ('GBC', './models_stochastic/20000-2/GBC.pkl'),
                                ('LR', './models_stochastic/20000-2/LR.pkl')],
                                20003: [('RFC',
                                './models_stochastic/20000-3/RFC.pkl'),
                                ('GBC', './models_stochastic/20000-3/GBC.pkl'),
                                ('LR', './models_stochastic/20000-3/LR.pkl')],
                                20004: [('RFC',
                                './models_stochastic/20000-4/RFC.pkl'),
                                ('GBC', './models_stochastic/20000-4/GBC.pkl'),
                                ('LR', './models_stochastic/20000-4/LR.pkl')]}
    return temp


#Note, this code is taken straight from the SKLEARN website, a nice way of viewing confusion matrix.
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Note: class is a listlike parameter. Pass in list of classes, eg: ["No Loan", "Loan"]
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        value = '{0:.2g}'.format(cm[i, j])
        plt.text(j,
                 i,
                 value,
                 fontsize=10,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')





def display_sklearn_feature_importance(gs_object, models, num_obs, n_features):
    '''
    Parameters:
    data: data object; coomatrix w/ encoded features
    n_features: number of features to visualize
    set: str;
        'lendingclub' - load lending club models
        'uci' - load uci models
    Returns:
    Graph of basic feature importance measurements
    '''
    
    X_train, X_test, y_train, y_test, encoder, unrolled_features = gs_object.objects[num_obs]
    rfc, gbc, logit = [i[1] for i in models[num_obs]]
    
    feature_importance = pd.DataFrame({
        "feature":
        unrolled_features,
        "RF_Feature_Importance":
        np.round(rfc.feature_importances_, 4),
        "GBC_Feature_Importance":
        np.round(gbc.feature_importances_, 4),
        "Logit_Coeff":
        np.round(logit.coef_[0], 4),
        "Max_Feature_Val":
        pd.DataFrame(X_train.toarray(), columns=unrolled_features).max(),
    })

    n = n_features
    
    feature_importance['coeff_max'] = feature_importance[
        'Logit_Coeff'] * feature_importance['Max_Feature_Val']
    temp = feature_importance.nlargest(n, 'RF_Feature_Importance')
    sns.barplot(temp['RF_Feature_Importance'], temp['feature'])
    plt.title('Random Forest - Feature Importance Top {}'.format(n_features))
    plt.show()

    temp = feature_importance.nlargest(n, 'GBC_Feature_Importance')
    sns.barplot(temp['GBC_Feature_Importance'], temp['feature'])
    plt.title('Gradient Boosted Classifier - Feature Importance Top {}'.format(
        n_features))
    plt.show()

    #We want to show the total possible feature impact here. Take the max of each feature in the training set by the logit coeff.
    lookup = pd.DataFrame(X_train.toarray(), columns=unrolled_features).max()
    temp = feature_importance.nlargest(int(n / 2), 'coeff_max')
    temp1 = feature_importance.nsmallest(int(n / 2), 'coeff_max')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['coeff_max'], temp['feature'])
    plt.title('Logistic Regression - Coefficients Top&Bottom {}'.format(
        int(n_features / 2)))
    plt.show()
    


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'),
                 raw=True)
    
def neg_pos_logit_coefficients(model, gs_object, num_obs):
    
    X_train, X_test, y_train, y_test, encoder, unrolled_features = gs_object.objects[num_obs]
    
    logistic_regress_coeff = pd.DataFrame({
        "features": unrolled_features,
        "Coef": model.coef_[0]
    })

    neg_coef = round(logistic_regress_coeff[
        logistic_regress_coeff['Coef'] < 0].sort_values('Coef', ascending=True),2).head(15)
    pos_coef = round(logistic_regress_coeff[
        logistic_regress_coeff['Coef'] > 0].sort_values('Coef', ascending=False),2).head(15)
    
    display_side_by_side(neg_coef, pos_coef)
    


def fetch_models(sample_size, gs_object):
    '''
    gs object is of class GridSearch
    sample size is a scalar
    '''
    models = []
    for k,v in gs_object.model_paths.items():
        for i in gs_object.model_paths[k]:
            if str(sample_size) in i[1]:
                file = open(i[1],'rb')
                models.append((i[0], pickle.load(file)))
    return models

def load_models_to_mem(size, gs_object):
    '''
    gs object is of class GridSearch
    sample size is a list of scalars
    '''
    model_dict = {}
    for i,j in gs_object.model_paths.items():
        models = []
        for k in j:
            file = open(k[1],'rb')
            models.append((k[0], pickle.load(file)))
        model_dict[i] = models
    print('models loaded into memory')
    return model_dict

def test_accuracy(size, gs_object, model_dict):
    '''
    gs object is of class GridSearch
    sample size is a list of scalars
    model_dict contains string references and model architecture
    Example:
        from utils.helpful_util import fetch_models, load_models_to_mem, test_accuracy

        model_dict = load_models_to_mem(size, gs_object=gs)
        scores = test_accuracy(size, gs, model_dict)
    '''
    scores = []
    for i in size:
        X_train, X_test, y_train, y_test, encoder, unrolled_features = gs_object.objects[i]
        for j in model_dict[i]:
            name = j[0]
            preds = j[1].predict(X_test)
            accuracy = sklearn.metrics.accuracy_score(preds, y_test)
            score = (name, i, accuracy)
            scores.append(score)
            # print(f'Accuracy on test set for {name}, size {i} = {accuracy:.2f}')
    # print('-' * 50)
    return scores

def test_accuracy_graph(test_acc_scores):
    sns.set(style="whitegrid")
    g = sns.catplot(x="model",
                    y="test_acc",
                    hue="sample",
                    data=test_acc_scores,
                    height=6,
                    kind="bar",
                    palette="muted",
                    legend=False)
    g.despine(left=True)
    g.set_ylabels("Test Accuracy")
    g.ax.legend(loc='best')
    plt.show()