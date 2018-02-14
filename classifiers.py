#import tensorflow as tf
import csv
import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


import itertools

TRAIN_FILE = '../train.feat'
DEV_FILE = '../dev.feat'
TEST_FILE = '../test.feat'
COLUMNS = ("doc_id", "part_num","sent_num",
            "word_num", "word", "pos",
            "parse_bit", "pred_lemma", "pred_frame_id",
            "sense","speaker","ne","corefs"
            )

FEATURES = ("word", "pos")

"""
FEATURES = ("word", "pos",
            "parse_bit", "pred_lemma", "pred_frame_id",
            "sense","speaker","ne"
            )
"""

LABEL = "corefs"

def prepare_data(data_source):
    with open(data_source) as source:
        X = list()
        y = list()
        reader = csv.DictReader(source)
        for row in reader:
            feats = dict()
            for feature in FEATURES:
                feats[feature] = row[feature]
            corefs = row['corefs']
            if corefs != '-':
                first_coref = corefs.split('|')[0]
                label = ''.join([char for char in first_coref if char.isdigit()])
            else:
                label = '-'
            X.append(feats)
            y.append(label)
    return X,y

def train(X_train, y_train):
    """Train CRF model
    Inputs:
        X_train: list of feature dicts for training set
        y_train: list of labels for training set
    Returns:
        model: trained CRF model
    """
    
    model = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(crf, X_dev, y_dev):
    """Evaluate the model
    Inputs:
        crf: trained CRF model
        X_dev: list of feature dicts for dev set
        y_dev: list of labels for dev set
    Returns:
        None (prints metrics)
    """

    #Get the labels we're evaluating
    labels = list(crf.classes_)
    print(labels)
    #Most labels are 'O', so we ignore,
    #otherwise our scores will seem higher than they actually are.
    #labels.remove('-')

    print("Predicting labels")
    y_pred = crf.predict(X_dev)

    #print(y_pred[:10]) #for debugging
    #print(y_dev[:10]) #for debugging

    print("Displaying accuracy")
    metrics.flat_f1_score(y_dev, y_pred,
                      average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print("Displaying detailed metrics")
    print(metrics.flat_classification_report(
        y_dev, y_pred, labels=sorted_labels, digits=3
    ))


if __name__ == "__main__":
    pass
