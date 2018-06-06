import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model import standard_sc

def logreg_auc(training_instances, test_instances):

    # training instances
    training_instances = standard_sc(training_instances)
    training_X = np.array(training_instances.drop(columns = 'output'))
    training_y = np.array(training_instances['output'])

    # test instances
    test_instances = standard_sc(test_instances)
    test_X = np.array(test_instances.drop(columns='output'))
    test_y = np.array(test_instances['output'])

    # model
    logreg = LogisticRegression(C=1e5)
    logreg.fit(training_X, training_y)
    probs = logreg.predict_proba(test_X)[:, 0]

    test_y[test_y == 2] = 0

    return roc_auc_score(test_y, probs)

