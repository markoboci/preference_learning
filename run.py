import json
import pandas as pd
import numpy as np
import os

from prepare_data import prepare_training_test_datasets
from model import determine_alpha, custom_minimize, predict
from logreg_model import logreg_auc


# load params
#------------

with open('config.json', 'r') as infile:
    params = json.load(infile)

method = params['METHOD']
class_imbalance = params['CLASS_IMBALANCE']
n_orders = params['N_ORDERS']
datasets = params['DATASETS']
repetitions = params['REPETITIONS']
regularization = params['REGULARIZATION']
replace = params['WITH_REPLACEMENT']
training_test_imbalance = params['TRAINING_TEST_IMBALANCE']
reg_param = params['REG_PARAM']



# create output dir
#----------------
if not os.path.exists('output/'):
    os.makedirs('output/')


for dataset in datasets:

    plis_auc_values = []
    logreg_auc_values = []

    # create log file
    #----------------
    log_file_path = 'output/dataset=' + dataset + '_method=' + method + '_class_imbalance=' + str(class_imbalance) + \
                    '_n_pairs=' + str(n_orders) + '_reg=' + regularization + '_tt_imbalance=' + str(training_test_imbalance) + '.log.txt'
    log_file = open(log_file_path, 'w')

    log = 'DATASET: ' + dataset + '\n'
    log += 'METHOD: ' + method + '\n'
    log += 'NUMBER OF PAIRS (N_ORDERS): ' + str(n_orders) + '\n'
    log += 'CLASS IMBALANCE: ' + str(class_imbalance) + '\n'
    log += 'TRAINING/TEST IMBALANCE: ' + str(training_test_imbalance) + '\n'
    log += 'REPETITIONS: ' + str(repetitions) + '\n'
    log += 'WITH REPLACEMENT: ' + replace + '\n'
    if regularization == 'no':
        log += 'REGULARIZATION PARAM: ' + str(reg_param) + '\n'
        log += 'REGULARIZATION: ' + regularization + '\n\n'
    else:
        log += 'REGULARIZATION: ' + regularization + '\n\n'
    log_file.write(log)

    for i in range(repetitions):

        # preparing data and loading data
        # -------------------------------
        print("1. PREPARING AND LOADING DATA ...")
        dataset_path = 'data/' + dataset + '_parsed.csv'
        training_instances, test_instances, training_orders, test_orders = prepare_training_test_datasets(dataset_path, tt_imbalance=training_test_imbalance, replace = replace, n_orders=n_orders, class_imbalance=class_imbalance, random_seed=i)
        if i == 0:
            log = 'Number of instances in training set: ' + str(len(training_instances)) + '\n'
            log += 'Number of instances in test set: ' + str(len(test_instances)) + '\n'
            log += 'Number of pairs in training set: ' + str(len(training_orders)) + '\n'
            log += 'Number of pairs in test set: ' + str(len(test_orders)) + '\n\n\n'
            log_file.write(log)

        log = str(i + 1) + '. iteration\n'
        log_file.write(log)


        # determine regularization parameter (cross-validation on training data)
        # ----------------------------------------------------------------------
        if regularization == 'yes':
            print("2. DETERMING REGULARIZATION PARAMETER ...")
            reg_param, mean_auc = determine_alpha(training_instances, replace = replace, n_folds=5, reg_params=[0.0001, 0.001, 0.01, 0.1, 1, 10], method=method)
            log = 'Reg. param: ' + str(reg_param) + '; mean(AUC): ' + str(round(mean_auc, 4)) + '\n'
            log_file.write(log)
            print("aplpha = " + str(reg_param))
        else:
            print("2. REGULARIZATION SKIPPED: alpha = " + str(reg_param))

        # optimization
        # ------------
        print("3. OPTIMIZATION ...")
        res = custom_minimize(method, training_instances, training_orders, reg_param)


        # evaluation
        # ----------
        print("4. EVALUATION ...")
        auc = predict(res.x, test_instances, test_orders, method)
        print(auc)
        plis_auc_values.append(auc)
        log = 'AUC (PLIS): ' + str(round(auc, 4)) + '\n'
        log_file.write(log)


        # logistic regression
        print("5. LOGISTIC REGRESSION ...")
        auc = logreg_auc(training_instances, test_instances)
        logreg_auc_values.append(auc)
        log = 'AUC (LogReg): ' + str(round(auc, 4)) + '\n\n\n'
        log_file.write(log)


    log = "PLIS mean AUC: " + str(np.mean(plis_auc_values)) + '\n'
    log += "PLIS std AUC: " + str(np.std(plis_auc_values)) + '\n'
    log += "Logistic regression mean AUC: " + str(np.mean(logreg_auc_values)) + '\n'
    log += "Logistic regression std AUC: " + str(np.std(logreg_auc_values))
    log_file.write(log)


log_file.close()
