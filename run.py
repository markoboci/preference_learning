import json
import pandas as pd
import numpy as np
import os
import time

from prepare_data import prepare_training_test_datasets
from model import determine_alpha, custom_minimize, predict
from logreg_model import logreg_auc


# load params
#------------

with open('config.json', 'r') as infile:
    params = json.load(infile)

method = params['METHOD']
class_imbalance_list = params['CLASS_IMBALANCE']
n_orders_list = params['N_ORDERS']
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

    for class_imbalance in class_imbalance_list:

        for n_orders in n_orders_list:

            plis_auc_training = []
            plis_auc_test = []
            logreg_auc_training = []
            logreg_auc_test = []
            plis_exec_times = []
            logreg_exec_times = []

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
                random_seed = int(i + class_imbalance * 10 ** 7 + n_orders * 10 ** 4)
                training_instances, test_instances, training_orders, test_orders = prepare_training_test_datasets(dataset_path, tt_imbalance=training_test_imbalance, replace = replace, n_orders=n_orders, class_imbalance=class_imbalance, random_seed=random_seed)
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
                    reg_param, mean_auc = determine_alpha(training_instances, replace = replace, n_folds=5, reg_params=[0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], method=method)
                    log = 'Reg. param: ' + str(reg_param) + '; mean(AUC): ' + str(round(mean_auc, 4)) + '\n'
                    log_file.write(log)
                    print("aplpha = " + str(reg_param))
                else:
                    print("2. REGULARIZATION SKIPPED: alpha = " + str(reg_param))

                # optimization
                # ------------
                print("3. OPTIMIZATION ...")
                start = time.time()
                res = custom_minimize(method, training_instances, training_orders, reg_param)
                end = time.time()
                plis_exec = round(end - start, 3)
                plis_exec_times.append(plis_exec)
                log = "PLIS exec time: " + str(plis_exec) + "\n"
                log_file.write(log)


                # evaluation
                # ----------
                print("4. EVALUATION ...")
                auc_training = predict(res.x, training_instances, training_orders, method)
                auc_test = predict(res.x, test_instances, test_orders, method)
                plis_auc_training.append(auc_training)
                plis_auc_test.append(auc_test)
                log = 'AUC (PLIS): ' + str(round(auc_test, 4)) + '\n'
                log_file.write(log)


                # logistic regression
                print("5. LOGISTIC REGRESSION ...")
                auc_test, auc_training, logreg_exec = logreg_auc(training_instances, test_instances)
                logreg_auc_test.append(auc_test)
                logreg_auc_training.append(auc_training)
                logreg_exec_times.append(logreg_exec)
                log = 'LogReg exec time: ' + str(logreg_exec) + '\n'
                log += 'AUC (LogReg): ' + str(round(auc_test, 4)) + '\n\n\n'
                log_file.write(log)


            log = "PLIS mean AUC test: " + str(round(np.mean(plis_auc_test), 4)) + '\n'
            log += "PLIS std AUC test: " + str(round(np.std(plis_auc_test), 4)) + '\n'
            log += "PLIS mean AUC training: " + str(round(np.mean(plis_auc_training), 4)) + '\n'
            log += "PLIS std AUC training: " + str(round(np.std(plis_auc_training), 4)) + '\n'
            log += "PLIS mean exec time: " + str(round(np.mean(plis_exec_times), 4)) + '\n'
            log += "PLIS std exec time: " + str(round(np.std(plis_exec_times), 4)) + '\n'

            log += "LogReg mean AUC test: " + str(round(np.mean(logreg_auc_test), 4)) + '\n'
            log += "LogReg std AUC test: " + str(round(np.std(logreg_auc_test), 4)) + '\n'
            log += "LogReg mean AUC training: " + str(round(np.mean(logreg_auc_training), 4)) + '\n'
            log += "LogReg std AUC training: " + str(round(np.std(logreg_auc_training), 4)) + '\n'
            log += "LogReg mean exec time: " + str(round(np.mean(logreg_exec_times), 4)) + '\n'
            log += "LogReg std exec time: " + str(round(np.std(logreg_exec_times), 4))
            log_file.write(log)


            log_file.close()
