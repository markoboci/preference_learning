import os
import pandas as pd

# all log files in output folder
all_files = os.listdir('output/')

# columns in summary file
dataset = []
method = []
number_of_pairs = []
class_imbalance = []
repetitions = []
with_replacement = []
regularization = []
instances_training = []
instances_test = []
pairs_training = []
pairs_test = []
plis_auc_mean = []
plis_auc_sd = []
logreg_auc_mean = []
logreg_auc_sd = []


for _file in all_files:
    if _file[-8:] == ".log.txt":
        log_file = open('output/' + _file, 'r').read().splitlines()
        for line in log_file:
            if line.startswith("DATASET"):
                dataset.append(line.replace("DATASET: ", ""))
                continue
            if line.startswith("METHOD"):
                method.append(line.replace("METHOD: ", ""))
                continue
            if line.startswith("NUMBER OF PAIRS"):
                number_of_pairs.append(line.replace("NUMBER OF PAIRS (N_ORDERS): ", ""))
                continue
            if line.startswith("CLASS IMBALANCE"):
                class_imbalance.append(line.replace("CLASS IMBALANCE: ", ""))
                continue
            if line.startswith("REPETITIONS"):
                repetitions.append(line.replace("REPETITIONS: ", ""))
                continue
            if line.startswith("WITH REPLACEMENT"):
                with_replacement.append(line.replace("WITH REPLACEMENT: ", ""))
                continue
            if line.startswith("REGULARIZATION:"):
                regularization.append(line.replace("REGULARIZATION: ", ""))
                continue
            if line.startswith("Number of instances in training"):
                instances_training.append(line.replace("Number of instances in training set: ", ""))
                continue
            if line.startswith("Number of instances in test"):
                instances_test.append(line.replace("Number of instances in test set: ", ""))
                continue
            if line.startswith("Number of pairs in training"):
                pairs_training.append(line.replace("Number of pairs in training set: ", ""))
                continue
            if line.startswith("Number of pairs in test"):
                pairs_test.append(line.replace("Number of pairs in test set: ", ""))
                continue
            if line.startswith("PLIS mean"):
                plis_auc_mean.append(line.replace("PLIS mean AUC: ", "")[0:6])
                continue
            if line.startswith("PLIS std"):
                plis_auc_sd.append(line.replace("PLIS std AUC: ", "")[0:6])
                continue
            if line.startswith("Logistic regression mean"):
                logreg_auc_mean.append(line.replace("Logistic regression mean AUC: ", "")[0:6])
                continue
            if line.startswith("Logistic regression std"):
                logreg_auc_sd.append(line.replace("Logistic regression std AUC: ", "")[0:6])
                continue

# create dataframe
df = pd.DataFrame({'dataset' : dataset,
                   'method' : method,
                   'number_of_pairs' : number_of_pairs,
                   'class_imbalance' : class_imbalance,
                   'repetitions' : repetitions,
                   'with_replacement' : with_replacement,
                   'regularization' : regularization,
                   'instances_training' : instances_training,
                   'instances_test' : instances_test,
                   'pairs_training' : pairs_training,
                   'pairs_test' : pairs_test,
                   'plis_auc_mean' : plis_auc_mean,
                   'plis_auc_sd' : plis_auc_sd,
                   'logreg_auc_mean' : logreg_auc_mean,
                   'logreg_auc_sd' : logreg_auc_sd})

# rearrange columns in dataframe
cols = ['method', 'dataset', 'class_imbalance', 'plis_auc_mean', 'plis_auc_sd', 'logreg_auc_mean', 'logreg_auc_sd',
        'number_of_pairs', 'repetitions', 'with_replacement', 'regularization', 'instances_training', 'instances_test', 'pairs_training', 'pairs_test']
df = df[cols]

# sort dataframe
df = df.sort_values(by = ['method', 'class_imbalance', 'dataset'])

# write dataframe to a file
df.to_csv('output/summarized_results.csv', header = True, index = False)