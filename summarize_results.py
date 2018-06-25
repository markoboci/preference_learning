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

# PLIS
plis_auc_mean_test = []
plis_auc_sd_test = []
plis_auc_mean_training = []
plis_auc_sd_training = []
plis_exec_time_mean = []
plis_exec_time_sd = []

# LogReg
logreg_auc_mean_test = []
logreg_auc_sd_test = []
logreg_auc_mean_training = []
logreg_auc_sd_training = []
logreg_exec_time_mean = []
logreg_exec_time_sd = []


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

            # PLIS
            if line.startswith("PLIS mean AUC test"):
                plis_auc_mean_test.append(line.replace("PLIS mean AUC test: ", "")[0:6])
                continue
            if line.startswith("PLIS std AUC test"):
                plis_auc_sd_test.append(line.replace("PLIS std AUC test: ", "")[0:6])
                continue
            if line.startswith("PLIS mean AUC training"):
                plis_auc_mean_training.append(line.replace("PLIS mean AUC training: ", "")[0:6])
                continue
            if line.startswith("PLIS std AUC training"):
                plis_auc_sd_training.append(line.replace("PLIS std AUC training: ", "")[0:6])
                continue
            if line.startswith("PLIS mean exec time"):
                plis_exec_time_mean.append(line.replace("PLIS mean exec time: ", "")[0:6])
                continue
            if line.startswith("PLIS std exec time"):
                plis_exec_time_sd.append(line.replace("PLIS std exec time: ", "")[0:6])
                continue

            # LogReg
            if line.startswith("LogReg mean AUC test"):
                logreg_auc_mean_test.append(line.replace("LogReg mean AUC test: ", "")[0:6])
                continue
            if line.startswith("LogReg std AUC test"):
                logreg_auc_sd_test.append(line.replace("LogReg std AUC test: ", "")[0:6])
                continue
            if line.startswith("LogReg mean AUC training"):
                logreg_auc_mean_training.append(line.replace("LogReg mean AUC training: ", "")[0:6])
                continue
            if line.startswith("LogReg std AUC training"):
                logreg_auc_sd_training.append(line.replace("LogReg std AUC training: ", "")[0:6])
                continue
            if line.startswith("LogReg mean exec time"):
                logreg_exec_time_mean.append(line.replace("LogReg mean exec time: ", "")[0:6])
                continue
            if line.startswith("LogReg std exec time"):
                logreg_exec_time_sd.append(line.replace("LogReg std exec time: ", "")[0:6])
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
                   'plis_auc_mean_test' : plis_auc_mean_test,
                   'plis_auc_sd_test' : plis_auc_sd_test,
                   'plis_auc_mean_training' : plis_auc_mean_training,
                   'plis_auc_sd_training' : plis_auc_sd_training,
                   'plis_exec_time_mean' : plis_exec_time_mean,
                   'plis_exec_time_sd' : plis_exec_time_sd,
                   'logreg_auc_mean_test' : logreg_auc_mean_test,
                   'logreg_auc_sd_test' : logreg_auc_sd_test,
                   'logreg_auc_mean_training' : logreg_auc_mean_training,
                   'logreg_auc_sd_training' : logreg_auc_sd_training,
                   'logreg_exec_time_mean' : logreg_exec_time_mean,
                   'logreg_exec_time_sd' : logreg_exec_time_sd})


# rearrange columns in dataframe
cols = ['method', 'dataset', 'class_imbalance', 'number_of_pairs', 'plis_auc_mean_test', 'plis_auc_sd_test', 'logreg_auc_mean_test',
        'logreg_auc_sd_test', 'plis_auc_mean_training', 'plis_auc_sd_training', 'logreg_auc_mean_training', 'logreg_auc_sd_training',
        'repetitions', 'with_replacement', 'regularization', 'instances_training', 'instances_test', 'pairs_training', 'pairs_test',
        'plis_exec_time_mean', 'plis_exec_time_sd', 'logreg_exec_time_mean', 'logreg_exec_time_sd']

df = df[cols]

# sort dataframe
df = df.sort_values(by = ['method', 'class_imbalance', 'number_of_pairs', 'dataset'])

# write dataframe to a file
df.to_csv('output/summarized_results.csv', header = True, index = False)