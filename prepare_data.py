import pandas as pd
import numpy as np
import random
import pickle
import json
import itertools


# aux function to create orders (pairs) from labeled instances
def create_orders(df, replace, size = 0.2):
    c1_index = np.array(df['output'][df['output'] == 1].index)
    c2_index = np.array(df['output'][df['output'] == 2].index)
    abs_size = int(np.floor(len(c1_index) * len(c2_index) * size))

    if replace == 'yes':
        c1_instances = np.random.choice(c1_index, abs_size, replace = True)
        c2_instances = np.random.choice(c2_index, abs_size, replace = True)
        orders = np.array([c1_instances, c2_instances]).transpose()
        return orders.tolist()
    elif replace == 'no':
        orders = set()
        while len(orders) < abs_size:
            c1_instance = np.random.choice(c1_index, 1).tolist()[0]
            c2_instance = np.random.choice(c2_index, 1).tolist()[0]
            orders.add((c1_instance, c2_instance))
        return list(orders)
    else:
        return []


# n_orders - percent of all pairs (comparisons) to be used in the training and test sets
# class_imbalance - ratio between smaller and larger class, should be the real number between 0 (only one class present) and 1 (both classes are equaly present)
def prepare_training_test_datasets(parsed_file_name, tt_imbalance = 0.7, replace = 'no', n_orders = 0.2, class_imbalance = 1, random_seed = 111):

    np.random.seed(random_seed)

    dataset_name = parsed_file_name.split("/")[-1].split(".")[0].replace("_parsed", "")
    parsed_data = pd.read_csv(parsed_file_name, header = 0)

    c1 = np.array(parsed_data['output'][parsed_data['output'] == 1].index)
    c2 = np.array(parsed_data['output'][parsed_data['output'] == 2].index)

    # setting class imbalance to a given value
    # if class_imbalance < 0 use all instances from the raw dataset regardless of initial class ratio in the dataset
    (larger_class, smaller_class) = (c1, c2) if len(c1) > len(c2) else (c2, c1)
    class_ratio = len(smaller_class) / len(larger_class)
    if class_imbalance > 0:
        if class_imbalance < class_ratio:
            # reducing smaller class to get the given ratio
            smaller_class = np.random.choice(smaller_class, int(np.floor(len(larger_class) * class_imbalance)), replace = False)
        else:
            # reducing larger class to get the given ratio
            larger_class = np.random.choice(larger_class, int(np.floor(len(smaller_class) / class_imbalance)), replace = False)

        (c1, c2) = (larger_class, smaller_class) if len(c1) > len(c2) else (smaller_class, larger_class)

    print('Dataset: ' + dataset_name + '\nInitial class imbalance: ' + str(np.round(class_ratio, 2)) + '\n')
    print("Size of c1 class: " + str(len(c1)))
    print("Size of c2 class: " + str(len(c2)))

    np.random.shuffle(c1)
    np.random.shuffle(c2)

    # training and test will be the same size
    cutoff_c1 = int(np.floor(len(c1) * tt_imbalance))
    cutoff_c2 = int(np.floor(len(c2) * tt_imbalance))

    training_c1 = c1[:cutoff_c1]
    test_c1 = c1[cutoff_c1:]
    training_c2 = c2[:cutoff_c2]
    test_c2 = c2[cutoff_c2:]

    # split raw dataset into training and test instances
    training_instances = parsed_data.loc[list(training_c1) + list(training_c2)]
    training_instances.reset_index(drop=True, inplace=True)
    test_instances = parsed_data.loc[list(test_c1) + list(test_c2)]
    test_instances.reset_index(drop=True, inplace=True)

    # create orders
    training_orders = create_orders(training_instances, replace, size = n_orders)
    test_orders = create_orders(test_instances, replace, size = n_orders)

    training_size = len(training_orders)
    test_size = len(test_orders)
    print('Number of pairs in traing set: ' + str(training_size))
    print('Number of pairs in test set: ' + str(test_size) + '\n')

    return training_instances, test_instances, training_orders, test_orders






