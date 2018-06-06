import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.optimize import minimize
import json
from sklearn.metrics.pairwise import euclidean_distances
from math import log, exp
import time
from scipy.optimize import LinearConstraint
import itertools
import operator

from prepare_data import create_orders



# NORMALIZATION
#--------------

# standardization
def standard_sc(df):
    # last column of all datasets will always be output variable, thus there is no need for normalization
    n_col = df.shape[1]
    df.iloc[:,range(n_col - 1)] = preprocessing.scale(df.iloc[:,range(n_col - 1)])
    return df

# min-max normalization
def min_max_sc(df):
    n_col = df.shape[1]
    min_max_scaler = preprocessing.MinMaxScaler()
    df.iloc[:, range(n_col - 1)] = min_max_scaler.fit_transform(df.iloc[:, range(n_col - 1)])
    return df



# OPTIMIZATION
#-------------

def topsis_pis(df):
    pis = df.drop(columns='output').max(axis=0)
    pis += np.random.normal(0,0.1,1)
    return pis


# learn only PIS (vector z) - unconstrained optimization
#-------------------------------------------------------
def negLL_PIS(z, a, b, reg_param):
    az_dist = euclidean_distances(a, [z])[:,0]
    bz_dist = euclidean_distances(b, [z])[:,0]
    return np.sum(az_dist) + np.sum(np.log(np.exp(-az_dist) + np.exp(-bz_dist))) + reg_param * np.sum(z * z)

# gradient
def negLL_PIS_jac(z, a, b, reg_param):
    grad = np.zeros_like(z)
    az_dist = euclidean_distances(a, [z])[:, 0]
    bz_dist = euclidean_distances(b, [z])[:, 0]
    for i in range(len(z)):
        dist_parc_a = - 2 * (a[:,i] - z[i]) / az_dist
        dist_parc_b = - 2 * (b[:,i] - z[i]) / bz_dist
        grad[i] = np.sum(dist_parc_a) + np.sum((- np.exp(-az_dist) * dist_parc_a - np.exp(-bz_dist) * dist_parc_b) / (np.exp(-az_dist) + np.exp(-bz_dist))) + reg_param * 2 * z[i]
    return grad



# learn only weights (vector w) - constrained optimization
#---------------------------------------------------------
def negLL_weight(w, a, b):
    az_dist = np.sqrt(np.sum(a ** 2 * w ** 2, axis = 1))
    bz_dist = np.sqrt(np.sum(b ** 2 * w ** 2, axis = 1))
    return np.sum(az_dist) + np.sum(np.log(np.exp(-az_dist) + np.exp(-bz_dist)))

# gradient
def negLL_weight_jac(w, a, b):
    grad = np.zeros_like(w)
    az_dist = np.sqrt(np.sum(a ** 2 * w ** 2, axis=1))
    bz_dist = np.sqrt(np.sum(b ** 2 * w ** 2, axis=1))
    for i in range(len(w)):
        dist_parc_a = 2 * w[i] * a[:, i] ** 2 / az_dist
        dist_parc_b = 2 * w[i] * b[:, i] ** 2 / bz_dist
        grad[i] = np.sum(dist_parc_a) + np.sum((- np.exp(-az_dist) * dist_parc_a - np.exp(-bz_dist) * dist_parc_b) / (np.exp(-az_dist) + np.exp(-bz_dist)))
    return grad

# equality constraint: sum of all weights has to be equal to 1
eq_cons_w = {'type' : 'eq',
           'fun' : lambda x: np.array([sum(x) - 1]),
           'jac' : lambda x: np.ones(len(x))}

# inequality constraint: each individual weight has to be posititve
def ineq_constr_jac_weights(x):
    m = len(x)
    ineq_cons = []
    for i in range(m):
        vec = [0] * m
        vec[i] = 1
        ineq_cons.append(vec)
    return np.array(ineq_cons)

ineq_cons_w = {'type' : 'ineq',
             'fun' : lambda x: np.array(x),
             'jac' : ineq_constr_jac_weights}



# learn PIS and weights (vectors z and w) - constrained optimization
#-------------------------------------------------------------------
def negLL_PIS_weight(zw, a, b, reg_param):
    assert(len(zw) % 2 == 0)
    z = zw[0:len(zw)//2]
    w = zw[len(zw)//2:]
    az_dist = np.sqrt(np.sum((a - z) ** 2 * w ** 2, axis = 1))
    bz_dist = np.sqrt(np.sum((b - z) ** 2 * w ** 2, axis = 1))
    return np.sum(az_dist) + np.sum(np.log(np.exp(-az_dist) + np.exp(-bz_dist))) + reg_param * np.sum(z * z)

# gradient
def negLL_PIS_weight_jac(zw, a, b, reg_param):
    assert (len(zw) % 2 == 0)
    z = zw[0:len(zw) // 2]
    w = zw[len(zw) // 2:]
    m = len(z)
    grad = np.zeros_like(zw)
    az_dist = np.sqrt(np.sum((a - z) ** 2 * w ** 2, axis=1))
    bz_dist = np.sqrt(np.sum((b - z) ** 2 * w ** 2, axis=1))
    for i in range(len(z)):
        dist_parc_z_a = - 2 * w[i] ** 2 * (a[:,i] - z[i]) / az_dist
        dist_parc_z_b = - 2 * w[i] ** 2 * (b[:,i] - z[i]) / bz_dist
        dist_parc_w_a = 2 * w[i] * (a[:,i] - z[i]) ** 2 / az_dist
        dist_parc_w_b = 2 * w[i] * (b[:,i] - z[i]) ** 2 / bz_dist
        grad[i] = np.sum(dist_parc_z_a) + np.sum((- np.exp(-az_dist) * dist_parc_z_a - np.exp(-bz_dist) * dist_parc_z_b) / (np.exp(-az_dist) + np.exp(-bz_dist))) + reg_param * 2 * z[i]
        grad[i + m] = np.sum(dist_parc_w_a) + np.sum((- np.exp(-az_dist) * dist_parc_w_a - np.exp(-bz_dist) * dist_parc_w_b) / (np.exp(-az_dist) + np.exp(-bz_dist)))
    return grad

# equality constraint: sum of all weights has to be equal to 1, vector z has no constraints
eq_cons_pis_w = {'type' : 'eq',
           'fun' : lambda x: np.array([sum(x[len(x)//2:]) - 1]),
           'jac' : lambda x: [0] * (len(x)//2) + [1] * (len(x)//2)}

# inequality constraint: each individual weight has to be posititve, vecotr z has no constraints
def ineq_constr_jac_weights_z(x):
    m = len(x)//2
    ineq_cons = []
    for i in range(m):
        vec = [0] * 2 * m
        vec[m + i] = 1
        ineq_cons.append(vec)
    return np.array(ineq_cons)

ineq_cons_pis_w = {'type' : 'ineq',
             'fun' : lambda x: np.array(x[len(x)//2:]),
             'jac' : ineq_constr_jac_weights_z}



# invokes minimization method based on provided 'method' argument
#----------------------------------------------------------------
def custom_minimize(method, training_instances, training_orders, reg_param):

    training_instances = standard_sc(training_instances)
    training_orders = np.array(training_orders)
    a = np.array(training_instances.drop(columns='output').loc[training_orders[:, 1]])
    b = np.array(training_instances.drop(columns='output').loc[training_orders[:, 0]])
    m = len(a[0])

    if method == 'PIS':
        initial_vector = topsis_pis(training_instances)
        #res = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method = 'SLSQP', jac = negLL_PIS_jac, options = {'ftol' : 1e-6, 'disp' : True, 'maxiter' : 300})
        #res1 = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method='SLSQP', options={'ftol': 1e-6, 'disp': True, 'eps' : 2e-6})
        #res2 = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method='nelder-mead', options={'xtol': 1e-6, 'disp': True})
        #res3 = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method='BFGS', jac=negLL_PIS_jac, options={'gtol': 1e-3, 'disp': True, 'maxiter': 100})
        #res4 = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method='BFGS', options={'gtol': 1e-3, 'disp': True, 'eps' : 1e-5, 'maxiter' : 100})
        #res5 = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method='CG', jac=negLL_PIS_jac,options={'gtol': 1e-3, 'disp': True, 'maxiter': 100})
        res6 = minimize(negLL_PIS, initial_vector, args=(a, b, reg_param), method='CG', options={'gtol': 1e-2, 'disp': True, 'eps': 1e-5, 'maxiter': 200})

        #print("SLSQP With grad: ", str(res.x))
        #print("SLSQP Without grad: ", str(res1.x))
        #print("nelder-mead: " + str(res2.x))
        #print("BFGS With grad: ", str(res3.x))
        #print("BFGS Without grad: ", str(res4.x))
        #print("CG With grad: ", str(res5.x))
        #print("CG Without grad: ", str(res6.x))

        return res6

    if method == 'weight':
        initial_vector = [1/m] * m
        res = minimize(negLL_weight, initial_vector, args = (a, b), method='SLSQP', jac = negLL_weight_jac, constraints=[eq_cons_w, ineq_cons_w], options={'ftol': 1e-9, 'disp': True})
        #res1 = minimize(negLL_weight, initial_vector, args=(a, b), method='SLSQP', constraints=[eq_cons_w, ineq_cons_w], options={'ftol': 1e-9, 'disp': True})
        #print("With grad: ", str(res.x))
        #print("Without grad: ", str(res1.x))
        return res


    if method == 'PIS_weight':
        initial_vector = list(topsis_pis(training_instances)) + [1 / m] * m
        res = minimize(negLL_PIS_weight, initial_vector, args = (a, b, reg_param), method='SLSQP', jac = negLL_PIS_weight_jac, constraints=[eq_cons_pis_w, ineq_cons_pis_w], options={'ftol': 1e-6, 'disp': True})
        #res1 = minimize(negLL_PIS_weight, initial_vector, args=(a, b, reg_param), method='SLSQP',constraints=[eq_cons_pis_w, ineq_cons_pis_w], options={'ftol': 1e-6, 'disp': True, 'eps' : 2e-6})
        #print("With grad: ", str(res.x))
        #print("Without grad: ", str(res1.x))
        return res



# EVALUATION
#-----------

def custom_distance(x, zw, method):
    if method == 'PIS':
        assert(len(zw) == len(x))
        z = zw
        return np.sqrt(np.sum((x - z) ** 2))
    if method == 'weight':
        assert(len(zw) == len(x))
        w = zw
        return np.sqrt(np.sum(w ** 2 * x ** 2))
    if method == 'PIS_weight':
        assert(len(zw) // 2 == len(x))
        z = zw[0:len(zw)//2]
        w = zw[len(zw)//2:]
        return np.sqrt(np.sum(w ** 2 * (x - z) ** 2))


def predict(zw, test_instances, test_orders, method):
    predictions = []
    all_predictions = []
    c1_index_test = np.array(test_instances['output'][test_instances['output'] == 1].index)
    c2_index_test = np.array(test_instances['output'][test_instances['output'] == 2].index)
    test_instances = standard_sc(test_instances)
    test_instances = test_instances.drop(columns = 'output')

    for order in test_orders:
        a = np.array(test_instances.loc[order[1]])
        b = np.array(test_instances.loc[order[0]])
        if custom_distance(b, zw, method) > custom_distance(a, zw, method):
            predictions.append(True)
        else:
            predictions.append(False)

    # auc
    for c1_ind in c1_index_test:
        for c2_ind in c2_index_test:
            a = np.array(test_instances.loc[c2_ind])
            b = np.array(test_instances.loc[c1_ind])
            if custom_distance(b, zw, method) > custom_distance(a, zw, method):
                all_predictions.append(True)
            else:
                all_predictions.append(False)

    acc = sum(predictions) / len(predictions)
    auc = sum(all_predictions) / (len(c1_index_test) * len(c2_index_test))

    return auc



# DETERMINE REGULARIZATION PARAMETER
#-----------------------------------
def make_folds(list_, k):
    np.random.shuffle(list_)
    l = len(list_)
    return [list_[i * l // k : (i+1) * l // k] for i in range(k)]

def determine_alpha(training_instances, replace = 'no', n_orders = 0.2, n_folds = 5, reg_params = [0.0001, 0.001, 0.01, 0.1, 1, 10], method = 'PIS'):

    cv_auc = {key: 0 for key in reg_params}

    c1_index = list(training_instances['output'][training_instances['output'] == 1].index)
    c2_index = list(training_instances['output'][training_instances['output'] == 2].index)

    folds_c1_index = make_folds(c1_index, n_folds)
    folds_c2_index = make_folds(c2_index, n_folds)

    for k in range(n_folds):

        cv_test_index = folds_c1_index[k] + folds_c2_index[k]
        cv_training_index = []
        for i in range(n_folds):
            if i != k:
                cv_training_index += folds_c1_index[i] + folds_c2_index[i]

        cv_training_instances = training_instances.loc[cv_training_index]
        cv_test_instances = training_instances.loc[cv_test_index]

        cv_training_orders = create_orders(cv_training_instances, replace, n_orders)
        cv_test_orders = create_orders(cv_test_instances, replace, n_orders)


        for reg_param in reg_params:
            res = custom_minimize(method, cv_training_instances, cv_training_orders, reg_param)
            auc = predict(res.x, cv_test_instances, cv_test_orders, method)
            cv_auc[reg_param] += auc

    print(cv_auc)
    max_auc = max(cv_auc.items(), key=operator.itemgetter(1))
    return max_auc[0], max_auc[1] / n_folds

