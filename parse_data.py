import pandas as pd
import numpy as np

# Car evaluation
#---------------

# # file_name = 'data/car.data.txt'
# # df = pd.read_csv(file_name, sep = ",", header = None)
# #
# # # assigning column names
# # df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'output']
# #
# # # assigning numerical values to descrete attributes
# # df['buying'] = df['buying'].map({'vhigh': 1, 'high': 2, 'med' : 3, 'low' : 4})
# # df['maint'] = df['maint'].map({'vhigh': 1, 'high': 2, 'med' : 3, 'low' : 4})
# # df['doors'] = df['doors'].map({'2': 2, '3': 3, '4' : 4, '5more' : 5})
# # df['persons'] = df['persons'].map({'2': 1, '4' : 2, 'more' : 3})
# # df['lug_boot'] = df['lug_boot'].map({'small': 1, 'med' : 2, 'big' : 3})
# # df['safety'] = df['safety'].map({'low': 1, 'med' : 2, 'high' : 3})
# # df['output'] = df['output'].map({'unacc' : 1, 'acc' : 1, 'good' : 2, 'vgood' : 2})
# #
# # # remove rows with missing values
# # df.dropna(inplace = True)
# #
# # df.to_csv('data/car_evaluation_parsed.csv', header = True, index = False)



# Car MPG
#---------------

# # #1. mpg: continuous
# # #2. cylinders: multi-valued discrete
# # #3. displacement: continuous
# # #4. horsepower: continuous
# # #5. weight: continuous
# # #6. acceleration: continuous
# # #7. model year: multi-valued discrete
# # #8. origin: multi-valued discrete
# # #9. car name: string (unique for each instance)
# #
# # file_name = 'data/auto-mpg.data-original.txt'
# # car_mpg_file = open(file_name, 'r').readlines()
# #
# # mpg = []
# # cylinders = []
# # displacement = []
# # horsepower = []
# # weight = []
# # acceleration = []
# # #model_year = []
# #
# # for line in car_mpg_file:
# #     line = line.split()
# #     line = line[0:7]
# #     if 'NA' in line:
# #         continue
# #     mpg.append(float(line[0]))
# #     cylinders.append(int(line[1].replace(".", "")))
# #     displacement.append(float(line[2]))
# #     horsepower.append(float(line[3]))
# #     weight.append(float(line[4]))
# #     acceleration.append(float(line[5]))
# #     #model_year.append(int(line[6].replace(".", "")))
# #
# # #df = pd.DataFrame({'mpg' : mpg, 'cylinders' : cylinders, 'displacement' : displacement, 'hp' : horsepower, 'weight' : weight, 'acc' : acceleration, 'year' : model_year})
# # df = pd.DataFrame({'mpg' : mpg, 'cylinders' : cylinders, 'displacement' : displacement, 'hp' : horsepower, 'weight' : weight, 'acc' : acceleration})
# #
# # #print(df[['mpg']].describe(percentiles = [0.1, .25, .5, .75]))
# # #print(df[['cylinders']].describe())
# # #print(df[['displacement']].describe())
# # #print(df[['hp']].describe())
# # #print(df[['weight']].describe())
# # #print(df[['acc']].describe())
# #
# # df['output'] = pd.Series(pd.cut(np.array(df['mpg']), bins = [8, 15, 47], labels = [1, 2]))
# # df.drop(columns = ['mpg'], inplace = True)
# # #print(df[['output']].describe())
# # print(df.head())
# # df.dropna(inplace = True)
# # df.to_csv('data/car_mpg_parsed.csv', header = True, index = False)



# Concrete Strength
#------------------

# # df = pd.read_excel('data/Concrete_Data.xls', sheet_name='Sheet1')
# # df.columns = ['cement', 'slag', 'ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age', 'strength']
# # df.dropna(inplace = True)
# #
# # # negative effects
# # df['ash'] = - df['ash']
# # df['water'] = - df['water']
# # df['coarse_agg'] = - df['coarse_agg']
# #
# # # output variable
# # df['output'] = pd.Series(pd.cut(np.array(df['strength']), bins = [2, 25, 90], labels = [1, 2]))
# # df.drop(columns = ['strength'], inplace = True)
# #
# # print(df.head(20))
# # print(df.describe())
# # df.to_csv('data/concrete_data_parsed.csv', header = True, index = False)



# Breast cancer
#--------------

df = pd.read_csv('data/breast-cancer.data.txt', header = None)
df.columns = ['class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']
df.drop(columns = ['breast', 'breast_quad', 'irradiat'])


df['class'] = df['class'].map({'no-recurrence-events': 0, 'recurrence-events': 1})
df['age'] = df['age'].map({'10-19' : 1, '20-29' : 2, '30-39' :3, '40-49' : 4, '50-59' : 5, '60-69' : 6, '70-79' : 7, '80-89' : 8, '90-99' : 9})
df['menopause'] = df['menopause'].map({'lt40' : 1, 'ge40' : 2, 'premeno' : 0})
df['tumor_size'] = df['tumor_size'].map({'0-4' : 1, '5-9' : 2, '10-14' : 3, '15-19' : 4, '20-24' : 5, '25-29' : 6, '30-34' : 7, '35-39' : 8, '40-44' : 9, '45-49' : 10, '50-54' : 11, '55-59' : 12})
df['inv_nodes'] = df['inv_nodes'].map({'0-2' : 1, '3-5' : 2, '6-8' : 3, '9-11' : 4, '12-14' : 5, '15-17' : 6, '18-20' : 7, '21-23' : 8, '24-26' : 9, '27-29' : 10, '30-32' : 11, '33-35' : 12, '36-39' : 13})
df['node_caps'] = df['node_caps'].map({'yes' : 0, 'no' : 1})
df['output'] = df['deg_malig'].map({3 : 2, 2 : 1, 1 : 1})

df.drop(columns = ['deg_malig', 'breast', 'breast_quad', 'irradiat'], inplace = True)
df.dropna(inplace = True)
print(df.head())
df.to_csv('data/breast_cancer_parsed.csv', header = True, index = False)

