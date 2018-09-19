import numpy as np
import os
import datetime
import argparse
import sys
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# read the argv
argv = sys.argv

#x_file = "mini_matrix.txt"
x_file = argv[1]

#y_file = "mini_label.txt"
y_file = argv[2]

# fold 
fold = int(argv[3])

# result path
results_path = './Results/Logistic_Regression_Classifier'
folder_name = "/{0}_kfold_{1}_Logistic_Regression_Classifier". \
        format(x_file, fold)
log_path = results_path + folder_name + '/log/'
if not os.path.exists(log_path):
    os.mkdir(log_path)

# Get data
x_data = np.loadtxt(x_file, delimiter = "\t")
x_data = x_data[:, 6:]

# sample
n_labeled = x_data.shape[0]

y_data = np.loadtxt(y_file, delimiter = "\t")
#y_data = y_data.reshape(n_labeld, 1)
#y_data = np.column_stack((y_data, 1 - y_data))

# spliting
kf = KFold(n_splits = 5, random_state = 1)
kf.get_n_splits(x_data)
k_list = []
for train_index, test_index in kf.split(x_data):
    k_list.append([train_index, test_index])    
cur_index = k_list[fold - 1]

# training data
x_train = x_data[cur_index[0], :]
y_train = y_data[cur_index[0]]

# testing data
x_test = x_data[cur_index[1], :]
y_test = y_data[cur_index[1]]



# training
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

score = logisticRegr.score(x_test, y_test)
print(score)

with open(log_path + "log.txt", "w") as fo:
    fo.write("Classification Accuracy: " + str(score) + "\n")

