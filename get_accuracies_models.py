import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sn
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from matplotlib.path import Path
from sklearn.metrics import classification_report, confusion_matrix

from keras.optimizers import Adam
from src.bi_classifier3D import Biclassifier3D
from src.data_loader3D import dataGenerator, NiiSequence, NiiSequence2

from sklearn.model_selection import train_test_split

from src.arrange_dataset import arrange_dataset

import pandas as pd

#import accuracy, recall, precision, f1_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def run_first_test():
    print("RUNNING TEST 1")
    print("Preparing data")
    data_dir = arrange_dataset()
    train_data, train_labels = dataGenerator(data_dir, mode="train", nb_classes=2)
    test_data, test_labels = dataGenerator(data_dir, mode="val", nb_classes=2)
    batch_size = 16
    train_sequence = NiiSequence(train_data, batch_size, nb_classes=2, mode="HC", shuffle=False)
    test_sequence = NiiSequence(test_data, batch_size, nb_classes=2, mode="HC", shuffle=False)

    len_train_data = len(train_data)
    len_test_data = len(test_data)
    print("Length of the training set:",len(train_data))
    print("Length of the test set:",len(test_data))


    print("Loading model")
    ##model = tf.keras.models.load_model("experiences_2/classifier3D_bi-exp1-08-0.81")
    model = tf.keras.models.load_model("experiences_1/classifier3D_bi-exp1-04-0.81")

    print("Model loaded")

    y = model.predict(train_sequence)
    y_pred_train = y.argmax(axis=1)
    y_train = np.array(train_labels)
    #accuracy
    accuracy_train = np.sum(y_pred_train == y_train)/len(y_pred_train)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy_train)
    #precision
    precision_train = np.sum(y_pred_train[y_train==0]==0)/np.sum(y_pred_train==0)
    print("Precision for recognizing SMCI from PMCI is:", precision_train)
    #recall
    recall_train = np.sum(y_pred_train[y_train==0]==0)/np.sum(y_train==0)
    print("Recall for recognizing SMCI from PMCI is:", recall_train)
    #f1
    f1_train = 2*(precision_train*recall_train)/(precision_train+recall_train)
    print("F1 for recognizing SMCI from PMCI is:", f1_train)

    y = model.predict(test_sequence)
    y_pred_test = y.argmax(axis=1)
    y_test = np.array(test_labels)
    #accuracy
    accuracy_test = np.sum(y_pred_test == y_test)/len(y_pred_test)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy_test)
    #precision
    precision_test = np.sum(y_pred_test[y_test==0]==0)/np.sum(y_pred_test==0)
    print("Precision for recognizing SMCI from PMCI is:", precision_test)
    #recall
    recall_test = np.sum(y_pred_test[y_test==0]==0)/np.sum(y_test==0)
    print("Recall for recognizing SMCI from PMCI is:", recall_test)
    #f1
    f1_test = 2*(precision_train*recall_train)/(precision_train+recall_train)
    print("F1 for recognizing SMCI from PMCI is:", f1_test)

    return len_train_data, len_test_data, accuracy_train, precision_train, recall_train, f1_train, accuracy_test, precision_test, recall_test, f1_test


def run_second_test():
    print("RUNNING TEST 2")
    print("Preparing data")
    data_dir = arrange_dataset()
    train_data, train_labels = dataGenerator(data_dir, mode="train", nb_classes=4)
    test_data, test_labels = dataGenerator(data_dir, mode="val", nb_classes=4)
    batch_size = 16
    train_sequence = NiiSequence(train_data, batch_size, nb_classes=4, mode="HC", shuffle=False)
    test_sequence = NiiSequence(test_data, batch_size, nb_classes=4, mode="HC", shuffle=False)

    len_train_data = len(train_data)
    len_test_data = len(test_data)
    print("Length of the training set:",len(train_data))
    print("Length of the test set:",len(test_data))

    print("Loading model")
    # model = tf.keras.models.load_model("experiences_2/classifier3D_bi-exp2-07-0.46")
    model = tf.keras.models.load_model("experiences_1/classifier3D_bi-exp2-02-0.54")
    print("Model loaded")

    y = model.predict(train_sequence)
    y_pred_train = y.argmax(axis=1)
    y_train = np.array(train_labels)
    #accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Accuracy for 4 classes :", accuracy_train)
    #precision
    precision_train = precision_score(y_train, y_pred_train, average='macro')
    print("Precision for 4 classes :", precision_train)
    #recall
    recall_train = recall_score(y_train, y_pred_train, average='macro')
    print("Recall for 4 classes :", recall_train)
    #f1
    f1_train = f1_score(y_train, y_pred_train, average='macro')
    print("F1 for 4 classes :", f1_train)

    y = model.predict(test_sequence)
    y_pred_test = y.argmax(axis=1)
    y_test = np.array(test_labels)
    #accuracy
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Accuracy for 4 classes :", accuracy_test)
    #precision
    precision_test = precision_score(y_test, y_pred_test, average='macro')
    print("Precision for 4 classes :", precision_test)
    #recall
    recall_test = recall_score(y_test, y_pred_test, average='macro')
    print("Recall for 4 classes :", recall_test)
    #f1
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    print("F1 for 4 classes :", f1_test)

    return len_train_data, len_test_data, accuracy_train, precision_train, recall_train, f1_train, accuracy_test, precision_test, recall_test, f1_test


def run_third_test():
    print("RUNNING TEST 3")
    print("Preparing data")
    data_dir = arrange_dataset()

    train_set, train_labels = dataGenerator(data_dir, mode="train", nb_classes=4)
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            train_labels[i] = 0
        elif train_labels[i] == 2 or train_labels[i] == 3:
            train_labels[i] = 1
    test_set, test_labels = dataGenerator(data_dir, mode="val", nb_classes=4)
    for i in range(len(test_labels)):
        if test_labels[i] == 1:
            test_labels[i] = 0
        elif test_labels[i] == 2 or test_labels[i] == 3:
            test_labels[i] = 1

    print("Train set size for experience 3:", len(train_set))
    print("Test set size for experience 3:", len(test_set))

    batch_size = 16

    train_sequence = NiiSequence(train_set, batch_size, nb_classes=2.2, mode="HC", shuffle=False)
    test_sequence = NiiSequence(test_set, batch_size, nb_classes=2.2, mode="HC",shuffle=False)

    len_train_data = len(train_set)
    len_test_data = len(test_set)
    print("Length of the training set:",len(train_set))
    print("Length of the test set:",len(test_set))

    print("Loading model")
    # model = tf.keras.models.load_model("experiences_2/classifier3D_bi-exp3-10-0.77")
    model = tf.keras.models.load_model("experiences_1/classifier3D_bi-exp3-03-0.83")
    print("Model loaded")


    y = model.predict(train_sequence)
    y_pred_train = y.argmax(axis=1)
    y_train = np.array(train_labels)
    #accuracy
    accuracy_train = np.sum(y_pred_train == y_train)/len(y_pred_train)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy_train)
    #precision
    precision_train = np.sum(y_pred_train[y_train==0]==0)/np.sum(y_pred_train==0)
    print("Precision for recognizing SMCI from PMCI is:", precision_train)
    #recall
    recall_train = np.sum(y_pred_train[y_train==0]==0)/np.sum(y_train==0)
    print("Recall for recognizing SMCI from PMCI is:", recall_train)
    #f1
    f1_train = 2*(precision_train*recall_train)/(precision_train+recall_train)
    print("F1 for recognizing SMCI from PMCI is:", f1_train)

    y = model.predict(test_sequence)
    y_pred_test = y.argmax(axis=1)
    y_test = np.array(test_labels)
    #accuracy
    accuracy_test = np.sum(y_pred_test == y_test)/len(y_pred_test)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy_test)
    #precision
    precision_test = np.sum(y_pred_test[y_test==0]==0)/np.sum(y_pred_test==0)
    print("Precision for recognizing SMCI from PMCI is:", precision_test)
    #recall
    recall_test = np.sum(y_pred_test[y_test==0]==0)/np.sum(y_test==0)
    print("Recall for recognizing SMCI from PMCI is:", recall_test)
    #f1
    f1_test = 2*(precision_train*recall_train)/(precision_train+recall_train)
    print("F1 for recognizing SMCI from PMCI is:", f1_test)

    return len_train_data, len_test_data, accuracy_train, precision_train, recall_train, f1_train, accuracy_test, precision_test, recall_test, f1_test
    


if __name__ == "__main__":
    methods = ["CN + AD", "4 classes", "CN - SMCI + AD - PMCI"]
    len_trains = []
    len_tests = []
    accuracies_train = []
    precisions_train = []
    recalls_train = []
    f1s_train = []
    accuracies_test = []
    precisions_test = []
    recalls_test = []
    f1s_test = []
    l_train, l_test, acc_train, pre_train, rec_train, f1_train, acc_test, pre_test, rec_test, f1_test = run_first_test()
    len_trains.append(l_train)
    len_tests.append(l_test)
    accuracies_train.append(acc_train)
    precisions_train.append(pre_train)
    recalls_train.append(rec_train)
    f1s_train.append(f1_train)
    accuracies_test.append(acc_test)
    precisions_test.append(pre_test)
    recalls_test.append(rec_test)
    f1s_test.append(f1_test)
    l_train, l_test, acc_train, pre_train, rec_train, f1_train, acc_test, pre_test, rec_test, f1_test = run_second_test()
    len_trains.append(l_train)
    len_tests.append(l_test)
    accuracies_train.append(acc_train)
    precisions_train.append(pre_train)
    recalls_train.append(rec_train)
    f1s_train.append(f1_train)
    accuracies_test.append(acc_test)
    precisions_test.append(pre_test)
    recalls_test.append(rec_test)
    f1s_test.append(f1_test)
    l_train, l_test, acc_train, pre_train, rec_train, f1_train, acc_test, pre_test, rec_test, f1_test = run_third_test()
    len_trains.append(l_train)
    len_tests.append(l_test)
    accuracies_train.append(acc_train)
    precisions_train.append(pre_train)
    recalls_train.append(rec_train)
    f1s_train.append(f1_train)
    accuracies_test.append(acc_test)
    precisions_test.append(pre_test)
    recalls_test.append(rec_test)
    f1s_test.append(f1_test)
    df = pd.DataFrame(list(zip(methods, len_trains, len_tests, accuracies_train, precisions_train, recalls_train, f1s_train, accuracies_test, precisions_test, recalls_test, f1s_test)), columns =['Method', 'Length train', 'Length test', 'Accuracy train', 'Precision train', 'Recall train', 'F1 train', 'Accuracy test', 'Precision test', 'Recall test', 'F1 test'])
    print(df)
    df.to_csv("experiences_1/results_train.csv", index=False)
