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


def run_first_test():
    print("RUNNING TEST 1")
    print("Preparing data")
    data_dir = arrange_dataset()
    set_SP, labels_SP = dataGenerator(data_dir, mode="train", nb_classes=4)
    new_set, new_labels = [], []
    for p,l in zip(set_SP, labels_SP):
        if l == 1:
            new_set.append(p)
            new_labels.append(0)
        elif l == 2:
            new_set.append(p)
            new_labels.append(1)

    batch_size = 16
    sp_sequence = NiiSequence2(new_set, new_labels, batch_size, nb_classes=2, mode="HC", shuffle=False)

    len_set = len(new_set)
    print("Length of the dataset:",len(new_set))


    print("Loading model")
    #model = tf.keras.models.load_model("experiences_1/classifier3D_bi-exp1-04-0.81")
    model = tf.keras.models.load_model("experiences_2/classifier3D_bi-exp1-08-0.81")
    print("Model loaded")


    y = model.predict(sp_sequence)
    y_pred = y.argmax(axis=1)
    y_test = np.array(new_labels)
    #accuracy
    accuracy = np.sum(y_pred == y_test)/len(y_pred)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy)
    #precision
    precision = np.sum(y_pred[y_test==0]==0)/np.sum(y_pred==0)
    print("Precision for recognizing SMCI from PMCI is:", precision)
    #recall
    recall = np.sum(y_pred[y_test==0]==0)/np.sum(y_test==0)
    print("Recall for recognizing SMCI from PMCI is:", recall)
    #f1
    f1 = 2*(precision*recall)/(precision+recall)
    print("F1 for recognizing SMCI from PMCI is:", f1)

    cm = confusion_matrix(y_test, y_pred)
    cm_ = cm / cm.astype(float).sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(10,10))
    sn.heatmap(cm, annot=True, ax=ax, fmt='g', square=True, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['SMCI', 'PMCI'])
    ax.yaxis.set_ticklabels(['SMCI', 'PMCI'])
    plt.savefig("experiences_2/confusion_matrix_exp1.png")
    plt.show()

    return len_set, accuracy, precision, recall, f1


def run_second_test():
    print("RUNNING TEST 2")
    print("Preparing data")
    data_dir = arrange_dataset()
    set_SP, labels_SP = dataGenerator(data_dir, mode="val", nb_classes=4)
    batch_size = 16
    test_sequence = NiiSequence(set_SP, batch_size, nb_classes=4, mode="HC", shuffle=False)

    print("Length of the dataset:",len(set_SP))

    print("Loading model")
    #model = tf.keras.models.load_model("experiences_1/classifier3D_bi-exp2-02-0.54")
    model = tf.keras.models.load_model("experiences_2/classifier3D_bi-exp1-08-0.81")
    print("Model loaded")

    y = model.predict(test_sequence)
    y_pred = y.argmax(axis=1)
    y_test = np.array(labels_SP)
    accuracy = np.sum(y_pred == y_test)/len(y_pred)
    print("Accuracy for 4 classes classifier is:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    cm_ = cm / cm.astype(float).sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(10,10))
    sn.heatmap(cm, annot=True, ax=ax, fmt='g', square=True, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['CN', 'SMCI', 'PMCI', 'AD'])
    ax.yaxis.set_ticklabels(['CN', 'SMCI', 'PMCI', 'AD'])
    plt.savefig("experiences_2/confusion_matrix_exp2_1.png")
    plt.show()


    set_SP, labels_SP = dataGenerator(data_dir, mode="val", nb_classes=4)
    new_set, new_labels = [], []
    for p,l in zip(set_SP, labels_SP):
        if l == 1:
            new_set.append(p)
            new_labels.append(0)
        elif l == 2:
            new_set.append(p)
            new_labels.append(1)

    len_test = len(new_set)
    batch_size = 16
    sp_sequence = NiiSequence2(new_set, new_labels, batch_size, nb_classes=2, mode="HC", shuffle=False)
    print("Length of the dataset:",len(new_set))

    y = model.predict(sp_sequence)
    y_pred = y.argmax(axis=1)
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            y_pred[i] = 0
        elif y_pred[i] == 2 or y_pred[i] == 3:
            y_pred[i] = 1
    y_test = np.array(new_labels)
    #accuracy
    accuracy1 = np.sum(y_pred == y_test)/len(y_pred)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy1)
    #precision
    precision = np.sum(y_pred[y_test==0]==0)/np.sum(y_pred==0)
    print("Precision for recognizing SMCI from PMCI is:", precision)
    #recall
    recall = np.sum(y_pred[y_test==0]==0)/np.sum(y_test==0)
    print("Recall for recognizing SMCI from PMCI is:", recall)
    #f1
    f1 = 2*(precision*recall)/(precision+recall)
    print("F1 for recognizing SMCI from PMCI is:", f1)

    cm = confusion_matrix(y_test, y_pred)
    cm_ = cm / cm.astype(float).sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(10,10))
    sn.heatmap(cm, annot=True, ax=ax, fmt='g', square=True, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['SMCI', 'PMCI'])
    ax.yaxis.set_ticklabels(['SMCI', 'PMCI'])
    plt.savefig("experiences_2/confusion_matrix_exp2_2.png")
    plt.show()

    return len_test, accuracy1, precision, recall, f1


def run_third_test():
    print("RUNNING TEST 3")
    print("Preparing data")
    data_dir = arrange_dataset()

    set_SP, labels_SP = dataGenerator(data_dir, mode="val", nb_classes=4)
    new_set, new_labels = [], []
    for p,l in zip(set_SP, labels_SP):
        if l == 1:
            new_set.append(p)
            new_labels.append(0)
        elif l == 2:
            new_set.append(p)
            new_labels.append(1)

    len_test = len(new_set)
    batch_size = 16
    sp_sequence = NiiSequence2(new_set, new_labels, batch_size, nb_classes=2, mode="HC", shuffle=False)
    print("Length of the dataset:",len(new_set))

    print("Loading model")
    #model = tf.keras.models.load_model("experiences_1/classifier3D_bi-exp3-03-0.83")
    model = tf.keras.models.load_model("experiences_2/classifier3D_bi-exp1-08-0.81")
    print("Model loaded")


    y = model.predict(sp_sequence)
    y_pred = y.argmax(axis=1)
    y_test = np.array(new_labels)
    #accuracy
    accuracy = np.sum(y_pred == y_test)/len(y_pred)
    print("Accuracy for recognizing SMCI from PMCI is:", accuracy)
    #precision
    precision = np.sum(y_pred[y_test==0]==0)/np.sum(y_pred==0)
    print("Precision for recognizing SMCI from PMCI is:", precision)
    #recall
    recall = np.sum(y_pred[y_test==0]==0)/np.sum(y_test==0)
    print("Recall for recognizing SMCI from PMCI is:", recall)
    #f1
    f1 = 2*(precision*recall)/(precision+recall)
    print("F1 for recognizing SMCI from PMCI is:", f1)

    cm = confusion_matrix(y_test, y_pred)
    cm_ = cm / cm.astype(float).sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(10,10))
    sn.heatmap(cm, annot=True, ax=ax, fmt='g', square=True, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['SMCI', 'PMCI'])
    ax.yaxis.set_ticklabels(['SMCI', 'PMCI'])
    plt.savefig("experiences_2/confusion_matrix_exp3.png")
    plt.show()

    return len_test, accuracy, precision, recall, f1

if __name__ == "__main__":
    methods = ["CN + AD", "4 classes", "CN - SMCI + AD - PMCI"]
    len_sets = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    lt, acc, pre, rec, f1 = run_first_test()
    len_sets.append(lt)
    accuracies.append(acc)
    precisions.append(pre)
    recalls.append(rec)
    f1s.append(f1)
    lt, acc, pre, rec, f1 = run_second_test()
    len_sets.append(lt)
    accuracies.append(acc)
    precisions.append(pre)
    recalls.append(rec)
    f1s.append(f1)
    lt, acc, pre, rec, f1 = run_third_test()
    len_sets.append(lt)
    accuracies.append(acc)
    precisions.append(pre)
    recalls.append(rec)
    f1s.append(f1)
    df = pd.DataFrame(list(zip(methods, len_sets, accuracies, precisions, recalls, f1s)), columns =['Method', 'Test set length', 'Accuracy', 'Precision', 'Recall', 'F1'])
    df.to_csv("experiences_2/results_test.csv", index=False)
    print(df)
