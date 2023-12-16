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

data_dir = arrange_dataset()    

def run_first_exp():
    print("Preparing data")

    nb_classes = 2

    train_set, train_labels = dataGenerator(data_dir, mode="train", nb_classes=nb_classes)
    test_set, test_labels = dataGenerator(data_dir, mode="val", nb_classes=nb_classes)
    train_set, val_set, train_labels, val_labels = train_test_split(train_set, train_labels, test_size=0.2, random_state=42)

    print("Train set size for experience 1:", len(train_set))
    print("Validation set size for experience 1: ", len(val_set))
    print("Test set size for experience 1:", len(test_set))

    batch_size = 16

    train_sequence = NiiSequence(train_set, batch_size, nb_classes=nb_classes, mode="HC")
    val_sequence = NiiSequence(val_set, batch_size, nb_classes=nb_classes, mode="HC")
    test_sequence = NiiSequence(test_set, batch_size, nb_classes=nb_classes, mode="HC",shuffle=False)

    input_shape = (train_sequence[0][0].shape[1], train_sequence[0][0].shape[2], train_sequence[0][0].shape[3], train_sequence[0][0].shape[4])
    print("Input shape:", input_shape)
    bicl = Biclassifier3D(input_shape, n_classes=nb_classes, n_filters=8, kernel_size=3, activation='relu', dropout=0.3)
    model = bicl.build_model()
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=2e-3),
        metrics=['accuracy'],
    )

    filepath="experiences_2/classifier3D_bi-exp1-{epoch:02d}-{val_accuracy:.2f}"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1)

    callbacks_list = [checkpoint, early_stop, reduce_lr]

    model.fit(
        train_sequence,
        validation_data=val_sequence,
        validation_steps=len(val_set) // batch_size,
        callbacks=callbacks_list,
        epochs=10,
    )

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    y = model.predict(test_sequence)
    y_pred = y.argmax(axis=1)
    y_test = np.array(test_labels)
    print("Accuracy of the testing set for experience 1:", np.sum(y_pred == y_test)/len(y_pred))
    print("Preparing PMCI and SMCI data")

    set_SP, labels_SP = dataGenerator(data_dir, mode="train", nb_classes=4)
    print(len(set_SP), len(labels_SP))
    new_set, new_labels = [], []
    for p,l in zip(set_SP, labels_SP):
        if l == 1:
            new_set.append(p)
            new_labels.append(0)
        elif l == 2:
            new_set.append(p)
            new_labels.append(1)

    print("Len new set:", len(new_set))
    sp_sequence = NiiSequence2(new_set, new_labels, batch_size, nb_classes=2, mode="HC", shuffle=False)

    y_ = model.predict(sp_sequence)
    y__pred = y_.argmax(axis=1)
    y__test = np.array(new_labels)
    print("Accuracy of the M/S set for experience 1:", np.sum(y__pred == y__test)/len(y__pred))
    

def run_second_exp():
    print("Preparing data")

    nb_classes = 4

    train_set, train_labels = dataGenerator(data_dir, mode="train", nb_classes=nb_classes)
    test_set, test_labels = dataGenerator(data_dir, mode="val", nb_classes=nb_classes)
    train_set, val_set, train_labels, val_labels = train_test_split(train_set, train_labels, test_size=0.2, random_state=42)

    print("Train set size for experience 2:", len(train_set))
    print("Validation set size for experience 2: ", len(val_set))
    print("Test set size for experience 2:", len(test_set))

    batch_size = 16

    train_sequence = NiiSequence(train_set, batch_size, nb_classes=nb_classes, mode="HC")
    val_sequence = NiiSequence(val_set, batch_size, nb_classes=nb_classes, mode="HC")
    test_sequence = NiiSequence(test_set, batch_size, nb_classes=nb_classes, mode="HC",shuffle=False)

    input_shape = (train_sequence[0][0].shape[1], train_sequence[0][0].shape[2], train_sequence[0][0].shape[3], train_sequence[0][0].shape[4])
    print("Input shape:", input_shape)
    bicl = Biclassifier3D(input_shape, n_classes=nb_classes, n_filters=8, kernel_size=3, activation='relu', dropout=0.3)
    model = bicl.build_model()
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=2e-3),
        metrics=['accuracy'],
    )

    filepath="experiences_2/classifier3D_bi-exp2-{epoch:02d}-{val_accuracy:.2f}"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1)

    callbacks_list = [checkpoint, early_stop, reduce_lr]

    model.fit(
        train_sequence,
        validation_data=val_sequence,
        validation_steps=len(val_set) // batch_size,
        callbacks=callbacks_list,
        epochs=10,
    )

    y = model.predict(test_sequence)
    y_pred = y.argmax(axis=1)
    y_test = np.array(test_labels)
    print("Accuracy of the testing set for experience 2:", np.sum(y_pred == y_test)/len(y_pred))
    

def run_third_exp():
    print("Preparing data")

    nb_classes = 2

    train_set, train_labels = dataGenerator(data_dir, mode="train", nb_classes=4)
    test_set, test_labels = dataGenerator(data_dir, mode="val", nb_classes=4)
    train_set, val_set, train_labels, val_labels = train_test_split(train_set, train_labels, test_size=0.2, random_state=42)

    print("Train set size for experience 3:", len(train_set))
    print("Validation set size for experience 3: ", len(val_set))
    print("Test set size for experience 3:", len(test_set))

    batch_size = 16

    train_sequence = NiiSequence(train_set, batch_size, nb_classes=2.2, mode="HC")
    val_sequence = NiiSequence(val_set, batch_size, nb_classes=2.2, mode="HC")
    test_sequence = NiiSequence(test_set, batch_size, nb_classes=2.2, mode="HC",shuffle=False)

    input_shape = (train_sequence[0][0].shape[1], train_sequence[0][0].shape[2], train_sequence[0][0].shape[3], train_sequence[0][0].shape[4])
    print("Input shape:", input_shape)
    bicl = Biclassifier3D(input_shape, n_classes=nb_classes, n_filters=8, kernel_size=3, activation='relu', dropout=0.3)
    model = bicl.build_model()
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=2e-3),
        metrics=['accuracy'],
    )

    filepath="experiences_2/classifier3D_bi-exp3-{epoch:02d}-{val_accuracy:.2f}"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1)

    callbacks_list = [checkpoint, early_stop, reduce_lr]

    model.fit(
        train_sequence,
        validation_data=val_sequence,
        validation_steps=len(val_set) // batch_size,
        callbacks=callbacks_list,
        epochs=10,
    )

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    y = model.predict(test_sequence)
    y_pred = y.argmax(axis=1)
    y_test = np.array(test_labels)
    print("Accuracy of the testing set for experience 1:", np.sum(y_pred == y_test)/len(y_pred))
    print("Preparing PMCI and SMCI data")

    set_SP, labels_SP = dataGenerator(data_dir, mode="val", nb_classes=4)
    print(len(set_SP), len(labels_SP))
    new_set, new_labels = [], []
    for p,l in zip(set_SP, labels_SP):
        if l == 1:
            new_set.append(p)
            new_labels.append(0)
        elif l == 2:
            new_set.append(p)
            new_labels.append(1)

    print("Len new set:", len(new_set))
    sp_sequence = NiiSequence2(new_set, new_labels, batch_size, nb_classes=2, mode="HC", shuffle=False)

    y_ = model.predict(sp_sequence)
    y__pred = y_.argmax(axis=1)
    y__test = np.array(new_labels)
    print("Accuracy of the M/S set for experience 3:", np.sum(y__pred == y__test)/len(y__pred))
    

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    run_first_exp()
    run_second_exp()
    run_third_exp()