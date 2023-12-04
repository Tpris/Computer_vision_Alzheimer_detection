import os
from keras.utils import Sequence
import numpy as np
import nibabel as nib

def dataGenerator(data_dir, mode="train"):
    set = []
    labels = []
    path = os.path.join(data_dir, mode)
    AD = os.path.join(path, "AD")
    CN = os.path.join(path, "CN")
    PMCI = os.path.join(path, "PMCI")
    SMCI = os.path.join(path, "SMCI")
    for file in os.listdir(AD):
        if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
            set.append(os.path.join(AD, file))
            labels.append(0)
    for file in os.listdir(CN):
        if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
            set.append(os.path.join(CN, file))
            labels.append(1)
    for file in os.listdir(PMCI):
        if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
            set.append(os.path.join(PMCI, file))
            labels.append(2)
    for file in os.listdir(SMCI):
        if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
            set.append(os.path.join(SMCI, file))
            labels.append(3)

    return set, labels


def get_HC(irm):
    hc1 = irm[40:80, 90:130, 40:80]
    hc2 = irm[100:140, 90:130, 40:80]
    return hc1, hc2

def dim_augmentation(data):
    data = np.expand_dims(data, axis=3)
    return data

class NiiSequence(Sequence):
    def __init__(self, file_paths, batch_size, nb_classes=4, mode="full", shuffle=True):
        assert mode in ["full", "HC", "reduced"], "mode must be either 'full', 'HC' or 'reduced'"
        assert nb_classes in [2, 4], "nb_classes must be either 2 or 4"
        self.file_paths = file_paths
        if shuffle:
            np.random.shuffle(self.file_paths)
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.mode = mode

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = [self.load_and_preprocess(path) for path in batch_paths]
        batch_data = np.array(batch_data)

        batch_labels = [self.extract_label(path) for path in batch_paths]
        batch_labels = np.eye(self.nb_classes)[batch_labels]

        return batch_data, batch_labels
        
    def extract_label(self, file_path):
        class_name = os.path.basename(os.path.dirname(file_path))
        if self.nb_classes == 4:
            label = {"AD": 0, "CN": 1, "PMCI": 2, "SMCI": 3}[class_name]
        elif self.nb_classes == 2: #we consider AD and pMCI as AD and CN and sMCI as CN
            label = {"AD": 0, "CN": 1, "PMCI": 0, "SMCI": 1}[class_name]
        return label

    def load_and_preprocess(self, file_path):
        data = nib.load(file_path).get_fdata()
        
        if self.mode == "HC":
            hc1, hc2 = get_HC(data)
            data = np.concatenate((hc1, hc2), axis=2)
        elif self.mode == "reduced":
            mri = np.zeros((125, 150, 32))
            for k in range(32):
                mri[:,:,k] = data[30:155,40+k*2,10:160]
            data = mri
    
        data = dim_augmentation(data)
        return data


if __name__ == '__main__':
    dir_path = "./data"
    train_set, train_labels = dataGenerator(dir_path, mode="train")
    test_set, test_labels = dataGenerator(dir_path, mode="val")
    print("Len of train set:", len(train_set))
    print("Len of val set:", len(test_set))
    print("Train path example:", train_set[0])
    print("Train label example:", train_labels[0])

    train_gen = NiiSequence(train_set, batch_size=1, nb_classes=4, mode="full")
    test_gen = NiiSequence(test_set, batch_size=1, nb_classes=2, mode="HC")
    print("Shape of one MRI element:", train_gen[0][0].shape)
    print("Train label example after loading:", train_gen[0][1])

    print("Shape of one batch:", test_gen[0][0].shape)
    print("Test label example after loading:", test_gen[0][1])