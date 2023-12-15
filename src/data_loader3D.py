import os
from keras.utils import Sequence
import numpy as np
import nibabel as nib
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import random
from monai.transforms import AdjustContrast, RandBiasField, ShiftIntensity, Rand3DElastic

def dataGenerator(data_dir, mode="train", nb_classes=4):
    set = []
    labels = []
    path = os.path.join(data_dir, mode)
    AD = os.path.join(path, "AD")
    CN = os.path.join(path, "CN")
    if nb_classes == 4:
        PMCI = os.path.join(path, "PMCI")
        SMCI = os.path.join(path, "SMCI")
        for file in os.listdir(CN):
            if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
                set.append(os.path.join(CN, file))
                labels.append(0)
        for file in os.listdir(SMCI):
            if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
                set.append(os.path.join(SMCI, file))
                labels.append(1)
        for file in os.listdir(PMCI):
            if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
                set.append(os.path.join(PMCI, file))
                labels.append(2)
        for file in os.listdir(AD):
            if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
                set.append(os.path.join(AD, file))
                labels.append(3)
    else:
        for file in os.listdir(CN):
            if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
                set.append(os.path.join(CN, file))
                labels.append(0)
        for file in os.listdir(AD):
            if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
                set.append(os.path.join(AD, file))
                labels.append(1)
    return set, labels


def get_HC(irm):
    hc1 = irm[40:80, 90:130, 40:80]
    hc2 = irm[100:140, 90:130, 40:80]
    return hc1, hc2

def dim_augmentation(data):
    data = np.expand_dims(data, axis=3)
    return data

class NiiSequence(Sequence):
    def __init__(self, file_paths, batch_size, nb_classes=4, mode="full", shuffle=True, data_aug=[]):
        assert mode in ["full", "HC", "reduced"], "mode must be either 'full', 'HC' or 'reduced'"
        assert nb_classes in [2, 4, 2.2], "nb_classes must be either 2, 4 or 2.2"
        self.file_paths = file_paths
        if shuffle:
            np.random.shuffle(self.file_paths)
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.use_SP = False
        if self.nb_classes == 2.2:
            self.nb_classes = 2
            self.use_SP = True
        self.mode = mode
        self.data_aug = data_aug
        self.nb_aug = len(data_aug)

    def __len__(self):
        if not self.data_aug:
            return int(np.ceil(len(self.file_paths) / self.batch_size))
        return int(np.ceil(len(self.file_paths)*(self.nb_aug+1) / self.batch_size))

    def __getitem__(self, idx):
        batch_data, batch_labels = self.simplegetitem(idx)

        if not self.data_aug:
            return batch_data, batch_labels
        
        batch_labels_base = batch_labels.copy()
        concat_batch = (batch_data,)
        for name in self.data_aug:
            match(name):
                case 'gaussian_offset':
                    noise = random.randint(5,15)
                    batch_data1 = add_gaussian_offset(batch_data.copy(), sigma=noise)
                    batch_labels = np.concatenate((batch_labels,batch_labels_base), axis=0)
                    concat_batch += (batch_data1,)
                case 'gaussian_noise':
                    noise = random.randint(7,20)
                    batch_data1 = add_gaussian_noise(batch_data.copy(), sigma=noise)
                    batch_labels = np.concatenate((batch_labels,batch_labels_base), axis=0)
                    concat_batch += (batch_data1,)
                case 'adjustContrast':
                    noise = random.random()
                    adj = AdjustContrast(noise)
                    im_b = adj(batch_data.copy())
                    concat_batch += (im_b,)
                    batch_labels = np.concatenate((batch_labels,batch_labels_base), axis=0)
                case 'randBiasField':
                    bias = RandBiasField()
                    concat_one = []
                    for data in batch_data.copy():
                        im_b = bias(data)
                        concat_one += [im_b]
                    concat_batch += (np.array(concat_one),)
                    batch_labels = np.concatenate((batch_labels,batch_labels_base), axis=0)
                case 'shiftIntensity': 
                    noise = random.randint(5,30)
                    intens = ShiftIntensity(noise)
                    im_b = intens(batch_data.copy())
                    concat_batch += (im_b,)
                    batch_labels = np.concatenate((batch_labels,batch_labels_base), axis=0)
                case 'rand3DElastic':
                    el = Rand3DElastic((50,100),(50,1000), prob=1.0)
                    concat_one = []
                    for data in batch_data.copy():
                        im_b = el(data)
                        concat_one += [im_b]
                    concat_batch += (np.array(concat_one),)
                    batch_labels = np.concatenate((batch_labels,batch_labels_base), axis=0)

        batch_data = np.concatenate(concat_batch,axis=0)

        return batch_data, batch_labels

    def simplegetitem(self, idx):
        if self.data_aug:
            batch_paths = self.file_paths[int(idx * self.batch_size/(self.nb_aug+1)):int((idx + 1) * self.batch_size/(self.nb_aug+1))]
        else :
            batch_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = [self.load_and_preprocess(path) for path in batch_paths]
        batch_data = np.array(batch_data)

        batch_labels = [self.extract_label(path) for path in batch_paths]
        batch_labels = np.eye(self.nb_classes)[batch_labels]

        return batch_data, batch_labels
        
    def extract_label(self, file_path):
        class_name = os.path.basename(os.path.dirname(file_path))
        if self.nb_classes == 4:
            label = {"CN": 0, "SMCI": 1, "PMCI": 2, "AD": 3}[class_name]
        elif self.nb_classes == 2 and self.use_SP:
            label = {"CN": 0, "SMCI": 0, "PMCI": 1, "AD": 1}[class_name]
        elif self.nb_classes == 2:
            label = {"CN": 0, "AD": 1}[class_name]
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


class NiiSequence2(Sequence):
    def __init__(self, file_paths, labels, batch_size, nb_classes=4, mode="full", shuffle=True, data_aug=False):
        assert mode in ["full", "HC", "reduced"], "mode must be either 'full', 'HC' or 'reduced'"
        assert nb_classes in [2, 4], "nb_classes must be either 2 or 4"
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        if shuffle:
            assert len(file_paths) == len(labels)
            p = np.random.permutation(len(file_paths))
            self.file_paths = self.file_paths[p]
            self.labels = self.labels[p]
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.mode = mode
        self.data_aug = data_aug

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_data, batch_labels = self.simplegetitem(idx)

        if not self.data_aug:
            return batch_data, batch_labels
        
        num_aug = random.randint(0,2)
        match num_aug:
            case 1:
                batch_data = add_gaussian_offset(batch_data.copy(), sigma=25)
            case 2:
                batch_data = add_gaussian_noise(batch_data.copy(), sigma=25)


        return batch_data, batch_labels

    def simplegetitem(self, idx):
        batch_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = [self.load_and_preprocess(path) for path in batch_paths]
        batch_data = np.array(batch_data)

        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = np.eye(self.nb_classes)[batch_labels]

        return batch_data, batch_labels

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