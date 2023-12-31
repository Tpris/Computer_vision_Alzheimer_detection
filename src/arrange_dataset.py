import sys
import os
import csv
import numpy as np
import splitfolders


def rename_files(data_dir, data_info, dict):
    for i in range(1, len(data_info)):
        ID = data_info[i][dict["id"]]
        path, path_mask = get_path_from_ID(ID, data_dir)
        if not os.path.exists(path):
            print(f"Error : {path} does not exist.")
        else:
            # new path is label + ID + nii.gz
            new_path = data_dir + data_info[i][dict["label"]] + "-" + ID + ".nii.gz"
            new_path_mask = data_dir + data_info[i][dict["label"]] + "-" + ID + "-mask.nii.gz"
            os.rename(path, new_path)
            os.rename(path_mask, new_path_mask)
            print(f"Renamed {path} to {new_path}")


def convert_labels(data_info, dict):
    for i in range(1, len(data_info)):
        if data_info[i][dict["label"]] == "MCI":
            if data_info[i][dict["conversion"]] in ["1", "2", "3"]:
                data_info[i][dict["label"]] = "PMCI"
            elif data_info[i][dict["conversion"]] in ["4"]:
                data_info[i][dict["label"]] = "SMCI"
            else:
                data_info[i][dict["label"]] = "UNKNOWN"
    return data_info

def get_path_from_ID(ID, data_dir):
    path = data_dir + "n_mmni_fADNI_" + ID + "_1.5T_t1w.nii.gz"
    mask_path = data_dir + "mask_n_mmni_fADNI_" + ID + "_1.5T_t1w.nii.gz"
    return path, mask_path


def arrange_dataset():
    if not os.path.exists("./lib/AD"):
        os.mkdir("./lib/AD")
        print("Created directory ./lib/AD")
    if not os.path.exists("./lib/CN"):
        os.mkdir("./lib/CN")
        print("Created directory ./lib/CN")
    if not os.path.exists("./lib/PMCI"):
        os.mkdir("./lib/PMCI")
        print("Created directory ./lib/PMCI")
    if not os.path.exists("./lib/SMCI"):
        os.mkdir("./lib/SMCI")
        print("Created directory ./lib/SMCI")
    for file in os.listdir("./lib"):
        if file.endswith(".nii.gz") and not file.endswith("-mask.nii.gz"):
            if file.startswith("AD"):
                os.rename("./lib/"+file, "./lib/AD/"+file)
            elif file.startswith("CN"):
                os.rename("./lib/"+file, "./lib/CN/"+file)
            elif file.startswith("PMCI"):
                os.rename("./lib/"+file, "./lib/PMCI/"+file)
            elif file.startswith("SMCI"):
                os.rename("./lib/"+file, "./lib/SMCI/"+file)

    if not os.path.exists("./data"):
        splitfolders.ratio('./lib', output="./data", seed=1337, ratio=(.8, .2), group_prefix=None)
        print("Split dataset into train and val sets")
    
    print("Dataset arranged in ./data")
    return "./data"


if __name__ == '__main__':
    
    data_dir = "./Dataset/lib/"
    csv_path = './list_standardized_tongtong_2017.csv'

    if len(sys.argv) == 2:
        data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print("Usage : python arrange_dataset.py <data_folder>. Default is './Dataset/lib/'")
        print(f"Error : {data_dir} does not exist.")
        sys.exit(1)

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.array(data)
    c = {"id":0, "rooster_id":1, "age":2, "sex":3, "label":4, "conversion":5, "MMSE":6, "RAVLT":7, "FAQ":8, "CDR-SB": 9}
    
    data = convert_labels(data, c)
    rename_files(data_dir, data, c)