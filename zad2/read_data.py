import os
import pandas as pd
import scipy as sp
import numpy as np
import cv2
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(data_folder_path: str, start_timestamp: str, end_timestamp: str):
    ''' Read data from the folder and return it as a list of numpy arrays
    Args:
    data_folder_path: str - path to the folder with data
    start_timestamp: str - start timestamp in format rrrr-mm-dd-hh-mm-ss
    end_timestamp: str - end timestamp in format rrrr-mm-dd-hh-mm-ss
    Returns:
    list of numpy arrays
    '''
    # list of filenames to be found in the folder
    filenames = [
        "resp_defense_affine.npy",
        "resp_defense_binary.npy",
        "resp_home_affine.npy",
        "resp_home_binary.npy",
        "used_indexes.npy"
    ]
    all_data = [None]*5
    for root, dirs, files in os.walk(data_folder_path):
        for dir_name in dirs:
            # dir_name is a timestamp in format rrrr-mm-dd-hh-mm-ss
            # if timestamp in given range and folder has 5 .npy files
            if start_timestamp <= dir_name <= end_timestamp:
                # check if all files are in the folder
                if all([os.path.isfile(os.path.join(root, dir_name, filename)) for filename in filenames]):
                    # read data
                    resp_defense_affine = np.load(os.path.join(root, dir_name, filenames[0]))
                    resp_defense_binary = np.load(os.path.join(root, dir_name, filenames[1]))
                    resp_home_affine = np.load(os.path.join(root, dir_name, filenames[2]))
                    resp_home_binary = np.load(os.path.join(root, dir_name, filenames[3]))
                    used_indexes = np.load(os.path.join(root, dir_name, filenames[4]))
                    # do something with the data

                    if all_data[0] is None:
                        all_data[0] = resp_defense_affine
                        all_data[1] = resp_defense_binary
                        all_data[2] = resp_home_affine
                        all_data[3] = resp_home_binary
                        all_data[4] = used_indexes
                    else:
                        # append data to the list
                        all_data[0] = np.concatenate([all_data[0], resp_defense_affine], axis=0)
                        all_data[1] = np.concatenate([all_data[1], resp_defense_binary], axis=0)
                        all_data[2] = np.concatenate([all_data[2], resp_home_affine], axis=0)
                        all_data[3] = np.concatenate([all_data[3], resp_home_binary], axis=0)
                        all_data[4] = np.concatenate([all_data[4], used_indexes], axis=0)

                    # print("Data from", dir_name, "has been read")
                    # print("resp_defense_affine.shape=", resp_defense_affine.shape)
                    # print("resp_defense_binary.shape=", resp_defense_binary.shape)
                    # print("resp_home_affine.shape=", resp_home_affine.shape)
                    # print("resp_home_binary.shape=", resp_home_binary.shape)
                    # print("used_indexes.shape=", used_indexes.shape)
                    #
                    # print("all_data[0].shape=", all_data[0].shape)
                    # print("all_data[1].shape=", all_data[1].shape)
                    # print("all_data[2].shape=", all_data[2].shape)
                    # print("all_data[3].shape=", all_data[3].shape)
                    # print("all_data[4].shape=", all_data[4].shape)

    # remove duplicates if indexes are the same
    used_ids = []
    cleaned_data = [[], [], [], [], []]

    for i in range(all_data[4].shape[0]):
        if all_data[4][i] not in used_ids:
            used_ids.append(i)
    cleaned_data[0] = all_data[0][used_ids]
    cleaned_data[1] = all_data[1][used_ids]
    cleaned_data[2] = all_data[2][used_ids]
    cleaned_data[3] = all_data[3][used_ids]
    cleaned_data[4] = all_data[4][used_ids]
    resp_defense_affine, resp_defense_binary, resp_home_affine, resp_home_binary, used_indexes = cleaned_data
    return resp_defense_affine, resp_defense_binary, resp_home_affine, resp_home_binary, used_indexes


if __name__ == "__main__":
    data_folder_path = "./data/data"
    timestamp_start = "2024-03-16"
    timestamp_end = "2024-03-17"
    resp_defense_affine, resp_defense_binary, resp_home_affine, resp_home_binary, used_indexes = read_data(data_folder_path, timestamp_start, timestamp_end)
    M = resp_defense_affine.shape[1]
    N = 385

    # get only the first N rows
    A = resp_defense_affine[:N]
    B = resp_home_affine[:N]

    print(A.shape)
    A = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    print(A.shape)
    # calculate X from AX = B
    X = np.linalg.lstsq(A, B, rcond=None)[0]

    A_test = resp_defense_affine[N+1:]
    B_test = resp_home_affine[N+1:]

    print(A_test.shape)
    print(np.ones((A_test.shape[0], 1)).shape)
    A_test = np.concatenate([A_test, np.ones((A_test.shape[0], 1))], axis=1)
    print(A_test.shape)

    print("error=", np.linalg.norm(A_test @ X - B_test))