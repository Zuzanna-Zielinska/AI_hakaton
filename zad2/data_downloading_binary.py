# Description: This file contains the code for downloading data from the server
from enum import Enum

import numpy as np
import os


class EndpointType(Enum):
    BASE = "home"
    DEFENDER = "defense"

class EndpointEncoding(Enum):
    BINARY = "binary"
    AFFINE = "affine"


from taskdataset import TaskDataset
from utils import get_all_indices, download_data, reset_endpoint, QUERY_MAX_ITEMS


REQ_INDEXES = 1000
INDEX_STEP  = QUERY_MAX_ITEMS - REQ_INDEXES
NUMBER_OF_TRIES = 5
THRESHOLD = 0.5


def main_binary():
    
    all_indices = get_all_indices()
    
    # get 2000 indexes for endpoint A
    idxs_A = all_indices[:QUERY_MAX_ITEMS]

    while True:

        try:
            # reset endpoint A - base endpoint
            reset_endpoint(EndpointType.BASE.value, EndpointEncoding.BINARY.value)

            # get 2000 values from endpoint A
            data_A = download_data(idxs_A, EndpointType.BASE.value, EndpointEncoding.BINARY.value)

        except Exception as err:
            print(f"[WARNING] ERROR occured, content: [{err}]")
            continue

        break

    # get REQ_INDEXES indexes from endpoint A
    # REQ_INDEXES because it is affinic transformation matrix, so we need to have 384 rows of data and +1 for the bias
    idxs_B = idxs_A[:REQ_INDEXES]
    train_data_A = data_A[:REQ_INDEXES]
    new_idx = QUERY_MAX_ITEMS

    # while we don't have all dataset - 20000 data:
    while new_idx < 20000:
        test_values = [[] for _ in range(NUMBER_OF_TRIES)]
        test_data_B = None
        
        # get 5 different results for binary representation and accept the most correct one
        for i in range(NUMBER_OF_TRIES):
            print(f"[DEBUG] Current index range: [{new_idx}, {new_idx+INDEX_STEP}]")
            while True:
                try:
                    # reset endpoint B - defender endpoint
                    reset_endpoint(EndpointType.DEFENDER.value, EndpointEncoding.BINARY.value)

                    # get REQ_INDEXES values from endpoint B
                    train_data_B = download_data(idxs_B, EndpointType.DEFENDER.value, EndpointEncoding.BINARY.value)

                    # get INDEX_STEP values from endpoint B
                    test_data_B = download_data(all_indices[new_idx:new_idx+INDEX_STEP], EndpointType.DEFENDER.value, EndpointEncoding.BINARY.value)
                    break

                except Exception as err:
                    print(f"[WARNING] ERROR occured, content: [{err}]")
                    continue

            # add row with values 1 for bias
            train_data_B = np.concatenate([train_data_B, np.ones((train_data_B.shape[0], 1))], axis=1)

            # split data into train and validation
            # TODO?

            # calculate X from BX = A
            X = np.linalg.lstsq(train_data_B, train_data_A, rcond=None)[0]

            print(f"[INFO] Error: [{np.linalg.norm(train_data_B @ X - train_data_A):0.8f}]")

            # add row with values 1 for bias
            test_data_B = np.concatenate([test_data_B, np.ones((test_data_B.shape[0], 1))], axis=1)

            # calculate transformed data from endpoint B to A using X
            A_test_new = test_data_B @ X
            
            # binarize the data
            A_test_new = np.where(A_test_new > THRESHOLD, 1, 0)
            
            test_values[i] = A_test_new
        
        # increase counter
        new_idx += INDEX_STEP

        # get the most common value from the 5 tries
        A_test_new = np.median(test_values, axis=0)
        # A_test_new = np.mean(test_values, axis=0)

        # concatenate new data to the old one
        data_A = np.concatenate([data_A, A_test_new], axis=0)


    # save data to the file
    # The submission should be an *.npz file with two fields: ids – an array of 20k indices from the SybilAttack
    # dataset in the same order save to .npz file with two fields: ids – all_indices, data – data_A
    if not os.path.exists("./data"): os.mkdir("data")
    np.savez("data/example_submission_binary.npz", ids=np.array(all_indices), representations=data_A)


def get_random_solution():
    all_indices = get_all_indices()
    # data_A = (np.random.normal(loc=0., scale=1.7, size=(20000, 384)) > 0.05).astype(float)
    data_A = np.zeros((20000, 384), dtype=float)

    if not os.path.exists("./data"): os.mkdir("data")
    np.savez("data/example_submission_binary.npz", ids=np.array(all_indices), representations=data_A)


if __name__ == "__main__":
    main_binary()
    
    
