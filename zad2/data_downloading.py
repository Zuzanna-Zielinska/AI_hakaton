# Description: This file contains the code for downloading data from the server
from enum import Enum

import numpy as np
import os

from typing import List

class EndpointType(Enum):
    BASE = "home"
    DEFENDER = "defense"

class EndpointEncoding(Enum):
    BINARY = "binary"
    AFFINE = "affine"


from taskdataset import TaskDataset
from utils import get_all_indices, download_data, reset_endpoint, QUERY_MAX_ITEMS


REQ_INDEXES = 385
INDEX_STEP  = QUERY_MAX_ITEMS - REQ_INDEXES


def main_afinic():
    # reset endpoint A - base endpoint
    reset_endpoint(EndpointType.BASE.value, EndpointEncoding.AFFINE.value)

    all_indices = get_all_indices()
    # get 2000 indexes for endpoint A
    idxs_A = all_indices[:2000]

    # get 2000 values from endpoint A
    data_A = download_data(idxs_A, EndpointType.BASE.value, EndpointEncoding.AFFINE.value)

    # get REQ_INDEXES indexes from endpoint A
    # REQ_INDEXES because it is affinic transformation matrix, so we need to have 384 rows of data
    # and +1 for the bias
    idxs_B = idxs_A[:REQ_INDEXES]

    train_data_A = data_A[:REQ_INDEXES]

    new_idx = 2000

    # while we don't have all dataset - 20000 data:
    while new_idx < 20000:
        
        print(f"[DEBUG] Current index range: [{new_idx}, {new_idx+INDEX_STEP}]")
        
        # reset endpoint B - defender endpoint
        reset_endpoint(EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)
        
        try:
            # get REQ_INDEXES values from endpoint B
            train_data_B = download_data(idxs_B, EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)

            # get INDEX_STEP values from endpoint B
            test_data_B = download_data(all_indices[new_idx:new_idx+INDEX_STEP], EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)
        
        except Exception as err:
            print(f"[WARNING] ERROR occured, content: [{err}]")
            continue
        
        # add row with values 1 for bias
        train_data_B = np.concatenate([train_data_B, np.ones((train_data_B.shape[0], 1))], axis=1)

        # calculate X from BX = A
        X = np.linalg.lstsq(train_data_B, train_data_A, rcond=None)[0]
        
        # increase counter
        new_idx += INDEX_STEP

        # add row with values 1 for bias
        test_data_B = np.concatenate([test_data_B, np.ones((test_data_B.shape[0], 1))], axis=1)

        # calculate transformed data from endpoint B to A using X
        A_test_new = test_data_B @ X

        # concatenate new data to the old one
        data_A = np.concatenate([data_A, A_test_new], axis=0)
        
        print(f"[INFO] Error: [{np.linalg.norm(train_data_B @ X - train_data_A):0.8f}]")

    # save data to the file
    # The submission should be an *.npz file with two fields: ids – an array of 20k
    # indices from the SybilAttack dataset in the same order

    # save to .npz file with two fields: ids – all_indices, data – data_A

    # data_to_save = {"ids": all_indices, "data": data_A}
    if os.path.exists("./data"): os.mkdir("data")
    np.savez(
        "data/example_submission.npz",
        ids=np.array(all_indices),
        representations=data_A
    )


if __name__ == "__main__":
    main_afinic()
