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

#
# def download_data(id_numbers: List[int], endpoint_type: str, endpoint_encoding: str) -> List[List]:
#     pass
#
#
# def reset_endpoint(endpoint_type:str, endpoint_encoding:str) -> None:
#     pass
#
#
# def get_all_indices() -> List[int]:
#     pass


def main_afinic():
    # reset endpoint A - base endpoint
    reset_endpoint(EndpointType.BASE.value, EndpointEncoding.AFFINE.value)
    # reset endpoint B - defender endpoint
    reset_endpoint(EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)

    all_indices = get_all_indices()
    # get 2000 indexes for endpoint A
    idxs_A = all_indices[:2000]

    # get 2000 values from endpoint A
    data_A = download_data(idxs_A, EndpointType.BASE.value, EndpointEncoding.AFFINE.value)

    # get 385 indexes from endpoint A
    # 385 because it is affinic transformation matrix, so we need to have 384 rows of data
    # and +1 for the bias
    idxs_B = idxs_A[:385]

    train_data_A = data_A[:385]

    new_idx = 2000

    # while we don't have all dataset - 20000 data:
    while new_idx < 20000:
        # get 385 values from endpoint B
        train_data_B = download_data(idxs_B, EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)

        # add row with values 1 for bias
        train_data_B = np.concatenate([train_data_B, np.ones((train_data_B.shape[0], 1))], axis=1)

        # calculate X from BX = A
        X = np.linalg.lstsq(train_data_B, train_data_A, rcond=None)[0]

        # get 1615 values from endpoint B
        test_data_B = download_data(all_indices[new_idx:new_idx+1615], EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)
        new_idx += 1615

        # add row with values 1 for bias
        test_data_B = np.concatenate([test_data_B, np.ones((test_data_B.shape[0], 1))], axis=1)

        # calculate transformed data from endpoint B to A using X
        A_test_new = test_data_B @ X

        # concatenate new data to the old one
        data_A = np.concatenate([data_A, A_test_new], axis=0)

        # reset endpoint B
        reset_endpoint(EndpointType.DEFENDER.value, EndpointEncoding.AFFINE.value)

    # save data to the file
    # The submission should be an *.npz file with two fields: ids – an array of 20k
    # indices from the SybilAttack dataset in the same order

    # save to .npz file with two fields: ids – all_indices, data – data_A

    data_to_save = {"ids": all_indices, "data": data_A}
    np.savez("submission.npz", **data_to_save)
