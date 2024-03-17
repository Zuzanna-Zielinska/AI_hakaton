import json
import time
import requests
from typing import List

import torch
import numpy as np
from taskdataset import TaskDataset


SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "n1JQ0vM903jaKbKg"
QUERY_MAX_ITEMS = 2000


def _sybil_query(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise Exception(f"Invalid endpoint: [{home_or_defense}:{binary_or_affine}]")

    print(f"[INFO] QUERING [{home_or_defense}/{binary_or_affine}] for [{len(ids)}]")

    endpoint = f"/sybil/{binary_or_affine}/{home_or_defense}"
    url = SERVER_URL + endpoint
    ids = ",".join(map(str, ids))
    response = requests.get(url, params={"ids": ids}, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print(f"[INFO] QUERY [{home_or_defense}/{binary_or_affine}] response OK")
        representations = response.json()["representations"]
        ids = response.json()["ids"]
        return representations
    else:
        raise Exception(f"Sybil failed. Code: {response.status_code}, content: {response.json()}")


# Be careful. This can be done only 4 times an hour.
# Make sure your file has proper content.
def _sybil_submit(binary_or_affine: str, path_to_npz_file: str):
    if binary_or_affine not in ["binary", "affine"]:
        raise Exception(f"Invalid endpoint: [{binary_or_affine}]")

    endpoint = f"/sybil/{binary_or_affine}/submit"
    url = SERVER_URL + endpoint

    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})

    if response.status_code == 200:
        print("[INFO] Request OK")
        print(response.json())
    else:
        print(f"Request submit failed. Status code: {response.status_code}, content: {response.json()}")


def _sybil_reset(home_or_defense: str, binary_or_affine: str, ):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise Exception(f"Invalid endpoint: [{home_or_defense}:{binary_or_affine}]")

    endpoint = f"/sybil/{binary_or_affine}/reset/{home_or_defense}"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print(f"[INFO] RESET [{home_or_defense}/{binary_or_affine}] request OK")
        print(response.json())
    else:
        raise Exception(f"Sybil reset failed. Code: {response.status_code}, content: {response.json()}")


def get_all_indices() -> list:
    dataset = torch.load("./SybilAttack.pt")
    return dataset.ids


def download_data(id_numbers: list, endpoint_type: str, endpoint_encoding: str) -> np.ndarray:
    if len(id_numbers) > QUERY_MAX_ITEMS: raise Exception(f"Numbe of requested indices exceeds limit of {QUERY_MAX_ITEMS} ({len(id_numbers)})")
    max_items = len(id_numbers)
    step = 500
    cnt = 0
    accu = []
    while cnt < max_items:
        id_subset = id_numbers[cnt:cnt+step]
        resp = _sybil_query(id_subset, endpoint_type, endpoint_encoding)
        cnt += step
        accu.extend(resp)
        time.sleep(1)
    accu = np.array(accu)
    print(f"[DEBUG] Downloaded data: [{accu.shape}]")
    return accu


def reset_endpoint(endpoint_type: str, endpoint_encoding: str) -> None:
    _sybil_reset(endpoint_type, endpoint_encoding)
    time.sleep(1)
