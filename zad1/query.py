import json
import requests
import argparse
from typing import List

import numpy as np


SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "n1JQ0vM903jaKbKg"


def model_stealing_query(path_to_png_file: str) -> list:
    """Send querry to API

    Args:
        path_to_png_file (str): Path to image

    Raises:
        Exception: In case of request failure, returns error code
    """
    
    endpoint = "/modelstealing"
    url = SERVER_URL + endpoint
    with open(path_to_png_file, "rb") as f:
        print(f"[INFO] QUERING with [{path_to_png_file}]")
        response = requests.get(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("[INFO] QUERY OK")
            representation = response.json()["representation"]
            return representation
        else:
            raise Exception(f"Model stealing failed. Code: {response.status_code}, content: {response.json()}")

    
def model_stealing_reset() -> None:
    """Resets API

    Raises:
        Exception: In case of query failure, returns error code
    """
    
    endpoint = f"/modelstealing/reset"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print("[INFO] RESET OK")
        print(response.json())
    else:
        raise Exception(f"Model stealing reset failed. Code: {response.status_code}, content: {response.json()}")


def model_stealing_submit(path_to_onnx_file: str) -> None:
    """Submits solution

    Args:
        path_to_onnx_file (str): Path to saved model (in ONNX format)

    Raises:
        Exception: In case of query failure, returns error code

    Returns:
        float: Solution score
    """
    
    endpoint = "/modelstealing/submit"
    url = SERVER_URL + endpoint
    with open(path_to_onnx_file, "rb") as f:
        print(f"[INFO] QUERING with [{path_to_onnx_file}]")
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("[INFO] SUBMIT OK")
            print(response.json())
        else:
            # print("[DEBUG] JSON content:", response.content)
            try: 
                raise Exception(f"Model stealing submit failed. Code: {response.status_code}, content: {response.json()}")
            except Exception as err:
                raise Exception(f"Cannot parse JSON, content: {err}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Queries server. Args priority: (1) [-f], (2) [-s], (3) [-r]")
    parser.add_argument("-f", "--filename", type=str, default=None, help="Path to image in PNG format")
    parser.add_argument("-s", "--submit", type=str, default=None,help="Path to file in ONNX format")
    parser.add_argument("-r", "--reset", type=bool, default=False, help="Queries server reset")
    args = parser.parse_args()
    
    if args.filename is not None:
        resp = model_stealing_query(args.filename)
    elif args.submit is not None:
        model_stealing_submit(args.submit)
    elif args.reset:
        model_stealing_reset()
