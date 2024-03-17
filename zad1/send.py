
import json
import requests
from typing import List

import numpy as np

SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "n1JQ0vM903jaKbKg"

def model_stealing_submission(path_to_onnx_file: str) -> float:
    """Submits solution

    Args:
        path_to_onnx_file (str): Path to saved model (as STRING, in ONNX format)

    Raises:
        Exception: In case of query failure

    Returns:
        float: Solution score
    """

    ENDPOINT = "/modelstealing/submit"
    URL = SERVER_URL + ENDPOINT

    with open(path_to_onnx_file, "rb") as onnx_file:
        response = requests.post(
            URL, files={"file": onnx_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return response.content["score"]
            # return json.loads(response.content.decode())["score"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
        
model_stealing_submission("zad1/models/zad1_model_e8.onnx")
        
# import os

# #check file in folser
# print(os.listdir('zad1'))

