import torch
import numpy as np
import matplotlib.pyplot as plt
from taskdataset import TaskDataset
import torchvision
import requests

path_to_data = "datasets/"
image_id = 0
dataset = torch.load(path_to_data+"ModelStealingPub.pt")

def mix_images(main_image, im, alpha_main = 0.5):
    """ 
    Blend together images with specified opacity
    Returns 'PIL.Image.Image'
    """

    t_main = torchvision.transforms.functional.to_tensor(main_image)
    t_im = torchvision.transforms.functional.to_tensor(im)

    mix_im_t = t_main * alpha_main + t_im * (1-alpha_main)

    mix_im = torchvision.transforms.functional.to_pil_image(mix_im_t)
    
    return mix_im

def calculate_mix_vector(main_vector, vec, alpha=0.5):
    """
    Function multiply vector to find result for pure vector
    """

    assert alpha != 0

    scalar = 1/alpha

    diff = main_vector - vec

    diff *= scalar

    result = diff + main_vector
    return result



SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "n1JQ0vM903jaKbKg"

from datetime import datetime
import json
import pandas as pd
import pickle


def model_stealing(path_to_png_file: str) -> np.ndarray:
    """Send querry to API

    Args:
        path_to_png_file (str): Path to image to send to API (as STRING)

    Raises:
        Exception: In case of request failure

    Returns:
        np.ndarray: Image representation
    """
    
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    with open(path_to_png_file, "rb") as img_file:
        response = requests.get(
            URL, files={"file": img_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            # return response.content["representation"] # obsolete
            print("OK", image_id)
            return json.loads(response.content.decode())["representation"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


def ask_API(image):
    global image_id
    image_id += 1
    image.save(f"saved_png/mix_image_{image_id:05d}.png", format="PNG")
    array = model_stealing(f"saved_png/mix_image_{image_id:05d}.png")

    return np.array(array)

alpha = 0.5

def get_vectors(images):

    main_im = images[0]
    main_vec = ask_API(main_im)
    df = pd.DataFrame({"image":[main_im], 
                  "output": [main_vec]})
    for i in range(1, len(images)):
    # for i in range(1, 3):
        mix_im = mix_images(main_im, images[i], alpha)
        mix_vec = ask_API(mix_im)
        new_vec = calculate_mix_vector(mix_vec, main_vec, alpha)
        df.loc[len(df.index)] = [images[i], new_vec]
        if i%5 == 0:
            df.to_pickle(f'data_save{i}.pck')

    df.to_pickle('data.pck')

from datetime import datetime

# def test_vectors(images):
#     timestamp = datetime.timestamp(datetime.now())
#     main_im = images[0]
#     main_vec = ask_API(main_im)
#     df = pd.DataFrame({"image":[main_im], 
#                         "output": [main_vec]})
#     mix_im = mix_images(main_im, images[1], 0.5)
#     mix_vec = ask_API(mix_im)

#     df.loc[len(df.index)] = [mix_im, mix_vec]
#     unmix_im = images[1]
#     unmix_vec = ask_API(unmix_im)

#     df.loc[len(df.index)] = [unmix_im, unmix_vec]
#     df.to_pickle('test_data.pck')
#     return df


def get_pure_vectors(images):
    first = images[0]
    vec = ask_API(first)
    df = pd.DataFrame({"image":[first], 
                  "output": [vec]})
    for i in range(1, len(images)):
        im = images[i]
        vec = ask_API(im)
        df.loc[len(df.index)] = [im, vec]
        if i%5 == 0:
            df.to_pickle(f'data_save{i}.pck')

    df.to_pickle('data.pck')


get_pure_vectors(dataset.imgs)

