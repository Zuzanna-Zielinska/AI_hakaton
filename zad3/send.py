import os
import json
import requests
import argparse


SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "n1JQ0vM903jaKbKg"


# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.
def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        print(f"[INFO] QUERING with [{path_to_npz_file}]")
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("[INFO] Request OK")
            print(response.json())
        else:
            raise Exception(f"Defense submit failed. Code: {response.status_code}, content: {response.json()}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Send solution to server")
    parser.add_argument("-n", "--name", type=str, help="File name")
    args = parser.parse_args()
    file_name = args.name
    if os.path.exists(f"./{file_name}"):
        defense_submit(file_name)
    else: print(f"[ERROR] File '{file_name}' does not ecist")
