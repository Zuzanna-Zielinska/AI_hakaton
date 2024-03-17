import os
import json
import requests
import argparse

from utils import _sybil_submit


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Send solution to server")
    parser.add_argument("-n", "--name", type=str, help="File name")
    parser.add_argument("-e", "--encoding", choices=["affine", "binary"], type=str, help="Endpoint encoding")
    args = parser.parse_args()
    file_name = args.name
    endpoint_enc = args.encoding
    if os.path.exists(f"./{file_name}"):
        _sybil_submit(endpoint_enc, file_name)
    else: print(f"[ERROR] File '{file_name}' does not exist")
