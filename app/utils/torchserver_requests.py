
from typing import List, Optional, Callable, Tuple, Dict
import requests
import time
import subprocess
import json
import re
import tempfile
import shutil
import os
import logging

TORCH_SERVE_URL_LOAD = "http://torchserve:8081"
TORCH_SERVE_URL = "http://torchserve:8080"


def translate_torchserve(url: str, input_text: list, src: str, tgt: str) -> Tuple[str, List]:
    url = f"{TORCH_SERVE_URL}{url}"
    data = {
        "sample": input_text,
        "additional_info": {
            "src": src,
            "tgt": tgt
        }
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        json_response = response.json()
        translation = json_response["translation"]
        return 'success', translation
    except requests.exceptions.RequestException as e:
        print("Error making the request:", e)
        return 'failure', []
    except json.JSONDecodeError as e:
        print("Error decoding JSON response:", e)
        return 'failure', []
    except KeyError:
        print("Translation not found in the response")
        return 'failure', []


def check_model_load(model_name: str) -> str:
    url = f'{TORCH_SERVE_URL_LOAD}/models'

    response = None
    while not response:
        # logger.info('stuck in loop')
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException:
            time.sleep(1)  # Wait for 1 second before retrying

    if response.status_code == 200:
        data = response.json()
        model_names = [model['modelName'] for model in data['models']]
        if model_name in model_names:
            return 'Success'
        else:
            return 'Failed'

    return 'Failed'
