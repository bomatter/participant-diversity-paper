"""
This module provides a Python object that contains the configurations from the user_config.yml file.
The user_config.yml file should be in the parent directory of this file.

Example usage:

from core.user_config import user_config
print(user_config['data']['TUAB']['bids_root'])

"""

import yaml
from pathlib import Path

config_path = Path(__file__).parent / "../user_config.yml"

if not config_path.exists():
    raise FileNotFoundError(
        f"Configuration file does not exist. Please create the file by copying " +
        "the user_config.example.yml file and adjusting the configurations."
    )

with open(config_path, 'r') as ymlfile:
    user_config = yaml.safe_load(ymlfile)