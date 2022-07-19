import json


class Config:
    def __init__(self):
        self.settings = {}

        self.read_config_file()

    def read_config_file(self):
        with open('config/config.json') as f:
            config_json = json.load(f)
            for key, value in config_json.items():
                self.settings[key] = value
