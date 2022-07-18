import json


class Config:
    def __init__(self):
        self.audio_packet_duration = None
        self.minimum_frames_per_update = None

        self.read_config_file()

    def read_config_file(self):
        with open('config/config.json') as f:
            config_json = json.load(f)
            self.audio_packet_duration = config_json['audio_packet_duration']
            self.minimum_frames_per_update = config_json['minimum_frames_per_update']
