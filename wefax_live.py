from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import wavfile
from matplotlib import cm
from scipy import signal
from io import BytesIO
from PIL import Image
import scipy.fftpack
import numpy as np
import threading
import pyaudio
import base64
import scipy
import wave
import time
import sys
import os
import math

from colored_text import debug_log, Colors
from data_packet import DataPacket
from config import Config

config = Config()


stop_threads = False


class LiveDemodulator:
    def __init__(self, path: str):

        # ---- constants  ---- #
        self.RATE = config.settings['recording_settings']['sample_rate']
        self.CHUNK = self.RATE
        self.FORMAT = pyaudio.paInt16
        self.NUMPY_FORMAT = np.int16
        self.CHANNELS = 1
        self.OUTPUT_DIRECTORY = path
        self.AUDIO_PACKET_DURATION = config.settings['recording_settings']['audio_packet_duration']
        self.LINES_PER_MINUTE = 120
        self.TIME_FOR_ONE_FRAME = 1 / (self.LINES_PER_MINUTE / 60)  # in s
        self.SAMPLES_FOR_ONE_FRAME = int(self.TIME_FOR_ONE_FRAME * self.RATE)
        self.MINIMUM_FRAMES_PER_UPDATE = config.settings['recording_settings']['minimum_frames_per_update']
        # -------------------- #

        # -- debug variables -- #
        self.__save_audio_packets = config.settings['debug_settings']['save_audio_packets']
        self.__save_spectrum = config.settings['debug_settings']['save_spectrum']
        self.__save_frames = config.settings['debug_settings']['save_frames']
        # --------------------- #

        # ---- audio info ---- #
        self.start_tone_found = False
        self.phasing_signal_found = False
        self.image_process = "not started"
        self.stop_tone_found = False
        self.black_found = False
        # -------------------- #

        # ---- variables ---- #
        self.data_packets = []
        self.data_points = np.empty(1, dtype=int)
        self.threads = []
        self.audio_frames = []
        self.image_frames = []
        self.spectrum_websocket_stack = []
        self.frames_websocket_stack = []
        self.connected = False
        self.isRecording = False
        self.__amount_start_peaks_found = 0
        self.__amount_stop_peaks_found = 0
        self.saved_chunks = 0
        self.saved_lines = 0
        self.saved_frames = 0
        self.device_id = 0
        # -------------------- #

        self.p = pyaudio.PyAudio()
        self.stream = None

    def check_connection_status(func):
        def wrapper(self, *args, **kwargs):
            if self.connected:
                return func(self, *args, **kwargs)
            else:
                debug_log("sound device not connected. choose device before calling other methods", Colors.error)
                return "sound device not connected. choose device before calling other methods"

        return wrapper

    def change_lines_per_minute(self, lpm: int):
        old_lpm = self.LINES_PER_MINUTE
        self.LINES_PER_MINUTE = lpm
        self.TIME_FOR_ONE_FRAME = 1 / (self.LINES_PER_MINUTE / 60)  # in s
        self.SAMPLES_FOR_ONE_FRAME = int(self.TIME_FOR_ONE_FRAME * self.RATE)
        debug_log(f"changed lines per minute from {old_lpm} to {lpm}", Colors.warning)
        return f"changed lines per minute from {old_lpm} to {lpm}"

    def connect(self, device_index: int):
        global stop_threads
        device = self.p.get_device_info_by_index(device_index)
        debug_log(f"connecting to {device['name']} on channel {device['index']}", Colors.warning)
        self.device_id = device['index']
        try:
            self.connected = True
            for thread in self.threads:
                thread.join()
            stop_threads = False
            thread1 = threading.Thread(target=self.record_thread, args=())
            thread1.setDaemon(True)
            thread2 = threading.Thread(target=self.convert_points_to_frames_thread, args=())
            thread2.setDaemon(True)
            self.threads.append(thread1)
            self.threads.append(thread2)
            thread1.start()
            thread2.start()
            return "success"
        except Exception as e:
            debug_log(e, Colors.error)
            return str(e)

    def convert_points_to_frames_thread(self):
        global stop_threads
        if self.data_points.shape[0] > self.SAMPLES_FOR_ONE_FRAME * self.MINIMUM_FRAMES_PER_UPDATE:
            self.saved_lines += self.MINIMUM_FRAMES_PER_UPDATE

            if not self.stop_tone_found:
                self.image_process = f"converted {self.saved_lines} lines"

            debug_log("frame_convert", Colors.debug)
            frame_points = self.data_points[:self.SAMPLES_FOR_ONE_FRAME * self.MINIMUM_FRAMES_PER_UPDATE]

            frame_width = int(self.TIME_FOR_ONE_FRAME * self.RATE)
            w, h = frame_width, len(frame_points) // frame_width
            img = Image.new('L', (w, h), )
            px, py = 0, 0
            for p in range(len(frame_points)):
                lum = 255 - int(frame_points[p])
                img.putpixel((px, py), lum)
                px += 1
                if px >= w:
                    px = 0
                    py += 1
                    if py >= h:
                        break

            img = img.resize((w, 4 * h))

            min_frames = self.saved_frames * self.MINIMUM_FRAMES_PER_UPDATE
            max_frames = (self.saved_frames + 1) * self.MINIMUM_FRAMES_PER_UPDATE
            frame_output_path = f'{self.OUTPUT_DIRECTORY}frame_{min_frames}-{max_frames}.png'

            if self.__save_frames:
                img.save(frame_output_path)

            self.saved_frames += 1
            self.image_frames.append(img)

            self.data_points = self.data_points[self.SAMPLES_FOR_ONE_FRAME * self.MINIMUM_FRAMES_PER_UPDATE:]

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
            self.frames_websocket_stack.append(img_str)

            self.convert_points_to_frames_thread()
        else:
            if not stop_threads:
                timer = threading.Timer(1.0, self.convert_points_to_frames_thread)
                timer.setDaemon(True)
                timer.start()

    def __state_machine(self, __packet):
        if not self.start_tone_found:
            if __packet.contain_start_tone():
                self.__amount_start_peaks_found += 1
            else:
                self.__amount_start_peaks_found = 0
            if self.__amount_start_peaks_found * self.AUDIO_PACKET_DURATION >= 4:
                self.start_tone_found = True
                debug_log("start tone found", Colors.info)

        if self.start_tone_found and not self.phasing_signal_found:
            pulse_info = __packet.find_sync_pulse()
            if pulse_info["pulse_found"] is True:
                self.phasing_signal_found = True
                self.data_points = __packet.samples[pulse_info["peaks_samples"][-1]:]
                debug_log("sync pulse found", Colors.info)

        if self.start_tone_found and self.phasing_signal_found:
            if __packet.contain_stop_tone():
                self.__amount_stop_peaks_found += 1
            else:
                self.__amount_stop_peaks_found = 0
            if self.__amount_stop_peaks_found * self.AUDIO_PACKET_DURATION >= 4:
                self.stop_tone_found = True
                self.image_process = "image converted"
                debug_log("stop tone found", Colors.info)

    def __process_audio_samples(self, __samples):

        __packet = DataPacket(self.RATE,
                              __samples,
                              self.LINES_PER_MINUTE,
                              self.OUTPUT_DIRECTORY,
                              self.AUDIO_PACKET_DURATION,
                              self.saved_chunks)

        self.data_packets.append(__packet)

        img = __packet.spectrogram_image(save=self.__save_spectrum)

        self.__state_machine(__packet)

        self.data_points = np.append(self.data_points, __packet.samples)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]

        json_message = {"width": int(img.width),
                        "height": int(img.height),
                        "src": str(img_str),
                        "length": float(__packet.duration)}

        self.spectrum_websocket_stack.append(json_message)

    def record_thread(self):
        global stop_threads
        while True:
            if stop_threads:
                break
            if self.connected and self.isRecording:
                samples = self.__record(self.AUDIO_PACKET_DURATION)
                thread = threading.Thread(target=self.__process_audio_samples, args=(samples,))
                thread.setDaemon(True)
                self.threads.append(thread)
                thread.start()

    @check_connection_status
    def start_recording(self):
        global stop_threads

        if self.stream:
            stop_threads = True
            self.p.terminate()
            self.stream.stop_stream()
            self.stream.close()

        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  input_device_index=self.device_id,
                                  frames_per_buffer=self.RATE * self.AUDIO_PACKET_DURATION)

        self.isRecording = True
        debug_log("recording started", Colors.warning)
        return "recording started"

    @check_connection_status
    def stop_recording(self):
        if self.isRecording:
            self.isRecording = False
            debug_log("recording stopped", Colors.warning)
            self.create_image()
            return "recording stopped"
        else:
            debug_log("recording need to be started in order to stop it", Colors.error)
            return "recording need to be started in order to stop it"

    @check_connection_status
    def __record(self, duration):
        data = self.stream.read(self.RATE * duration)
        self.audio_frames.append(data)
        value_frames = np.fromstring(data, self.NUMPY_FORMAT)

        debug_log(f"recorded packet {self.saved_chunks}...", Colors.passed)

        if self.__save_audio_packets:
            filepath = self.OUTPUT_DIRECTORY + str(self.saved_chunks) + '.wav'
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join([data]))
            wf.close()
            debug_log("file saved", Colors.info)

        self.saved_chunks += 1
        return value_frames

    @check_connection_status
    def create_image(self):
        outfile = self.OUTPUT_DIRECTORY + "output.png"

        max_width = max([frame.width for frame in self.image_frames])
        img = Image.new('RGB', (max_width, 0))

        for frame in self.image_frames:
            temp_image = Image.new('RGB', (max_width, img.height + frame.height))
            temp_image.paste(img, (0, 0))
            temp_image.paste(frame, (0, img.height))
            img = temp_image

        img.save(outfile)

        debug_log("image saved", Colors.info)

        return outfile

    @check_connection_status
    def combine(self):
        outfile = self.OUTPUT_DIRECTORY + "output.wav"

        wf = wave.open(outfile, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        debug_log("file saved", Colors.info)

        return outfile

    def clear_image(self):
        self.image_frames = []
        debug_log("image cleared", Colors.info)
        return "image cleared"

    def end_stream(self):
        global stop_threads
        stop_threads = True
        for thread in self.threads:
            thread.join()
        if self.connected and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def audio_info(self):
        audio_length = sum([data_packet.duration for data_packet in self.data_packets])
        return {"data_packets": len(self.data_packets),
                "audio_length": audio_length,
                "start_tone_found": self.start_tone_found,
                "phasing_signal_found": self.phasing_signal_found,
                "image_process": self.image_process,
                "stop_tone_found": self.stop_tone_found,
                }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stream()
