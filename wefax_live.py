import pyaudio
import wave
import time
import os

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import io
from scipy.io.wavfile import read, write
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from PIL import Image
from matplotlib import cm


class DataPacket:
    def __init__(self, filepath, duration, number):
        self.duration = duration
        self.number = number
        self.filepath = filepath
        self.directory = '/'.join(str(self.filepath).split('/')[:-1]) + '/'
        self.sample_rate, self.samples = wavfile.read(self.filepath)

    def spectrogram_chart(self, save: bool = True, show: bool = False):
        frequencies, times, spectrogram = signal.spectrogram(self.samples, self.sample_rate)
        plt.pcolormesh(times, frequencies, spectrogram, cmap='gist_earth')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        max_freq = 5000
        arrange_y = [0, max_freq]
        arrange_labels_y = [f"{int(frequencies[0])}Hz", f"{max_freq}Hz"]
        plt.yticks(arrange_y, arrange_labels_y)

        arrange_x = [0, len(self.samples) / self.sample_rate - 0.03]
        arrange_labels_x = [f"{self.number * self.duration}s", f"{self.number * self.duration + self.duration}s"]
        plt.xticks(arrange_x, arrange_labels_x)

        plt.title(f'{self.filepath}')

        if save:
            plt.savefig(f'{self.directory}{self.number}_chart.png')
        if show:
            plt.show()

    def spectrogram_image(self, save: bool = True):
        frequencies, times, spectrogram = signal.spectrogram(self.samples, self.sample_rate)

        spectrogram_normalized = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        print(np.max(spectrogram_normalized))

        img = Image.fromarray((cm.gist_earth(spectrogram_normalized) * 255).astype(np.uint8))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if save:
            img.save(f'{self.directory}{self.number}_image.png')

        return img

    def __repr__(self):
        return f'part: {self.number * self.duration}s-{self.number * self.duration + self.duration}s packet len: {len(self.samples)}  sample rate:{self.sample_rate}'


class LiveDemodulator:
    def __init__(self, path: str):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 11025

        self.WAVE_OUTPUT_FILENAME = path

        self.saved_chunks = 0

        self.p = pyaudio.PyAudio()

        self.stream = None
        self.connected = False

        self.data_packets = []

    def check_connection_status(func):
        def wrapper(self, *args, **kwargs):
            if self.connected:
                return func(self, *args, **kwargs)
            else:
                print("sound device not connected. choose device before calling other methods")
                return None

        return wrapper

    def get_devices(self):
        return [self.p.get_device_info_by_index(i) for i in range(self.p.get_device_count())]

    def connect(self, device: dict):
        print(f"connecting to {device['name']} on channel {device['index']}")
        try:

            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      output=True,
                                      input_device_index=device['index'],
                                      frames_per_buffer=self.CHUNK)
            print('connected!')
            self.connected = True
            return "success"
        except Exception as e:
            print(e)
            return str(e)

    @check_connection_status
    def record(self, duration):
        print("recording_start")
        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            data = self.stream.read(self.CHUNK)
            frames.append(data)

        print("recording_end")
        filepath = self.WAVE_OUTPUT_FILENAME + str(self.saved_chunks) + '.wav'
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.data_packets.append(DataPacket(filepath, duration, self.saved_chunks))
        self.saved_chunks += 1
        print("file saved")

    @check_connection_status
    def combine(self):
        infiles = [f'{self.WAVE_OUTPUT_FILENAME}{i}.wav' for i in range(self.saved_chunks)]
        outfile = self.WAVE_OUTPUT_FILENAME + "output.wav"

        data = []
        for infile in infiles:
            w = wave.open(infile, 'rb')
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()

        output = wave.open(outfile, 'wb')
        output.setparams(data[0][0])
        for i in range(len(data)):
            output.writeframes(data[i][1])
        output.close()

    def end_stream(self):
        if self.connected:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stream()


if __name__ == "__main__":
    datestamp = str(round(time.time() * 1000))
    save_directory = f"static/temp/{datestamp}/"
    os.mkdir(save_directory)
    with LiveDemodulator(save_directory) as live_demodulator:
        devices = live_demodulator.get_devices()
        for device in devices:
            print(device['index'], device['name'])
            # print(device)
            # print('\n\n\n\n\n')
        live_demodulator.connect(devices[30])
        for x in range(2):
            live_demodulator.record(2)
        live_demodulator.combine()

        for packet in live_demodulator.data_packets:
            packet.spectrogram_chart(save=True, show=True)
            image = packet.spectrogram_image(save=True)
            print(packet)
        live_demodulator.end_stream()
    print('end')
