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
import threading

from PIL import Image
from matplotlib import cm

import scipy

import math
import scipy.fftpack


class DataPacket:
    def __init__(self, filepath, duration, number):
        self.duration = duration
        self.number = number
        self.filepath = filepath
        self.directory = '/'.join(str(self.filepath).split('/')[:-1]) + '/'
        self.sample_rate, self.samples = wavfile.read(self.filepath)
        self.fft_filepath = f'{self.directory}{self.number}_fft.png'
        self.chart_filepath = f'{self.directory}{self.number}_chart.png'
        self.spectrum_filepath = f'{self.directory}{self.number}_spectrum.png'

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
            plt.savefig(self.chart_filepath)
        if show:
            plt.show()

    def spectrogram_image(self, save: bool = True):
        frequencies, times, spectrogram = signal.spectrogram(self.samples, self.sample_rate, mode="magnitude")

        spectrogram_normalized = spectrogram / ((np.max(spectrogram)) + 0.0001)

        img = Image.fromarray((cm.gist_earth(spectrogram_normalized) * 255).astype(np.uint8))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.ROTATE_90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if save:
            img.save(self.spectrum_filepath)

        return img

    def find_sync_pulse(self, save: bool = True, show: bool = False):

        fft = np.fft.fft(self.samples)

        N = len(fft)
        n = np.arange(N)
        T = N / self.sample_rate
        freq = n / T

        n_oneside = N // 2
        freqs_one_side = freq[:n_oneside]
        amplitude_one_size = abs(fft[:n_oneside] / n_oneside)

        normalized_amplitude = amplitude_one_size / max(amplitude_one_size)
        peaks = scipy.signal.find_peaks(normalized_amplitude, distance=200, height=0.2)

        if save or show:
            plt.clf()

            plt.plot(freqs_one_side, normalized_amplitude)
            plt.plot(freqs_one_side[peaks[0]], peaks[1]['peak_heights'], "x")
            if save:
                plt.savefig(self.fft_filepath)
            if show:
                plt.show()

        found_peaks = peaks[0]
        print(found_peaks)

        for peak in freqs_one_side[peaks[0]]:
            if not 1000 < peak < 3000:
                return False

        if 4 <= len(found_peaks) <= 6:
            return True
        else:
            return False

    def process(self):
        filtered_signal = self.__demodulate(self.samples)
        digitalized_signal = self.__digitalize(filtered_signal)
        return digitalized_signal

    @staticmethod
    def __demodulate(data):
        hilbert_signal = scipy.signal.hilbert(data)
        filtered_signal = scipy.signal.medfilt(np.abs(hilbert_signal), 5)
        return filtered_signal

    @staticmethod
    def __digitalize(data):
        plow = 0.5
        phigh = 99.5
        (low, high) = np.percentile(data, (plow, phigh))
        delta = high - low
        digitalized = np.round(255 * (data - low) / (delta+0.000001))
        digitalized[digitalized < 0] = 0
        digitalized[digitalized > 255] = 255
        return [int(point) for point in digitalized]

    def __repr__(self):
        return f'part: {self.number * self.duration}s-{self.number * self.duration + self.duration}s packet len: {len(self.samples)}  sample rate:{self.sample_rate}'


class LiveDemodulator:
    def __init__(self, path: str, tcp_stream: bool = True):
        # ---- constants  ---- #
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 11025
        self.OUTPUT_DIRECTORY = path
        self.PACKET_DURATION = 1
        self.LINES_PER_MINUTE = 120
        self.TIME_FOR_ONE_FRAME = 1 / (self.LINES_PER_MINUTE / 60)  # in s
        self.SAMPLES_FOR_ONE_FRAME = int(self.TIME_FOR_ONE_FRAME * self.RATE)
        self.MINIMUM_FRAMES_PER_UPDATE = 10
        # -------------------- #

        # ---- audio info ---- #
        self.start_tone_found = False
        self.phasing_signal_found = False
        self.image_process = "not started"
        self.stop_tone_found = False
        self.black_found = False
        # -------------------- #

        # ---- variables ---- #
        self.data_packets = []
        self.data_points = []
        self.threads = []
        self.spectrum_websocket_stack = []
        self.frames_websocket_stack = []
        self.connected = False
        self.isRecording = False
        self.amount_peaks_found = 0
        self.saved_chunks = 0
        self.saved_frames = 0
        # -------------------- #

        self.p = pyaudio.PyAudio()
        self.stream = tcp_stream

    def check_connection_status(func):
        def wrapper(self, *args, **kwargs):
            if self.connected:
                return func(self, *args, **kwargs)
            else:
                print("sound device not connected. choose device before calling other methods")
                return "sound device not connected. choose device before calling other methods"

        return wrapper

    def get_devices(self):
        return [self.p.get_device_info_by_index(i) for i in range(self.p.get_device_count())]

    def connect(self, device_index: int):
        device = self.p.get_device_info_by_index(device_index)
        print(f"connecting to {device['name']} on channel {device['index']}")
        try:
            if self.connected:
                self.stream.stop_stream()
                self.stream.close()

            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      output=True,
                                      input_device_index=device['index'],
                                      frames_per_buffer=self.CHUNK)
            print('connected!')
            self.connected = True
            for thread in self.threads:
                thread.join()
            thread1 = threading.Thread(target=self.record_process, args=())
            thread2 = threading.Thread(target=self.convert_process, args=())
            self.threads.append(thread1)
            self.threads.append(thread2)
            thread1.start()
            thread2.start()
            return "success"
        except Exception as e:
            print(e)
            return str(e)

    def convert_process(self):
        while True:
            if len(self.data_points) > self.SAMPLES_FOR_ONE_FRAME * self.MINIMUM_FRAMES_PER_UPDATE:
                frame_points = self.data_points[:self.SAMPLES_FOR_ONE_FRAME * self.MINIMUM_FRAMES_PER_UPDATE]

                w, h = self.SAMPLES_FOR_ONE_FRAME, self.MINIMUM_FRAMES_PER_UPDATE
                img = Image.new('L', (w, h), )
                for y in range(h):
                    for x in range(w):
                        lum = 255 - frame_points[x*(y+1)]
                        img.putpixel((x, y), lum)

                img = img.resize((w, 4 * h))

                min_frames = self.saved_frames*self.MINIMUM_FRAMES_PER_UPDATE
                max_frames = (self.saved_frames+1)*self.MINIMUM_FRAMES_PER_UPDATE
                frame_output_path = f'{self.OUTPUT_DIRECTORY}frame_{min_frames}-{max_frames}.png'
                img.save(frame_output_path)
                self.saved_frames += 1

                self.data_points = self.data_points[self.SAMPLES_FOR_ONE_FRAME * self.MINIMUM_FRAMES_PER_UPDATE:]

                self.frames_websocket_stack.append(frame_output_path)

    def record_process(self):
        while True:
            if self.connected and self.isRecording:
                packet = self.record(self.PACKET_DURATION)
                img = packet.spectrogram_image(save=True)
                # packet.spectrogram_chart(save=True)
                if not self.start_tone_found:
                    if packet.find_sync_pulse(save=False, show=False):
                        self.amount_peaks_found += 1
                    else:
                        self.amount_peaks_found = 0
                    if self.amount_peaks_found * self.PACKET_DURATION >= 4:
                        self.start_tone_found = True

                self.data_points += packet.process()

                if self.stream:
                    json_message = {"width": int(img.width),
                                    "height": int(img.height),
                                    "src": str(packet.spectrum_filepath),
                                    "length": float(packet.duration)}
                    self.spectrum_websocket_stack.append(json_message)

    @check_connection_status
    def start_recording(self):
        self.isRecording = True
        print("recording started")
        return "recording started"

    @check_connection_status
    def stop_recording(self):
        if self.isRecording:
            self.isRecording = False
            print("recording stopped")
            return "recording stopped"
        else:
            print("recording need to be started in order to stop it")
            return "recording need to be started in order to stop it"

    @check_connection_status
    def record(self, duration):
        print("recording_start")
        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            data = self.stream.read(self.CHUNK)
            frames.append(data)

        print("recording_end")
        filepath = self.OUTPUT_DIRECTORY + str(self.saved_chunks) + '.wav'
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.data_packets.append(DataPacket(filepath, duration, self.saved_chunks))
        self.saved_chunks += 1
        print("file saved")
        return DataPacket(filepath, duration, self.saved_chunks)

    @check_connection_status
    def combine(self):
        infiles = [f'{self.OUTPUT_DIRECTORY}{i}.wav' for i in range(self.saved_chunks)]
        outfile = self.OUTPUT_DIRECTORY + "output.wav"

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
        return outfile

    def end_stream(self):
        if self.connected:
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
                "black_found": self.black_found
                }

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
        live_demodulator.connect(30)
        for x in range(2):
            live_demodulator.record(2)
        live_demodulator.combine()

        for packet in live_demodulator.data_packets:
            packet.spectrogram_chart(save=True, show=True)
            image = packet.spectrogram_image(save=True)
            print(packet)
        live_demodulator.end_stream()
    print('end')
