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

from config import Config

config = Config()

from colored_text import debug_log, Colors

stop_threads = False


class DataPacket:
    def __init__(self, sample_rate, samples, lines_per_minute, directory, duration, number):

        # notch filter constants
        self.__NOTCH_FILTER_FREQUENCY = config.settings['notch_filter_settings']['notch_filter_frequency']
        self.__NOTCH_FILTER_QUALITY_FACTOR = config.settings['notch_filter_settings']['notch_filter_quality_factor']

        # start tone constants
        __start_tone_settings = config.settings['start_tone_settings']
        self.__START_TONE_PEAKS_MINIMUM_DISTANCE = __start_tone_settings['peaks_minimum_distance']
        self.__START_TONE_PEAKS_MINIMUM_HEIGHT = __start_tone_settings['peaks_minimum_height']
        self.__START_TONE_PEAKS_MINIMUM_PROMINENCE = __start_tone_settings['peaks_minimum_prominence']
        self.__START_TONE_PEAKS_MINIMUM_FREQUENCY = __start_tone_settings['peaks_minimum_frequency']
        self.__START_TONE_PEAKS_MAXIMUM_FREQUENCY = __start_tone_settings['peaks_maximum_frequency']
        self.__START_TONE_PEAKS_MINIMUM_AMOUNT = __start_tone_settings['peaks_minimum_amount']
        self.__START_TONE_PEAKS_MAXIMUM_AMOUNT = __start_tone_settings['peaks_maximum_amount']

        # sync pulse constants
        __sync_pulse_settings = config.settings['sync_pulse_settings']
        self.__SYNC_PULSE_PEAKS_MINIMUM_HEIGHT = __sync_pulse_settings['peaks_minimum_height']
        self.__SYNC_PULSE_PEAKS_MINIMUM_PROMINENCE = __sync_pulse_settings['peaks_minimum_prominence']
        self.__SYNC_PULSE_PEAKS_MINIMUM_FREQUENCY = __sync_pulse_settings['peaks_minimum_frequency']
        self.__SYNC_PULSE_PEAKS_MAXIMUM_FREQUENCY = __sync_pulse_settings['peaks_maximum_frequency']
        self.__SYNC_PULSE_PEAKS_MINIMUM_AMOUNT = math.floor(1 / (lines_per_minute / 60) / duration)
        self.__SYNC_PULSE_PEAKS_MAXIMUM_AMOUNT = math.ceil(1 / (lines_per_minute / 60) / duration)

        # audio packet info
        self.lines_per_minute = lines_per_minute
        self.duration = duration
        self.number = number
        self.directory = directory
        self.sample_rate = sample_rate

        # audio data
        self.raw_samples = samples
        self.samples = self.__process_samples()

        # files paths
        self.start_tone_chart_filepath = f'{self.directory}{self.number}_start_tone_chart.png'
        self.sync_pulse_chart_filepath = f'{self.directory}{self.number}_sync_pulse_chart.png'

        self.fft_chart_filepath = f'{self.directory}{self.number}_fft_chart.png'
        self.demodulated_chart_filepath = f'{self.directory}{self.number}_demodulated_chart.png'
        self.spectrogram_chart_filepath = f'{self.directory}{self.number}_spectrogram_chart.png'
        self.spectrogram_image_filepath = f'{self.directory}{self.number}_spectrogram_image.png'
        self.audio_chart_filepath = f'{self.directory}{self.number}_audio_chart.png'
        self.processed_chart_filepath = f'{self.directory}{self.number}_processed_chart.png'

    def fft_chart(self, show: bool = False):
        fft = np.fft.fft(self.raw_samples)

        N = len(fft)
        n = np.arange(N)
        T = N / self.sample_rate
        freq = n / T

        n_oneside = N // 2
        freqs_one_side = freq[:n_oneside]
        amplitude_one_size = abs(fft[:n_oneside] / n_oneside)
        normalized_amplitude = amplitude_one_size / (max(amplitude_one_size) + 0.0001)

        plt.plot(freqs_one_side, normalized_amplitude)

        plt.savefig(self.fft_chart_filepath)
        debug_log("fft chart saved", Colors.debug)

        if show:
            plt.show()
        plt.clf()

    def audio_chart(self, show: bool = False):
        Time = np.linspace(0, len(self.samples) / self.sample_rate, num=len(self.samples))

        plt.figure(1)
        plt.title("Signal Wave...")
        plt.plot(Time, self.samples, '-ok')
        plt.savefig(self.audio_chart_filepath)
        debug_log("audio chart saved", Colors.debug)
        if show:
            plt.show()
        plt.clf()

    def demodulated_chart(self, show: bool = False):
        data_am_crop = self.__demodulate(self.samples)
        plt.plot(data_am_crop)
        plt.savefig(self.demodulated_chart_filepath)
        debug_log("demodulated chart saved", Colors.debug)
        if show:
            plt.show()
        plt.clf()

    def processed_chart(self, show: bool = False):
        am = self.__demodulate(self.samples)
        digitalized = self.__digitalize(am)

        fig, axs = plt.subplots(3, 1)
        axs[0].set_title(f'demodulated signal')
        axs[0].plot(am)
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('value')
        axs[0].grid(True)
        axs[0].set_xlim(left=0)
        axs[0].set_ylim(bottom=0)
        axs[0].set_xlim(right=len(am))

        x = np.arange(0, len(digitalized), 1)
        axs[1].set_title(f'digitalized signal')
        axs[1].step(x, digitalized)
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('lum')
        axs[1].grid(True)
        axs[1].set_xlim(left=0)
        axs[1].set_xlim(right=len(digitalized))
        axs[1].set_ylim(bottom=0)
        axs[1].set_ylim(top=255)

        w, h = len(digitalized), 300
        img = Image.new('L', (w, h), )
        for h in range(h):
            for p in range(w):
                # lum = 255 - frame_points[p]
                lum = digitalized[p]
                img.putpixel((p, h), lum)

        img = img.resize((w, 4 * h))

        axs[2].set_title(f'pixels')
        axs[2].imshow(img, cmap='gist_gray')

        fig.tight_layout()

        plt.savefig(self.processed_chart_filepath)
        debug_log("processed chart saved", Colors.debug)
        if show:
            plt.show()
        plt.clf()

    def spectrogram_chart(self, show: bool = False):
        frequencies, times, spectrogram = signal.spectrogram(self.samples, self.sample_rate)
        plt.pcolormesh(times, frequencies, spectrogram, cmap='gist_earth')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        max_freq = frequencies[-1]
        arrange_y = [0, max_freq]
        arrange_labels_y = [f"{int(frequencies[0])}Hz", f"{max_freq}Hz"]
        plt.yticks(arrange_y, arrange_labels_y)

        arrange_x = [0, len(self.samples) / self.sample_rate - 0.03]
        arrange_labels_x = [f"{self.number * self.duration}s", f"{self.number * self.duration + self.duration}s"]
        plt.xticks(arrange_x, arrange_labels_x)

        plt.title(f'audio packet {self.number}')

        plt.savefig(self.spectrogram_chart_filepath)
        debug_log("spectrogram chart saved", Colors.debug)
        if show:
            plt.show()
        plt.clf()

    def spectrogram_image(self, save: bool = True):
        frequencies, times, spectrogram = signal.spectrogram(self.samples, self.sample_rate, mode="magnitude")

        spectrogram_normalized = spectrogram / ((np.max(spectrogram)) + 0.0001)

        img = Image.fromarray((cm.gist_earth(spectrogram_normalized) * 255).astype(np.uint8))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.ROTATE_90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if save:
            img.save(self.spectrogram_image_filepath)
            debug_log("spectrogram image saved", Colors.debug)

        return img

    def start_tone_chart(self, show: bool = False):
        """
        !!! DEBUG FUNCTION !!!

        this function will plot a chart that contains fast fourier transform of audio with start tone peaks found in it
        :param show: chart will pop up on the screen
        :return: nothing
        """
        frequency, amplitude = self.__fourier_transform()

        distance = self.__START_TONE_PEAKS_MINIMUM_DISTANCE
        height = self.__START_TONE_PEAKS_MINIMUM_HEIGHT
        prominence = self.__START_TONE_PEAKS_MINIMUM_PROMINENCE

        peaks = scipy.signal.find_peaks(amplitude, distance=distance, height=height, prominence=prominence)

        found = "found" if self.contain_start_tone() else "not found"
        plt.title(f"start tone {found}")
        plt.ylabel("amplitude")
        plt.xlabel("frequency [Hz]")
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d Hz'))

        plt.xlim(xmin=0)
        if frequency[-1] > 4000:
            plt.xlim(xmax=4000)
        plt.plot(frequency, amplitude)
        plt.plot(frequency[peaks[0]], peaks[1]['peak_heights'], ".", color='r', markersize=10)
        plt.savefig(self.start_tone_chart_filepath)
        if show:
            plt.show()
        plt.clf()

    def sync_pulse_chart(self, show: bool = False):
        """
        !!! DEBUG FUNCTION !!!

        this function will plot a chart that show if sync pulse was found in audio packet
        :param show: chart will pop up on the screen
        :return: nothing
        """
        frequency, amplitude = self.__fourier_transform()

        sync_pulse_info = self.find_sync_pulse()

        found = "found" if sync_pulse_info['pulse_found'] is True else "not found"

        found_f_peak = "found" if sync_pulse_info['frequency_peak_found'] is True else "not found"
        found_s_peak = "found" if sync_pulse_info['samples_peak_found'] is True else "not found"

        fig, axs = plt.subplots(2, 1)

        fig.suptitle(f"sync pulse {found}", fontsize=16)
        axs[0].set_title(f'fourier transform peak {found_f_peak}', fontsize=10)
        axs[0].plot(frequency, amplitude)
        axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d Hz'))
        axs[0].set_xlabel('frequency [HZ]')
        axs[0].set_ylabel('amplitude')
        axs[0].grid(True)
        axs[0].set_xlim(left=0)
        axs[0].set_ylim(bottom=0)
        axs[0].set_xlim(right=len(frequency))
        axs[0].plot(sync_pulse_info['peaks_fft'][0], sync_pulse_info['peaks_fft'][1], ".", color='r', markersize=10)

        if frequency[-1] > 3000:
            axs[0].set_xlim(left= 1000, right=3000)

        axs[1].set_title(f'samples peak {found_s_peak}', fontsize=10)

        ticks_range = np.arange(0, len(self.samples) + 1, len(self.samples) / 10)
        labels_range = [f"{round(float(x/self.sample_rate),2)}s" for x in ticks_range]

        axs[1].set_xticks(ticks_range)
        axs[1].set_xticklabels(labels_range)

        axs[1].plot(self.samples)

        axs[1].set_xlabel('time')
        axs[1].set_ylabel('value')
        axs[1].set_xlim(left=0, right=len(self.samples))
        axs[1].set_ylim(bottom=0)
        for peak in sync_pulse_info['peaks_samples']:
            axs[1].axvline(peak, color='r')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.1, hspace=1)
        plt.savefig(self.sync_pulse_chart_filepath)
        if show:
            plt.show()
        plt.clf()

    def find_sync_pulse(self) -> dict:
        frequency, amplitude = self.__fourier_transform()

        height = self.__SYNC_PULSE_PEAKS_MINIMUM_HEIGHT
        prominence = self.__SYNC_PULSE_PEAKS_MINIMUM_PROMINENCE

        peaks = scipy.signal.find_peaks(amplitude, height=height, prominence=prominence)
        peaks_freqs = peaks[0]
        in_frequency_range = lambda \
            peak: self.__SYNC_PULSE_PEAKS_MINIMUM_FREQUENCY <= peak <= self.__SYNC_PULSE_PEAKS_MAXIMUM_FREQUENCY
        peaks_correct_frequency_displace = all([in_frequency_range(peak) for peak in frequency[peaks_freqs]])

        def pattern_search():

            samples = lambda x: int((x / (len(self.samples) / self.sample_rate)) * len(self.samples))
            sync = [255] + [0] * samples(0.025) + [255]
            mindistance = samples(0.4)
            peaks = [(-mindistance, 0)]

            # minimum distance between peaks


            # need to shift the values down to get meaningful correlation values
            signalshifted = [x - 128 for x in self.samples]
            sync = [x - 128 for x in sync]
            for i in range(len(self.samples) - len(sync)):

                corr = np.dot(sync, signalshifted[i: i + len(sync)])

                if i - peaks[-1][0] > mindistance:
                    peaks.append((i, corr))
                elif corr > peaks[-1][1]:
                    peaks[-1] = (i, corr)
            return [peak[0] for peak in peaks][1:]

        pulses = pattern_search()
        out_info = {}
        out_info["frequency_peak_found"] = True if peaks_correct_frequency_displace and len(peaks_freqs) == 1 else False
        out_info["samples_peak_found"] = True if len(pulses) else False
        out_info["pulse_found"] = True if out_info["frequency_peak_found"] is True and out_info["samples_peak_found"] is True else False
        out_info['peaks_fft'] = [frequency[peaks[0]], peaks[1]['peak_heights']]
        out_info['peaks_samples'] = pulses
        return out_info

    def contain_start_tone(self) -> bool:
        """
        this function, using the fourier transform, checks if the audio packet has a start tone ↵
        ↳ containing 5 peaks separated by a distance of 300Hz between each other
        :return: True or False depending on whether the audio packet has a start tone or not
        """

        frequency, amplitude = self.__fourier_transform()

        distance = self.__START_TONE_PEAKS_MINIMUM_DISTANCE
        height = self.__START_TONE_PEAKS_MINIMUM_HEIGHT
        prominence = self.__START_TONE_PEAKS_MINIMUM_PROMINENCE

        peaks = scipy.signal.find_peaks(amplitude, distance=distance, height=height, prominence=prominence)

        peaks_freqs = peaks[0]
        in_frequency_range = lambda peak: self.__START_TONE_PEAKS_MINIMUM_FREQUENCY <= peak <= self.__START_TONE_PEAKS_MAXIMUM_FREQUENCY
        peaks_correct_frequency_displace = all([in_frequency_range(peak) for peak in frequency[peaks_freqs]])
        peaks_correct_amount = self.__START_TONE_PEAKS_MINIMUM_AMOUNT <= len(peaks_freqs) <= self.__START_TONE_PEAKS_MAXIMUM_AMOUNT

        return True if peaks_correct_frequency_displace and peaks_correct_amount else False

    def __fourier_transform(self):
        """
        this function creates a fourier transform based on audio samples ↵
        ↳ and looks for frequency peaks based on input parameters

        :return: x_axis of frequencies, y_axis of amplitude, found_peaks
        """

        fft = np.fft.fft(self.raw_samples)
        fft_len = len(fft)
        fft_arrange = np.arange(fft_len)
        fft_time_len = fft_len / self.sample_rate
        freq = fft_arrange / fft_time_len

        n_oneside = fft_len // 2
        freqs_one_side = freq[:n_oneside]
        amplitude_one_size = abs(fft[:n_oneside] / n_oneside)
        normalized_amplitude = amplitude_one_size / (max(amplitude_one_size) + 0.0001)

        return freqs_one_side, normalized_amplitude

    def __process_samples(self) -> np.ndarray:
        """
        this function applies filters, demodulates and returns a digitized signal with a value from 0 to 255

        :return: ndarray of digitalized samples
        """
        notched_signal = self.__notch_filter(self.raw_samples)
        filtered_signal = self.__demodulate(notched_signal)
        digitalized_signal = self.__digitalize(filtered_signal)
        return digitalized_signal

    def __notch_filter(self, samples: np.ndarray) -> np.ndarray:
        """
        this function takes audio samples and applies a notch filter to them
        to read more about notch filter please refer to: https://en.wikipedia.org/wiki/Band-stop_filter

        :param samples: ndarray of audio samples
        :return: ndarray of samples witch notch filter applied
        """

        b_notch, a_notch = signal.iirnotch(self.__NOTCH_FILTER_FREQUENCY,
                                           self.__NOTCH_FILTER_QUALITY_FACTOR,
                                           self.sample_rate)

        return signal.filtfilt(b_notch, a_notch, samples)

    @staticmethod
    def __demodulate(samples: np.ndarray) -> np.ndarray:
        """
        this function takes audio samples and uses the hilbert transform to ↵
        ↳ convert frequency modulation to amplitude modulation
        to read more about hilbert transform please refer to: https://en.wikipedia.org/wiki/Hilbert_transform

        :param samples: ndarray of audio samples
        :return: ndarray of am modulation samples
        """

        hilbert_signal = np.abs(scipy.signal.hilbert(samples))
        filtered_samples = scipy.signal.medfilt(hilbert_signal, 3)
        return filtered_samples

    @staticmethod
    def __digitalize(am_samples: np.ndarray) -> np.ndarray:
        """
        this function converts amplitude modulated samples and digitizes them from 0 to 255

        :param am_samples: ndarray of am modulation audio samples
        :return: ndarray of digitalized samples
        """
        plow = 0.5
        phigh = 99.5
        (low, high) = np.percentile(am_samples, (plow, phigh))
        delta = high - low
        digitalized = np.rint(255 * (am_samples - low) / (delta + 0.000001))
        digitalized[digitalized < 0] = 0
        digitalized[digitalized > 255] = 255
        return digitalized.astype(int)

    def __repr__(self):
        info = f'data packet {self.number} info: {self.number * self.duration}s-{self.number * self.duration + self.duration}s packet len: {len(self.samples)}  sample rate:{self.sample_rate}'
        return debug_log(info, Colors.debug)


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
        self.spectrum_websocket_stack = []
        self.frames_websocket_stack = []
        self.connected = False
        self.isRecording = False
        self.amount_peaks_found = 0
        self.saved_chunks = 0
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

            """w, h = self.SAMPLES_FOR_ONE_FRAME, self.MINIMUM_FRAMES_PER_UPDATE
            img = Image.new('L', (w, h), )
            for y in range(h):
                for x in range(w):
                    lum = 255 - frame_points[x*(y+1)]
                    img.putpixel((x, y), lum)

            img = img.resize((int(w/10), int(h)))"""

            min_frames = self.saved_frames * self.MINIMUM_FRAMES_PER_UPDATE
            max_frames = (self.saved_frames + 1) * self.MINIMUM_FRAMES_PER_UPDATE
            frame_output_path = f'{self.OUTPUT_DIRECTORY}frame_{min_frames}-{max_frames}.png'

            if self.__save_frames:
                img.save(frame_output_path)

            self.saved_frames += 1

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

    def __process_audio_samples(self, __samples):

        __packet = DataPacket(self.RATE,
                              __samples,
                              self.LINES_PER_MINUTE,
                              self.OUTPUT_DIRECTORY,
                              self.AUDIO_PACKET_DURATION,
                              self.saved_chunks)

        self.data_packets.append(__packet)

        img = __packet.spectrogram_image(save=self.__save_spectrum)

        if not self.start_tone_found:
            if __packet.contain_start_tone():
                self.amount_peaks_found += 1
            else:
                self.amount_peaks_found = 0
            if self.amount_peaks_found * self.AUDIO_PACKET_DURATION >= 4:
                self.start_tone_found = True
                debug_log("start tone found", Colors.info)

        if self.start_tone_found and not self.phasing_signal_found:
            pulse_info = __packet.find_sync_pulse()
            if pulse_info["pulse_found"] is True:
                self.phasing_signal_found = True
                self.data_points = __packet.samples[pulse_info["peaks_samples"][-1]:]
                debug_log("sync pulse found", Colors.info)

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

    def end_stream(self):
        global stop_threads
        stop_threads = True
        for thread in self.threads:
            thread.join()
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
