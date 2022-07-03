from scipy.io import wavfile
import scipy.signal
from scipy import signal
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from progress_bar import plot_bar
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import sys
import os

class Demodulator:
    def __init__(self, filepath: str, lines_per_minute: int = 120, quiet: bool = False):
        self.filepath = filepath
        self.filename = self.filepath.split('/')[-1].split('.')[0]
        self.lines_per_minute = lines_per_minute
        self.time_for_one_frame = 1 / (self.lines_per_minute / 60)  # in s
        self.quiet = quiet

        if self.quiet:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = sys.__stdout__


        print("#" * 10 + ' ' * 5 + str(self.filename).ljust(20) + ' ' * 5 + "#" * 10)

        self.audio_data, self.sample_rate, self.length = self.__read_file()

        if self.sample_rate != 11025:
            self.audio_data, self.sample_rate, self.length = self.__resample(self.audio_data, self.sample_rate,
                                                                             self.length)

        self.demodulated_data = self.__demodulate(self.audio_data)
        self.digitalized_data = self.__digitalize(self.demodulated_data)
        self.phasing_signals = self.__find_sync_pulse(self.digitalized_data, self.sample_rate, self.time_for_one_frame)
        self.start_frame = self.phasing_signals[-1]

        self.output_image = self.__convert_to_image(self.digitalized_data[self.start_frame:],
                                                    self.time_for_one_frame,
                                                    self.sample_rate)
        print("#" * 50)
        
        if self.quiet:
            sys.stdout = sys.__stdout__



    def signal_chart(self, start, end):
        time_start = start
        time_end = end
        length = time_end - time_start
        tick_freq = 0.5
        arrange = np.arange(0, length * self.sample_rate + 1, tick_freq * self.sample_rate)
        arrange_labels = [round(float(x), 1) for x in np.arange(time_start, time_end + 0.01, tick_freq)]

        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f s'))
        data_crop = self.digitalized_data[time_start * self.sample_rate:int(time_end * self.sample_rate)]
        data_am_crop = self.__demodulate(data_crop, quiet=True)
        plt.ylim(ymin=0, ymax=255)
        plt.xlim(xmin=0, xmax=length * self.sample_rate)
        plt.plot(data_am_crop)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title("Signal")
        plt.xticks(arrange, arrange_labels)

        for sig in self.phasing_signals:
            plt.axvline(x=sig, color='red', linestyle='--')

        plt.savefig(self.filename, dpi=600)
        plt.show()

    @staticmethod
    def __demodulate(data: list, quiet: bool = False):
        if not quiet:
            print("DEMODULATING SIGNAL:")
            plot_bar(0, 1, 50, False)
        hilbert_signal = scipy.signal.hilbert(data)
        filtered_signal = scipy.signal.medfilt(np.abs(hilbert_signal), 5)
        if not quiet:
            plot_bar(1, 1, 50, False)
            print()
        return filtered_signal

    @staticmethod
    def __digitalize(data):
        print("DIGITALIZING SIGNAL:")
        plot_bar(0, 1, 50, False)
        plow = 0.5
        phigh = 99.5
        (low, high) = np.percentile(data, (plow, phigh))
        delta = high - low
        digitalized = np.round(255 * (data - low) / delta)
        digitalized[digitalized < 0] = 0
        digitalized[digitalized > 255] = 255
        plot_bar(1, 1, 50, False)
        print()
        return [int(point) for point in digitalized]

    @staticmethod
    def __find_sync_pulse(data, sample_rate, frame_len):
        print("FINDING SYNC PULSE:")

        def pattern_search():

            samples = lambda x: int(x * frame_len * sample_rate)

            sync = [1] * samples(0.005) + [0] * samples(0.001) + [1] * samples(0.005)
            peaks = [(0, 0)]

            # minimum distance between peaks
            mindistance = int(frame_len * sample_rate * 0.8)

            # need to shift the values down to get meaningful correlation values
            signalshifted = [x - 128 for x in data]
            sync = [x - 128 for x in sync]
            for i in range(len(data) - len(sync)):

                corr = np.dot(sync, signalshifted[i: i + len(sync)])

                if i - peaks[-1][0] > mindistance:
                    # if previous peak is too far, keep it and add this value to the list as a new peak
                    peaks.append((i, corr))
                    plot_bar(i, len(data), 50, True, "samples")
                elif corr > peaks[-1][1]:
                    # else if this value is bigger than the previous maximum, set this one
                    peaks[-1] = (i, corr)

                if len(peaks) == 100:
                    plot_bar(len(data), len(data), 50, True, "samples")
                    print()
                    break

            return [peak[0] for peak in peaks]

        def deviation_search(x):
            allowed_deviation = 500  # samples
            max_deviation = frame_len * sample_rate + allowed_deviation
            min_deviation = frame_len * sample_rate - allowed_deviation
            return True if max_deviation > x > min_deviation else False

        def find_sync_pulses():
            clear_peaks = []

            for i in range(1, len(peaks) - 1):
                distance_between_peaks = peaks[i] - peaks[i - 1]
                if deviation_search(distance_between_peaks):
                    clear_peaks.append(peaks[i])

            return clear_peaks

        def find_peak_groups():
            groups = []
            group = []
            for i in range(1, len(distanced_peaks) - 1):
                distance_between_peaks = peaks[i] - peaks[i - 1]
                if deviation_search(distance_between_peaks):
                    group.append(peaks[i])
                else:
                    groups.append(group)
                    group = []
            return groups

        peaks = pattern_search()
        distanced_peaks = find_sync_pulses()
        peak_groups = find_peak_groups()
        return max(peak_groups, key=len)

    @staticmethod
    def __convert_to_image(data, time_for_one_frame, sample_rate):
        print("CONVERTING SIGNAL TO IMAGE:")
        frame_width = int(time_for_one_frame * sample_rate)
        w, h = frame_width, len(data) // frame_width
        image = Image.new('L', (w, h), )
        px, py = 0, 0
        for p in range(len(data)):
            lum = 255 - data[p]
            image.putpixel((px, py), lum)
            px += 1
            if px >= w:
                if (py % 50) == 0:
                    plot_bar(py + 1, h, 50, True, "lines")
                px = 0
                py += 1
                if py >= h:
                    plot_bar(h, h, 50, True, "lines")
                    break

        image = image.resize((w, 4 * h))
        print()
        return image

    @staticmethod
    def __create_image_from_matrix(matrix):
        w = max([len(line) for line in matrix])
        h = len(matrix)
        print(w, h)
        image = Image.new('L', (w, h), )

        for y in range(h):
            for x in range(w):
                image.putpixel((x, y), matrix[y][x])

        return image

    def __read_file(self):
        sample_rate, data = wavfile.read(self.filepath)

        if len(data.shape) == 2:
            self.__warning("WARNING: two channels audio detected. Program will try to merge audio to one channel")
            print("MERGING AUDIO CHANNELS:")
            data = self.__merge_channels(data)
            print()

        length = len(data) / sample_rate
        return data, sample_rate, length

    @staticmethod
    def __merge_channels(audio_channels):
        parts = len(audio_channels)
        one_channel_audio = []
        for p in range(parts):
            if p % 1000 == 0 or p == parts - 1:
                plot_bar(p + 1, parts, 50, True, "samples")
            one_channel_audio.append(np.divide(np.add(audio_channels[p][0], audio_channels[p][1]), 2))
        return one_channel_audio

    def __resample(self, audio_data, sample_rate, length):
        self.__warning("WARNING: audio sample rate is not 11025 samples per second. Program will try to resample audio")
        print(f"RESAMPLING AUDIO FROM {round(sample_rate / 1000, 2)} KhZ TO 11.025 KHZ:")
        plot_bar(0, 1, 50, False)
        data = scipy.signal.resample(audio_data, int(11025 * length))
        plot_bar(1, 1, 50, False)
        print()
        sample_rate = 11025
        length = len(data) / sample_rate
        return data, sample_rate, length

    @staticmethod
    def __warning(text):
        print(f"\033[0;33m{text}\033[0m")

    def show_output_image(self):
        plt.imshow(self.output_image, cmap='gray')
        plt.show()

    def save_output_image(self, filepath: str):
        self.output_image.save(filepath)


if __name__ == "__main__":

    demodulator = Demodulator('input/input_lq.wav', lines_per_minute=120)
    # demodulator.show_output_image()
    demodulator.save_output_image("input.png")
    demodulator.signal_chart(0, 25.5)
