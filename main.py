from scipy.io import wavfile
import scipy.signal
from scipy import signal
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from progress_bar import plot_bar
from PIL import Image
from matplotlib.ticker import FormatStrFormatter

class Demodulator:
    def __init__(self, filepath: str, lines_per_minute: int = 120):
        self.filepath = filepath
        self.lines_per_minute = lines_per_minute
        self.audio_data, self.sample_rate, self.length = self.__read_file()

        if self.sample_rate != 11025:
            self.audio_data, self.sample_rate, self.length = self.__resample(self.audio_data, self.sample_rate,
                                                                             self.length)



        self.output_image = self.__demodulate()

    def __demodulate(self):
        def hilbert(data):
            analytical_signal = signal.hilbert(data)
            amplitude_envelope = np.abs(analytical_signal)
            return amplitude_envelope

        data_am = hilbert(self.audio_data)
        frame_width = int((1 / (self.lines_per_minute / 60)) * self.sample_rate)
        w, h = frame_width, len(data_am) // frame_width
        image = Image.new('L', (w, h), )
        px, py = 0, 0
        max_val = max(data_am)
        min_val = min(data_am)

        remap = lambda x: int(((x - min_val) * 255) / (max_val - min_val))

        print("REMAPPING AM SIGNAL:")
        plot_bar(0, 1, 50, False)
        normalized = [remap(data_am[i]) for i in range(len(data_am))]
        plot_bar(1, 1, 50, False)
        print()

        print("DEMODULATING AUDIO:")

        for p in range(len(data_am)):
            lum = 255 - normalized[p]
            image.putpixel((px, py), lum)
            px += 1
            if px >= w:
                if (py % 50) == 0:
                    plot_bar(py + 1, h, 50, True, "lines")
                px = 0
                py += 1
                if py >= h:
                    plot_bar(py + 1, h, 50, True, "lines")
                    break

        image = image.resize((w, 4 * h))

        """time_start = 5
        time_end = 20
        length = time_end - time_start
        tick_freq = 0.5
        arrange = np.arange(0, length*self.sample_rate+1, tick_freq*self.sample_rate)
        arrange_labels = [round(float(x), 1) for x in np.arange(time_start, time_end+0.01, tick_freq)]

        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f s'))
        data_crop = normalized[time_start * self.sample_rate:int(time_end * self.sample_rate)]
        data_am_crop = hilbert(data_crop)
        plt.ylim(ymin=0, ymax=255)
        plt.xlim(xmin=0, xmax=length*self.sample_rate)
        plt.plot(data_am_crop)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title("Signal")
        plt.xticks(arrange, arrange_labels)
        plt.show()"""

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
                plot_bar(p + 1, parts, 50, True, "blocks")
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
    # demodulator = Demodulator('input/wefax_lq.wav')
    demodulator = Demodulator('input/input_lq.wav', lines_per_minute=120)
    demodulator.show_output_image()
    demodulator.save_output_image("test.png")
