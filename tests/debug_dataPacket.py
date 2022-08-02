import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wefax_live import DataPacket
from scipy.io import wavfile


def debug_data_packet(sample_rate, samples, directory, duration, number):
    packet = DataPacket(sample_rate, samples, directory, duration, number)
    packet.processed_chart(show=True)
    #packet.demodulated_chart(show=True)
    #packet.fft_chart(show=True)
    #packet.audio_chart(show=True)
    #packet.spectrogram_chart(show=True)




if __name__ == "__main__":
    _sample_rate, _samples = wavfile.read('test_files/30.wav')
    debug_data_packet(_sample_rate, _samples, 'static/temp/', int(_samples[0]/_sample_rate), 0)
