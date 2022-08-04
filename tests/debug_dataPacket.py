import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wefax_live import DataPacket
from scipy.io import wavfile
import math

def debug_data_packet(sample_rate, samples, lines_per_minute, directory, duration, number):
    packet = DataPacket(sample_rate, samples, lines_per_minute, directory, duration, number)
    #packet.processed_chart(show=True)
    #packet.demodulated_chart(show=True)
    #packet.fft_chart(show=True)
    #packet.audio_chart(show=True)
    #packet.spectrogram_chart(show=True)
    #packet.start_tone_chart(show=True)
    packet.sync_pulse_chart(show=True)




if __name__ == "__main__":
    _sample_rate, _samples = wavfile.read(sys.argv[1])
    duration = len(_samples)/_sample_rate
    print(f"sample rate: {_sample_rate}  samples: {len(_samples)}  duration: {duration}")
    debug_data_packet(_sample_rate, _samples, 120, 'static/temp/', duration, 0)
