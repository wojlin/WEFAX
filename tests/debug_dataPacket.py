import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_packet import DataPacket
from scipy.io import wavfile


def debug_data_packet(sample_rate, samples, lines_per_minute, directory, duration, number, debug_type, show):
    packet = DataPacket(sample_rate, samples, lines_per_minute, directory, duration, number)
    if debug_type == "start":
        packet.start_tone_chart(show=show)
    elif debug_type == "stop":
        packet.stop_tone_chart(show=show)
    elif debug_type == "sync":
        packet.sync_pulse_chart(show=show)
    elif debug_type == "processed":
        packet.processed_chart(show=show)
    elif debug_type == "fft":
        packet.fft_chart(show=show)
    elif debug_type == "audio":
        packet.audio_chart(show=show)
    elif debug_type == "spectrogram":
        packet.spectrogram_chart(show=show)


if __name__ == "__main__":
    filename = sys.argv[1]
    debug_type = sys.argv[2]
    show = sys.argv[3]

    _sample_rate, _samples = wavfile.read(filename)
    duration = len(_samples) / _sample_rate
    print(f"sample rate: {_sample_rate}  samples: {len(_samples)}  duration: {duration}")
    debug_data_packet(_sample_rate, _samples, 120, 'static/temp/', duration, 0, debug_type, show)
