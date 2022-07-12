import pyaudio
import wave
import time
import os


class LiveDecoder:
    def __init__(self, datestamp: str):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100

        os.mkdir(f"static/temp/{datestamp}")
        self.WAVE_OUTPUT_FILENAME = f"static/temp/{datestamp}/"

        self.saved_chunks = 0

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)



    def record(self, duration):
        print("recording_start")
        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            data = self.stream.read(self.CHUNK)
            frames.append(data)

        print("recording_end")

        wf = wave.open(self.WAVE_OUTPUT_FILENAME + str(self.saved_chunks) + '.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.saved_chunks += 1
        print("file saved")

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
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(exc_type, exc_val, exc_tb)
        self.end_stream()


if __name__ == "__main__":
    datestamp = str(round(time.time() * 1000))
    with LiveDecoder(datestamp) as live_decoder:
        for x in range(10):
            live_decoder.record(1)
        live_decoder.combine()
        live_decoder.end_stream()