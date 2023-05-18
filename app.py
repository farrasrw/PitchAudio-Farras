from flask import Flask, render_template, request, redirect
import pyaudio
import wave
import numpy as np

app = Flask(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

CHUNK = 204800
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("* recording")

    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return redirect('/')

def pitch_shift(snd_array, n, window_size=2**13, h=2**11):
    factor = 2**(1.0 * n / 12.0)
    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(snd_array) / factor + window_size))
    for i in np.arange(0, len(snd_array) - (window_size + h), h*factor):
        i = int(i)
        a1 = snd_array[i: i + window_size]
        a2 = snd_array[i + h: i + window_size + h]
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % (2 * np.pi)
        a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
        i2 = int(i / factor)
        result[i2: i2 + window_size] += hanning_window * a2_rephased.real
    return result.astype('int16')

@app.route('/pitchshift', methods=['POST'])
def pitchshift():
    n = int(request.form['pitch'])
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
    data = wf.readframes(wf.getnframes())
    wf.close()
    audio_array = np.frombuffer(data, dtype=np.int16)
    pitched = pitch_shift(audio_array, n)
    output_filename = f"output_{n}.wav"
    output_wavefile = wave.open(output_filename, 'wb')
    output_wavefile.setnchannels(CHANNELS)
    output_wavefile.setsampwidth(audio.get_sample_size(FORMAT))
    output_wavefile.setframerate(RATE)
    output_wavefile.writeframes(pitched.tobytes())
    output_wavefile.close()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
