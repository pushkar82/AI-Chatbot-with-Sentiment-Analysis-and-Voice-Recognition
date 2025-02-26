import sounddevice as sd
from scipy.io.wavfile import write
import wave

# Parameters
sampling_rate = 16000  
duration = 5  # seconds
filename = "temp.wav"

# Record audio
print("Recording...")
audio = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype='int16')
sd.wait()  # Wait until recording is complete
print("Recording complete. Saving audio...")


write(filename, sampling_rate, audio)
print(f"Audio saved as {filename}.")
