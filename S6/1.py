

import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio
import numpy as np
samplerate,Y = wavfile.read('test2.wav')

plt.plot(Y)

Audio(Y, rate=samplerate)

freq = np.fft.fftfreq(len(Y))
# plt.plot(freq)
fft = np.fft.fft(Y)

halffreq = freq[:int(len(Y)/2)]

for i in range(len(halffreq)):
    if halffreq[i] < 0.1: 
        fft[i] = 0.0
        fft[int(len(Y)/2) + i] = 0.0
inverse = np.fft.ifft(fft)
plt.plot((inverse),'b'),plt.plot(Y,'r')

Audio(inverse, rate=samplerate)
