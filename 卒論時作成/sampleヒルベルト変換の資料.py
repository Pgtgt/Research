import numpy
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

def FilteredSignal(signal, fs, cutoff):
    B, A = butter(1, cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(B, A, signal, axis=0)
    return filtered_signal

fs = 100000.
T = .1
time = numpy.arange(0., T, 1 / fs)
frequency = 1000
noise = numpy.random.normal(0, 0.3, int(fs/10))
signal = 4+numpy.sin(2 * numpy.pi * frequency * time)*numpy.cos(2 * numpy.pi * frequency/500 * time)
signal_center = signal-4
analyticSignal = hilbert(signal_center)
amplitudeEvelope = numpy.abs(analyticSignal)
cutoff =500
filteredSignal = FilteredSignal(amplitudeEvelope, fs, cutoff)

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(time, signal_center)

ax2.plot(time, amplitudeEvelope)

ax2.plot(time, filteredSignal)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Amplitud')
ax2.grid(True)
plt.show()
