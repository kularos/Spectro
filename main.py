import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class StableSpectrogram:
    def __init__(self, rate=44100, chunk=4096, window_size=50):
        self.rate = rate
        self.chunk = chunk
        self.window_size = window_size

        # --- Fixed Intensity Scale (dB) ---
        # Adjust these if the display is too dark or too bright
        self.vmin = 20  # Minimum dB to show (Noise floor)
        self.vmax = 100  # Maximum dB to show (Peak volume)

        # Audio setup
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # Buffers
        self.freq_bins = self.chunk // 2
        self.spec_data = np.zeros((self.freq_bins, self.window_size))

        # Plotting Setup
        self.fig, self.ax = plt.subplots(figsize=(12, 7))

        # extent=[left, right, bottom, top]
        self.img = self.ax.imshow(
            self.spec_data,
            aspect='auto',
            origin='lower',
            extent=[0, self.window_size, 0, self.rate / 2],
            cmap='magma',
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation='bilinear'
        )

        # Static Axis Formatting
        self.ax.set_title("Stable Live Spectrogram (Fixed dB Scale)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_xlabel("Time (Chunks)")

        # Set specific ticks for a clean linear scale
        max_f = self.rate // 2
        self.ax.set_yticks(np.arange(0, max_f, 1000 if max_f < 10000 else 2000))

        # Add a colorbar that stays static
        self.cbar = self.fig.colorbar(self.img)
        self.cbar.set_label("Intensity (dB)")

        self.window = np.hanning(self.chunk)

    def update(self, frame):
        try:
            raw_data = self.stream.read(self.chunk, exception_on_overflow=False)
            data_int = np.frombuffer(raw_data, dtype=np.int16)

            # FFT processing
            fft_complex = np.fft.fft(data_int * self.window)
            fft_mag = np.abs(fft_complex[:self.freq_bins])

            # Convert to dB: 20 * log10(amplitude)
            # We add a small epsilon to avoid log(0)
            fft_db = 20 * np.log10(fft_mag + 1e-6)

            # Roll and update
            self.spec_data = np.roll(self.spec_data, -1, axis=1)
            self.spec_data[:, -1] = fft_db

            self.img.set_array(self.spec_data)

        except Exception as e:
            print(f"Update error: {e}")

        return self.img,

    def start(self):
        # Using a slightly higher interval to reduce CPU load
        self.ani = FuncAnimation(self.fig, self.update, interval=20, blit=True)
        plt.show()

    def __del__(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


if __name__ == "__main__":
    # Lower chunk = faster response, Higher chunk = thinner frequency lines
    spec = StableSpectrogram(rate=44100, chunk=8000)
    spec.start()