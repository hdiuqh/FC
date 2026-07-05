import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch


DT = 0.05
FS = 1 / DT

class TerrainFeatureExtractor:

    def __init__(self, az):

        self.az = az

    # -------------------------------------------------
    # Vertical Jerk
    # -------------------------------------------------

    def compute_jerk(self):

        jerk = np.gradient(self.az, DT)

        return jerk

    # -------------------------------------------------
    # RMS
    # -------------------------------------------------

    def compute_rms(self, signal):

        return np.sqrt(np.mean(signal**2))

    # -------------------------------------------------
    # PSD
    # -------------------------------------------------

    def compute_psd(self, signal):

        freq, psd = welch(
            signal,
            fs=FS,
            nperseg=256
        )

        return freq, psd

    # -------------------------------------------------
    # Band Energy
    # -------------------------------------------------

    def compute_band_energy(self, freq, psd):

        low = (freq >= 0.5) & (freq < 3)

        mid = (freq >= 3) & (freq < 10)

        high = freq >= 10

        low_energy = np.trapz(psd[low], freq[low])

        mid_energy = np.trapz(psd[mid], freq[mid])

        high_energy = np.trapz(psd[high], freq[high])

        return low_energy, mid_energy, high_energy

    # -------------------------------------------------
    # Dominant Frequency
    # -------------------------------------------------

    def dominant_frequency(self, freq, psd):

        idx = np.argmax(psd)

        return freq[idx]

    # -------------------------------------------------
    # Complete Feature Vector
    # -------------------------------------------------

    def extract(self):

        jerk = self.compute_jerk()

        rms = self.compute_rms(jerk)

        freq, psd = self.compute_psd(jerk)

        low, mid, high = self.compute_band_energy(freq, psd)

        dom = self.dominant_frequency(freq, psd)

        feature = {

            "RMS": rms,

            "PSD_low": low,

            "PSD_mid": mid,

            "PSD_high": high,

            "Dominant_frequency": dom

        }

        return feature, jerk, freq, psd

def plot_result(jerk, freq, psd, feature):

    plt.figure(figsize=(12, 8))

    # --------------------------
    # Jerk
    # --------------------------

    plt.subplot(211)

    plt.plot(jerk)

    plt.title("Vertical Jerk")

    plt.xlabel("Sample")

    plt.ylabel("Jerk (m/s³)")

    plt.grid(True)

    plt.text(
        0.02,
        0.9,
        f"RMS = {feature['RMS']:.3f}",
        transform=plt.gca().transAxes
    )

    # --------------------------
    # PSD
    # --------------------------

    plt.subplot(212)

    plt.semilogy(freq, psd)

    plt.xlabel("Frequency (Hz)")

    plt.ylabel("PSD")

    plt.title("Power Spectral Density")

    # ISO reference

    iso = 0.02 / (freq + 0.1)**2

    plt.semilogy(freq, iso, '--')

    # highlight frequency bands

    plt.axvspan(0.5, 3,
                alpha=0.2)

    plt.axvspan(3, 10,
                alpha=0.2)

    plt.grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("medium_terrain.csv")

    az = df["az"].values

    extractor = TerrainFeatureExtractor(az)

    feature, jerk, freq, psd = extractor.extract()

    print("======== Terrain Feature ========")

    for k, v in feature.items():

        print(k, ":", v)

    plot_result(jerk, freq, psd, feature)