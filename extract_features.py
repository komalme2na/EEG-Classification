import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

# Load raw EEG data
raw = mne.io.read_raw_edf("data/S001R03.edf", preload=True)
raw.filter(1., 40.)  # Bandpass filter
sfreq = int(raw.info['sfreq'])  # Sampling frequency

# Parameters
win_size = 2  # seconds
step_size = 2  # seconds (non-overlapping windows)
window_samples = win_size * sfreq

# Frequency bands (Hz)
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}

def extract_band_power(epoch_data, sfreq):
    freqs, psd = welch(epoch_data, sfreq, nperseg=len(epoch_data))
    features = {}
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        power = np.mean(psd[:, mask], axis=1)
        features.update({f"{band}_ch{ch}": p for ch, p in enumerate(power)})
    return features

# Extract features
X = []
start = 0
data = raw.get_data(picks='eeg')
n_samples = data.shape[1]

while start + window_samples <= n_samples:
    window = data[:, start:start + window_samples]
    feats = extract_band_power(window, sfreq)
    X.append(feats)
    start += step_size * sfreq

df = pd.DataFrame(X)
df.to_csv("features.csv", index=False)
print(" Features extracted and saved to features.csv")

