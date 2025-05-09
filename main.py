import mne
import matplotlib.pyplot as plt

# Load the raw EEG data
edf_path = "data/S001R03.edf"  # path of our file
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Print basic info
print("Channels:", raw.ch_names)
print("Sampling rate:", raw.info['sfreq'])

# Applying a bandpass filter: keep 1–40 Hz
raw.filter(l_freq=1.0, h_freq=40.0)

# Picking few EEG channels to plot
picks = mne.pick_types(raw.info, eeg=True)
raw.plot(order=picks[:10], duration=5, n_channels=10, scalings='auto')
