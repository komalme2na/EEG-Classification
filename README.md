
# EEG Signal Classification 🧠

This mini-project demonstrates a full EEG signal classification pipeline using Python, MNE, and scikit-learn. this project covers EEG data loading, preprocessing, feature extraction, and binary classification. hope you like it.



##  📊Project Structure

EEG-Classification/
├── data/
│   └── S001R03.edf             # EEG file (BCI dataset)
├── main.py                     # Load & visualize raw EEG data
├── extract_features.py         # Band power feature extraction
├── train_model.py              # Random forest classification
├── features.csv                # Extracted features
└── README.md                   # This file

📌 Objectives
- Load and filter raw EEG signals using MNE

- Extract meaningful features (band power) from EEG channels

- Train a machine learning model to classify brain states

Lay the groundwork for real-world brain-computer interface (BCI) projects

📂 Data Used
Dataset: PhysioNet BCI2000

Sample file: S001R03.edf (Motor Imagery EEG)


🧠 Feature Extraction
We compute band power features in these EEG frequency bands:

Band	Frequency (Hz)
Delta	1–4
Theta	4–8
Alpha	8–13
Beta	13–30

📈 Classification
Model: RandomForestClassifier

Task: Binary classification (simulated labels for demo)

Metric: Accuracy, Precision, Recall, F1

🌟 Future Work
Use real task-labeled EEG datasets (e.g., motor imagery)

Compare multiple classifiers (SVM, LSTM, CNNs)

Real-time BCI implementation with live EEG streams

Add visualizations and feature importance plots


📄 License
This project is open-source 









