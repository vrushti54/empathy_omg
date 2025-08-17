# Empathy OMG – Audio Baseline Project

This project implements baseline models (MLP, LSTM, GRU, CNN1D) for audio-based empathy detection.  
The pipeline follows an end-to-end flow:

- Input: `.wav` audio files  
- Conversion: Audio → Mel-Spectrogram (`.npy`)  
- Dataset Splits: Train / Validation / Test  
- Model Training: MLP, LSTM, GRU, CNN1D  
- Evaluation Metrics: CCC, PCC, MAE, MSE  

**Note**: Dummy data produces meaningless scores. Use official OMG-Empathy splits for real evaluation.

---

## 🔹 Setup
```ps1
cd C:\empathy_omg
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
.\run_all.ps1
