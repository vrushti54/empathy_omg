# Re-extract features (safe to re-run)
python scripts\preprocess.py --wav_dir data/raw/audio --out_dir data/features/audio_melspec

# Rebuild labels & splits (uses data/labels/labels.csv if present;
# the make_labels script only creates a dummy one when you don't have real labels)
if (-Not (Test-Path data\labels\labels.csv)) {
  python scripts\make_labels_from_filenames.py
}

python scripts\make_splits.py

# Train MLP and LSTM
python train.py
python train_lstm.py

# Show results
Write-Host "`n== MLP results ==" -ForegroundColor Cyan
Get-Content results\mlp_audio_melspec.json
Write-Host "`n== LSTM results ==" -ForegroundColor Cyan
Get-Content results\lstm_audio_melspec.json
