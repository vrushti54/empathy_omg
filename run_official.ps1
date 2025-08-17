# Convert official CSVs -> JSON splits
python scripts\convert_omg_splits.py --dir_csv C:\path\to\omg_csvs
# Extract features, train all, evaluate all, refresh summary
python scripts\preprocess.py --wav_dir data/raw/audio --out_dir data/features/audio_melspec
python train.py
python train_lstm.py
python train_gru.py
python train_cnn1d.py
python eval_test.py --model mlp
python eval_test.py --model lstm
python eval_test.py --model gru
python eval_test.py --model cnn1d
python scripts\collect_results.py
