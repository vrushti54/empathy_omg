## 🔹 Pipeline Flow

```mermaid
flowchart LR
    A[Audio .wav] --> B[Mel-Spectrogram (.npy)]
    B --> C[Data Splits: Train / Val / Test]
    C --> D[MLP]
    C --> E[LSTM]
    C --> F[GRU]
    C --> G[1D-CNN]
    D & E & F & G --> H[Metrics: CCC / PCC / MAE / MSE]

