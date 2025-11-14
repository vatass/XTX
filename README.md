# LSTM Limit Order Book Forecasting – Inference Instructions

This repository provides the production inference script for evaluating the
trained LSTM model on **raw Limit Order Book dataset**.

Requirements:

- `raw.csv` — raw limit order book data **without target y**
- `best_lstm.pt` — trained LSTM model weights
- `x_scaler.pkl` — StandardScaler used during training
- `inference_final.py` — the inference script

The script produces **scaled predictions for every row of the dataset** after
the sequence warm-up length (default 40 timesteps).

---

## Running Inference

```bash
python inference_final.py \
    --raw_data raw.csv \
    --model_path best_lstm.pt \
    --x_scaler x_scaler.pkl \
    --output_path predictions.csv
```

---

## Output Format

The script produces a file:

```
predictions.csv
```

with columns:

- `prediction_scaled` — the model’s output in **scaled space**
- `row_index` — the original row of the dataset corresponding to the prediction

Example:

```
prediction_scaled,row_index
0.01234,40
0.00988,41
0.00811,42
...
```


---

## Notes

- The script applies **identical feature engineering** as during training.
- Sequence length is fixed to 40 by default.
- Predictions are **in scaled space**, matching the model's training behavior.

---
