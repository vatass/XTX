# LSTM Limit Order Book Forecasting – Inference Instructions

This repository contains the inference script used to evaluate the LSTM-based
time-series forecasting model for the XTX Limit Order Book task.

## Overview

During training, the model was evaluated on **fold 7** of a 7-fold
TimeSeriesSplit. To guarantee identical behavior during external evaluation,
the training script stored:

- `fold7_indices.npy` — the exact row indices of the fold-7 validation set  
- `x_scaler_fold7.pkl` — the fitted StandardScaler for features  
- `best_lstm_fold7.pt` — the trained LSTM model weights  

The provided inference script reconstructs fold-7 **exactly**, applies the same
feature engineering and scaling, generates sequences of length 40, and
produces predictions that match the original validation performance:

```
R² ≈ 0.052146  
MAE ≈ 1.384  
MSE ≈ 3.785  
```

---

## Files Required

| File | Purpose |
|------|---------|
| `raw.csv` | Full raw LOB dataset (provided externally) |
| `inference_final.py` | Final inference script |
| `fold7_indices.npy` | Saved validation indices from training |
| `x_scaler_fold7.pkl` | Trained StandardScaler for features |
| `best_lstm_fold7.pt` | Trained LSTM model |

---

## Running Inference

Run:

```bash
python inference_final.py \
    --raw_data raw.csv \
    --fold_idx_path fold7_indices.npy \
    --model_path best_lstm_fold7.pt \
    --x_scaler x_scaler_fold7.pkl \
    --output_path predictions.csv
```

This will produce:

```
predictions.csv
```

with columns:

- `prediction_scaled` — predicted y (scaled units)  
- `row_index` — original row index in fold-7 after sequence offset  

---

## Output Format

The output file has one prediction per valid timestep:

```
prediction_scaled, row_index
0.01234, 40
0.00988, 41
...
```

Predictions are in **scaled space** because the evaluation benchmark uses
**scaled R²**.

---

## Notes

- No ground truth labels (`y`) are required or used.  
- The script never drops or reorders rows except as dictated by the
  stored `fold7_indices.npy`, ensuring exact reproducibility.  
- Sequence length is fixed at 40 (same as training).  
- Feature engineering exactly matches the training pipeline.  

---

## Contact

If any clarification is needed regarding preprocessing or evaluation, please
reach out.
