#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse


# ============================================================
#  LSTMRegressor — MUST match the TRAINING version EXACTLY
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # TRAINING USES LAYERNORM ONLY ON THE FINAL TIMESTEP
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)            # (B, L, H)
        out_last = out[:, -1, :]         # last timestep
        out_norm = self.layer_norm(out_last)
        out_final = self.fc(out_norm).squeeze(-1)
        return out_final


# ============================================================
#  Dataset for inference
# ============================================================
class LOBSeqDatasetTest(Dataset):
    def __init__(self, X, y=None, sequence_length=40):
        self.sequence_length = sequence_length
        self.has_y = y is not None

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length+1:i+1])
            if self.has_y:
                y_seq.append(y[i])

        self.X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)

        if self.has_y:
            self.y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.has_y:
            return self.X_seq[idx], self.y_seq[idx]
        return self.X_seq[idx]


# ============================================================
# Load scaler
# ============================================================
def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Main inference
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_x_path", required=True)
    parser.add_argument("--test_y_path", required=True)
    parser.add_argument("--x_scaler", required=True)
    parser.add_argument("--y_scaler", required=True)
    parser.add_argument("--output_path", default="predictions.csv")
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------
    X = pd.read_csv(args.test_x_path).values
    y = pd.read_csv(args.test_y_path).values.squeeze()

    # ------------------------------------------------------------
    # Load scalers
    # ------------------------------------------------------------
    x_scaler = load_scaler(args.x_scaler)
    y_scaler = load_scaler(args.y_scaler)

    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).squeeze()

    # ------------------------------------------------------------
    # Sequence dataset
    # ------------------------------------------------------------
    dataset = LOBSeqDatasetTest(
        X_scaled,
        y_scaled,
        sequence_length=args.sequence_length
    )

    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        input_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # Handle DataParallel checkpoints
    state_dict = torch.load(args.model_path, map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    preds_scaled = []
    true_scaled = []

    with torch.no_grad():
        for batch in loader:
            xb, yb = batch
            xb = xb.to(device)

            out = model(xb).cpu().numpy()
            preds_scaled.append(out)
            true_scaled.append(yb.numpy())

    preds_scaled = np.concatenate(preds_scaled)
    true_scaled = np.concatenate(true_scaled)

    # Undo scaling
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    true = y_scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------
    out_df = pd.DataFrame({
        "prediction": preds,
        "target": true,
        "index": np.arange(len(preds)) + args.sequence_length
    })
    out_df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    r2 = r2_score(true, preds)
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)

    print(f"R²  = {r2:.6f}")
    print(f"MSE = {mse:.6f}")
    print(f"MAE = {mae:.6f}")


if __name__ == "__main__":
    main()
