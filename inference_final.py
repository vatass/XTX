#!/usr/bin/env python3
"""
Inference script for the XTX LOB forecasting task.

This script:
  • Loads RAW limit order book data
  • Loads saved fold-7 validation indices (ensures exact reconstruction)
  • Applies the SAME feature engineering used during training
  • Loads saved x-scaler and trained LSTM model
  • Builds sequences (seq_len = 40 by default)
  • Produces predictions.csv with 1 prediction per valid timestep
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import argparse


# ============================================================
#  LSTMRegressor — must match the training model 1:1
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)         
        out_last = out[:, -1, :]      
        out_norm  = self.layer_norm(out_last)
        pred = self.fc(out_norm).squeeze(-1)
        return pred


# ============================================================
#  Match training feature engineering
# ============================================================
def preprocess_features(df, top_levels):

    raw_cols = [
        f"{side}{attr}_{lvl}"
        for side in ["ask", "bid"]
        for attr in ["Rate", "Size", "Nc"]
        for lvl in top_levels
    ]

    rate_cols = [col for col in raw_cols if "Rate" in col]

    # --- Mid-price ---
    df["mid_price"] = (df["askRate_0"] + df["bidRate_0"]) / 2

    # --- Relative prices ---
    for col in rate_cols:
        df[col + "_rel"] = df[col] - df["mid_price"]

    df.drop(columns=["mid_price"], inplace=True)

    # --- Momentum features ---
    df["mid_price_diff"] = df["askRate_0"].diff() - df["bidRate_0"].diff()
    df["spread_diff"]    = (df["askRate_0"] - df["bidRate_0"]).diff()

    # --- Drop raw rate columns (as in training) ---
    df.drop(columns=rate_cols, inplace=True)

    # --- Remove NaN from diffs ---
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
#  Dataset for inference (no y)
# ============================================================
class LOBSeqDataset(Dataset):
    def __init__(self, X, sequence_length=40):
        self.X = X
        self.sequence_length = sequence_length
        self.X_seq = []

        for i in range(sequence_length, len(X)):
            self.X_seq.append(X[i-sequence_length+1:i+1])

        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx]


# ============================================================
# Utilities
# ============================================================
def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data",      required=True)
    parser.add_argument("--fold_idx_path", required=True)
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--x_scaler",      required=True)
    parser.add_argument("--output_path",   default="predictions.csv")
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--hidden_dim",      type=int, default=32)
    parser.add_argument("--num_layers",      type=int, default=2)
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--top_levels",      type=int, default=4)
    args = parser.parse_args()

    print("Loading raw LOB data...")
    df_raw = pd.read_csv(args.raw_data)

    print("Loading fold-7 indices...")
    idx = np.load(args.fold_idx_path)

    print("Extracting fold-7 rows BEFORE preprocessing...")
    df_val_raw = df_raw.iloc[idx].copy()

    print("Applying feature engineering...")
    df_val_fe = preprocess_features(df_val_raw, range(args.top_levels))

    print("Feature shape after preprocessing:", df_val_fe.shape)

    print("Loading x-scaler...")
    x_scaler = load_scaler(args.x_scaler)
    X_scaled = x_scaler.transform(df_val_fe.values)

    print("Building sequence dataset...")
    dataset = LOBSeqDataset(X_scaled, sequence_length=args.sequence_length)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    print("Loading trained LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRegressor(
        input_dim=df_val_fe.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print("Running inference...")
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())

    preds = np.concatenate(preds)

    print(f"Saving predictions to {args.output_path} ...")
    pd.DataFrame({
        "prediction_scaled": preds,
        "row_index": np.arange(args.sequence_length, args.sequence_length + len(preds))
    }).to_csv(args.output_path, index=False)

    print("Inference complete.")


if __name__ == "__main__":
    main()
