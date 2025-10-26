import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import argparse
# 🧠 LSTM Model
import torch
import torch.nn as nn
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class HFTLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
        super(HFTLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)             # (batch_size, seq_len, hidden_dim1)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)           # (batch_size, seq_len, hidden_dim2)
        out = self.dropout2(out[:, -1, :]) # Take last time step
        out = self.fc(out)                 # (batch_size, 1)
        return out.squeeze(-1)             # (batch_size,)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 🧠 LSTM with dropout between layers (only active if num_layers > 1)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ⚖️ Layer normalization to stabilize hidden activations
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 🧩 Final fully connected projection with dropout for regularization
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x)

        # Use last timestep output
        out_last = out[:, -1, :]

        # Normalize before feeding to linear layer
        out_norm = self.layer_norm(out_last)

        # Predict final target
        out_final = self.fc(out_norm).squeeze()

        return out_final

# 🧪 Dataset
class LOBSeqDataset(Dataset):
    def __init__(self, X, y, sequence_length=1):
        self.sequence_length = sequence_length
        self.X_seq = []
        self.y_seq = []

        # Only keep complete sequences
        for i in range(sequence_length, len(X)):
            self.X_seq.append(X[i - sequence_length:i])
            self.y_seq.append(y[i])

        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)  # (N, seq_len, feat_dim)
        self.y_seq = torch.tensor(np.array(self.y_seq), dtype=torch.float32)

    def __len__(self):
        return self.X_seq.shape[0]  

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]

# ⚙️ Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(dataloader, desc="Training", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        wandb.log({"batch_train_loss": loss.item()})
    return total_loss / len(dataloader.dataset)

# 🎯 Evaluation loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "val_loss": total_loss / len(dataloader.dataset),
        "val_r2": r2_score(targets, preds),
        "val_mse": mean_squared_error(targets, preds),
        "val_mae": mean_absolute_error(targets, preds), 
    }

# 🧠 Main training entry
from sklearn.model_selection import TimeSeriesSplit
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import argparse
import wandb
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_dim1", type=int, default=32)
    parser.add_argument("--hidden_dim2", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--data_path", type=str, default="train.csv.gz")
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    # --- Load raw data ---
    df = pd.read_csv(args.data_path)
    top_levels = range(4)

    raw_cols = [f"{side}{attr}_{lvl}" for side in ["ask", "bid"] for attr in ["Rate", "Size", "Nc"] for lvl in top_levels]
    rate_cols = [col for col in raw_cols if "Rate" in col]
    size_nc_cols = [col for col in raw_cols if "Size" in col or "Nc" in col]

    # Mid-price centering
    df["mid_price"] = (df["askRate_0"] + df["bidRate_0"]) / 2
    for col in rate_cols:
        df[col + "_rel"] = df[col] - df["mid_price"]
    df.drop(columns=["mid_price"], inplace=True)

    # Momentum features
    df["mid_price_diff"] = df["askRate_0"].diff() - df["bidRate_0"].diff()
    df["spread_diff"] = (df["askRate_0"] - df["bidRate_0"]).diff()

    # Drop raw rate columns
    df.drop(columns=rate_cols, inplace=True)

    # Clean up
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features and target
    feature_cols = [col for col in df.columns if col != "y"]
    X_all = df[feature_cols].values
    y_all = df[["y"]].values  # 2D for scaler

    # Before TimeSeriesSplit
    global_y_scaler = StandardScaler()
    y_all_scaled = global_y_scaler.fit_transform(y_all)

    # TimeSeries Cross-validation
    tscv = TimeSeriesSplit(n_splits=2)
    fold = 0

    for train_idx, val_idx in tscv.split(X_all):
        fold += 1
        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train_raw, y_val_raw = y_all[train_idx], y_all[val_idx]

        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train_raw)
        X_val = x_scaler.transform(X_val_raw)
        y_train = y_all_scaled[train_idx].flatten()
        y_val = y_all_scaled[val_idx].flatten()

        # Dataset & loader
        train_ds = LOBSeqDataset(X_train, y_train, sequence_length=args.sequence_length)
        val_ds = LOBSeqDataset(X_val, y_val, sequence_length=args.sequence_length)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        # Model & training
        input_dim = X_train.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)

        # 🧩 Use all available GPUs
        if torch.cuda.device_count() > 1:
            print(f"🔁 Using {torch.cuda.device_count()} GPUs via DataParallel.")
            model = nn.DataParallel(model)

        # model = HFTLSTM(input_dim=input_dim, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, dropout=args.dropout).to(device)   

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()

        wandb.init(
            project="lob-lstm-cv",
            config=vars(args),  # ✅ logs all argparse arguments
            reinit=True
        )

        wandb.watch(model)

        best_val_loss = float("inf")
        stop_counter = 0
        patience = 10

        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            scheduler.step()  # <-- update the learning rate here
            metrics = evaluate(model, val_loader, criterion, device)
            wandb.log({"train_loss": train_loss, **metrics, "epoch": epoch, "fold": fold})

            print(f"[Fold {fold} | Epoch {epoch}] 🧠 Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {metrics['val_loss']:.4f} | R²: {metrics['val_r2']:.4f} | "
                  f"MSE: {metrics['val_mse']:.6f} | MAE: {metrics['val_mae']:.6f}")

            wandb.log({'val_r2': metrics['val_r2']})

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                stop_counter = 0
                torch.save(model.module.state_dict(), f"best_lstm_fold{fold}.pt")                
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print("⏹️ Early stopping.")
                    break

        print(f"✅ Finished Fold {fold}. Best model saved as 'best_lstm_fold{fold}.pt'")
        wandb.finish()

if __name__ == "__main__":
    main()
    ### 10/26/2025. The accelerated version.