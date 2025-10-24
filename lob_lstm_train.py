import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
from datetime import datetime

# 🧠 LSTM Model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# 🧪 Dataset
class LOBSeqDataset(Dataset):
    def __init__(self, X, y, sequence_length=1):
        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, sequence_length, X.shape[1] // sequence_length)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ⚙️ Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(dataloader, desc="Training", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        print(xb.shape, yb.shape)
        sys.exit(0)
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
def main():

    # TODO add arguments for sequence length, batch size, learning rate, and number of epochs
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
     
    # add arguments for the model architecture
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    # add arguments for the data path
    parser.add_argument("--data_path", type=str, default="train.csv.gz")

    args = parser.parse_args()

    # read the data

    df = pd.read_csv("train.csv.gz")

    # ✅ Extract top-4 level raw features (24 features)
    top_levels = range(4)
    raw_cols = [f"{side}{attr}_{lvl}" for side in ["ask", "bid"] for attr in ["Rate", "Size", "Nc"] for lvl in top_levels]
    X = df[raw_cols].fillna(0.0).values  # ✅ Replace NaNs with 0
    y = df["y"].values

    # Verify the train dataset if it contains any Nan

    # ✅ Chronological split
    split = int(0.8 * len(df))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # ✅ Normalize features
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)

    # ✅ Normalize targets
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # ✅ Dataset and Loader
    train_ds = LOBSeqDataset(X_train, y_train, sequence_length=20)
    val_ds = LOBSeqDataset(X_val, y_val, sequence_length=20)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    # Verify the train dataset if it contains any Nan
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("❌ Train dataset contains NaN values.")
  


    # Verify the validation dataset if it contains any Nan
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        print("❌ Validation dataset contains NaN values.")
        exit()

    # ✅ Model + Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_dim=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # ✅ wandb init
    wandb.init(project="lob-lstm-regression", name=f"LSTM_{datetime.now().strftime('%Y%m%d_%H%M')}")
    wandb.watch(model)

    best_val_loss = float("inf")
    patience = 5
    stop_counter = 0

    for epoch in range(1, 51):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)
        wandb.log({
            "train_loss": train_loss,
            **metrics,
            "epoch": epoch
        })

        print(f"[Epoch {epoch:2d}] 🧠 Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
              f"R²: {metrics['val_r2']:.4f} | MSE: {metrics['val_mse']:.6f} | MAE: {metrics['val_mae']:.6f}")

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            stop_counter = 0
            torch.save(model.state_dict(), "best_lstm_model.pt")
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("⏹️ Early stopping.")
                break

    print("✅ Finished. Best model saved as 'best_lstm_model.pt'.")

if __name__ == "__main__":
    main()
