import polars as pl
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from src.utils.path_utils import data_path, model_path


def main():
    """Train a simple feed-forward neural network on the wide feature set."""

    # Paths to the wide feature checkpoint files
    X_PATH = r"C:\Users\xxx\Documents\GitHub\EcoBici-AI\notebooks\feature_checkpoints\X_wide.parquet"
    Y_PATH = r"C:\Users\xxx\Documents\GitHub\EcoBici-AI\notebooks\feature_checkpoints\y_wide.parquet"

    print("\nLoading wide feature set …")
    X_pl = pl.read_parquet(X_PATH)
    y_pl = pl.read_parquet(Y_PATH)

    # ------------------------------------------------------------------
    # 1) Ensure *all* feature columns are numeric so that they can be fed
    #    to the neural network. We do *not* modify the parquet file on
    #    disk, only the in-memory frame.
    # ------------------------------------------------------------------
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )

    non_numeric_cols = [c for c, dt in X_pl.schema.items() if dt not in numeric_types]

    if "ts_start" in non_numeric_cols:
        non_numeric_cols.remove("ts_start")
        X_pl = X_pl.drop("ts_start")

    if non_numeric_cols:
        print(f"Converting {len(non_numeric_cols)} non-numeric columns → numeric …")
        for col in non_numeric_cols:
            try:
                dtype = X_pl.schema[col]
                if dtype == pl.Boolean:
                    # Booleans → {0,1}
                    X_pl = X_pl.with_columns(pl.col(col).cast(pl.Int8))
                else:
                    # Categorical / string → category codes (int32)
                    X_pl = X_pl.with_columns(
                            pl.col(col).cast(pl.Categorical).cast(pl.Int32)
                        )
            except:
                print(f"Error converting column {col} to numeric")

    # ------------------------------------------------------------------
    # 2) Convert to NumPy, fix NaNs/Infs and *standardise* the features.
    # ------------------------------------------------------------------
    X = X_pl.to_numpy()  # shape (n_samples, n_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Persist the scaler so we can apply the exact same transform later
    joblib.dump(scaler, data_path("scaler.pkl"))
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 3) Handle the *targets*: replace NaNs and *standardise* as well so
    #    the scale of the MSE loss is ~O(1). We keep a scaler to bring
    #    predictions back to the original space for metrics.
    # ------------------------------------------------------------------
    y_raw = y_pl.to_numpy()  # shape (n_samples, 382)
    y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Log-transform to dampen extreme peaks in counts
    y_log = np.log1p(y_raw)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_log)
    joblib.dump(y_scaler, data_path("y_scaler.pkl"))

    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    batch_size = 1024
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model definition
    output_dim = y.shape[1]

    # ------------------------------------------------------------------
    # Residual architecture ------------------------------------------------
    # ------------------------------------------------------------------
    class ResidualBlock(nn.Module):
        def __init__(self, features: int, hidden: int, dropout: float = 0.3):
            super().__init__()
            self.lin1 = nn.Linear(features, hidden)
            self.bn1 = nn.BatchNorm1d(hidden)
            self.lin2 = nn.Linear(hidden, features)
            self.bn2 = nn.BatchNorm1d(features)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.lin1(x)))
            out = self.dropout(out)
            out = self.bn2(self.lin2(out))
            return F.relu(out + residual)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        ResidualBlock(2048, 1024),
        nn.Dropout(0.3),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, output_dim),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # Mixed-precision utilities (CUDA only)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    EPOCHS = 100
    patience = 10  # for early stopping
    best_val_loss = math.inf
    epochs_no_improve = 0
    print("\nStarting training …")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                preds = model(xb)
                loss = criterion(preds, yb)

            if not torch.isfinite(loss):
                print(f"Non-finite loss detected ({loss.item()}) — stopping training early.")
                return

            if use_amp:
                scaler.scale(loss).backward()
                # Unscale for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds_scaled = []
        val_loss_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    preds = model(xb)
                    vloss = criterion(preds, yb)
                val_loss_total += vloss.item() * xb.size(0)
                val_preds_scaled.append(preds.cpu())

        val_loss = val_loss_total / len(val_loader.dataset)

        # Convert predictions back to original units for human-readable metrics
        val_preds_scaled = torch.cat(val_preds_scaled, dim=0).cpu().numpy()
        val_preds_log = y_scaler.inverse_transform(val_preds_scaled)

        # Replace any NaN/inf with finite bounds before exp
        max_log = 18.0  # exp(18) ≈ 6.5e7
        val_preds_log = np.nan_to_num(val_preds_log, nan=0.0, posinf=max_log, neginf=0.0)
        val_preds_log_clamped = np.clip(val_preds_log, a_min=0.0, a_max=max_log).astype(np.float64)
        val_preds = np.expm1(val_preds_log_clamped)

        val_true_log = y_scaler.inverse_transform(np.nan_to_num(y_val, nan=0.0))
        val_true_log = np.nan_to_num(val_true_log, nan=0.0, posinf=max_log, neginf=0.0)
        val_true_log_clamped = np.clip(val_true_log, a_min=0.0, a_max=max_log).astype(np.float64)
        val_true = np.expm1(val_true_log_clamped)

        # Metrics in log space (numerically stable) and original scale
        mse_log = mean_squared_error(val_true_log, val_preds_log, multioutput="uniform_average")
        mae_log = mean_absolute_error(val_true_log, val_preds_log, multioutput="uniform_average")
        mse = mean_squared_error(val_true, val_preds, multioutput="uniform_average")
        mae = mean_absolute_error(val_true, val_preds, multioutput="uniform_average")
        r2 = r2_score(val_true, val_preds, multioutput="uniform_average")

        print(
            f"Epoch {epoch + 1:03}/{EPOCHS} — "
            f"Train Loss: {train_loss:.4f} | Val Loss (scaled): {val_loss:.4f} | "
            f"Val MSE (cnt): {mse:.2f} MAE (cnt): {mae:.2f} R2: {r2:.4f} | "
            f"Val MSE (log): {mse_log:.4f} MAE (log): {mae_log:.4f}"
        )

        # Scheduler & early-stopping bookkeeping
        scheduler.step(val_loss)

        if val_loss + 1e-5 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model so far
            torch.save(model.state_dict(), model_path("nn_model_best.pth"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Save model
    MODEL_PATH = model_path("nn_model_last.pth")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nNeural network model saved to {MODEL_PATH}. Best model saved to {model_path('nn_model_best.pth')}")


if __name__ == "__main__":
    main() 