import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Dict, List, Tuple, Optional

from src.models.base_model import BaseModel

__all__ = [
    "BikeDestETAEnhanced",  # the raw network
    "BikeDestETAEnhancedModel"  # the BaseModel-compatible wrapper
]


class BikeDestETAEnhanced(nn.Module):
    """
    Multi-task network with categorical embeddings,
    learned task-uncertainty weights and ETA variance output.
    The network simultaneously predicts:
      1. Destination station (classification)
      2. Estimated Trip Arrival (ETA) as mean and variance (regression)
    """

    def __init__(
        self,
        cat_dims: Dict[str, int],        # {'feature_name': num_categories}
        cont_dim: int,                   # number of continuous features
        num_stations: int,               # number of destination classes
        emb_dim_rule=lambda n: min(50, round(1.6 * n ** 0.25)),
        shared_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        shared_dims = shared_dims or [256, 128]

        # --- Embedding layers -------------------------------------------------
        self.emb_layers = nn.ModuleDict({
            k: nn.Embedding(num, emb_dim_rule(num))
            for k, num in cat_dims.items()
        })
        total_emb = sum(e.embedding_dim for e in self.emb_layers.values())

        # --- Normalisation for continuous features ---------------------------
        self.cont_norm = nn.BatchNorm1d(cont_dim, affine=False)

        # --- Shared MLP block -------------------------------------------------
        layers: List[nn.Module] = []
        in_dim = total_emb + cont_dim
        for h in shared_dims:
            layers += [nn.Linear(in_dim, h),
                       nn.ReLU(),
                       nn.LayerNorm(h),
                       nn.Dropout(dropout)]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # --- Destination head (classification) -------------------------------
        self.dest_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_stations)  # raw logits
        )

        # --- ETA head: returns mean (mu) & log-variance (logσ²) ---------------
        self.eta_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # [μ, logσ²]
        )

        # --- Trainable log-variances to weigh the two tasks ------------------
        #   Following Kendall et al. 2018 for multi-task uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(2))  # [s_dest, s_eta]

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x_cat : LongTensor, shape (B, n_cat)
            Categorical features encoded as integer IDs.
        x_cont : Tensor, shape (B, cont_dim)
            Continuous features (float32/fp16).
        Returns
        -------
        dest_logits : Tensor, shape (B, num_stations)
        eta_mu      : Tensor, shape (B,)
        eta_sigma   : Tensor, shape (B,) – standard deviation (positive).
        """
        # Embeddings ---------------------------------------------------------
        emb = [layer(x_cat[:, i]) for i, layer in enumerate(self.emb_layers.values())]
        emb = torch.cat(emb, dim=1)

        # Continuous features -------------------------------------------------
        cont = self.cont_norm(x_cont.float())

        # Shared representation ----------------------------------------------
        shared = self.shared(torch.cat([emb, cont], dim=1))

        # Heads ---------------------------------------------------------------
        dest_logits = self.dest_head(shared)
        eta_mu, eta_logvar = torch.chunk(self.eta_head(shared), 2, dim=1)
        eta_sigma = torch.exp(0.5 * eta_logvar)

        return dest_logits, eta_mu.squeeze(1), eta_sigma.squeeze(1)

    # ---------------------------------------------------------------------
    # Multitask Loss
    # ---------------------------------------------------------------------
    def multitask_loss(
        self,
        dest_logits: torch.Tensor,
        dest_true: torch.Tensor,
        eta_mu: torch.Tensor,
        eta_sigma: torch.Tensor,
        eta_true: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute weighted multitask loss.

        Returns
        -------
        total_loss, cls_loss (detached), eta_loss (detached)
        """
        # Destination classification (Cross-Entropy)
        cls_loss = F.cross_entropy(
            dest_logits,
            dest_true,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        # ETA Negative Log-Likelihood of Gaussian ---------------------------
        # nll = 0.5 * log σ² + 0.5 * ( (y − μ)² / σ² )
        nll = 0.5 * torch.log(eta_sigma ** 2) + 0.5 * ((eta_true - eta_mu) ** 2) / (eta_sigma ** 2)
        eta_loss = nll.mean()

        # Learned weighting --------------------------------------------------
        precision_dest = torch.exp(-self.log_vars[0])
        precision_eta = torch.exp(-self.log_vars[1])
        total_loss = (
            precision_dest * cls_loss + self.log_vars[0] +
            precision_eta * eta_loss + self.log_vars[1]
        )

        return total_loss, cls_loss.detach(), eta_loss.detach()


class BikeDestETAEnhancedModel(BaseModel):
    """Wrapper that adapts `BikeDestETAEnhanced` to the `BaseModel` interface."""

    def __init__(
        self,
        cat_dims: Dict[str, int],
        cont_dim: int,
        num_stations: int,
        device: Optional[str] = None,
        **network_kwargs,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Internal nn.Module --------------------------------------------------
        self.model = BikeDestETAEnhanced(cat_dims, cont_dim, num_stations, **network_kwargs).to(self.device)
        self.cat_dims = cat_dims
        self.cont_dim = cont_dim
        self.num_stations = num_stations
        self.hps: Dict = {
            "epochs": 20,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "log_interval": 100,
        }
        # will be initialised in `set_hps` or `train`
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------
    def prepare_dataset(self, dataset):
        """Delegate to dataset for proper preparation (double dispatch)."""
        if hasattr(dataset, "prepare_for_nn"):
            dataset.prepare_for_nn(self)
        elif hasattr(dataset, "prepare_for_torch"):
            dataset.prepare_for_torch(self)
        else:
            raise AttributeError("Dataset lacks an appropriate prepare method for BikeDestETAEnhancedModel.")

    def set_hps(self, hps: Dict):
        """Set (and possibly override) hyper-parameters."""
        self.hps.update(hps)

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------
    def _get_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.get("lr", 1e-3),
            weight_decay=self.hps.get("weight_decay", 0.0),
        )

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
    ):
        """Train the network.

        Parameters
        ----------
        train_loader : DataLoader
            Yields (x_cat, x_cont, dest_true, eta_true)
        valid_loader : DataLoader, optional
            Same format as train_loader.
        """
        self.model.train()
        if self.optimizer is None:
            self.optimizer = self._get_optimizer()

        epochs = self.hps.get("epochs", 20)
        log_interval = self.hps.get("log_interval", 100)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                x_cat, x_cont, dest_true, eta_true = batch
                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)
                dest_true = dest_true.to(self.device)
                eta_true = eta_true.to(self.device)

                self.optimizer.zero_grad()
                dest_logits, eta_mu, eta_sigma = self.model(x_cat, x_cont)
                loss, _, _ = self.model.multitask_loss(dest_logits, dest_true, eta_mu, eta_sigma, eta_true)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Epoch {epoch:03d} | Batch {batch_idx + 1:04d} / {len(train_loader)} | Loss {loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch:03d} finished. Avg loss: {avg_loss:.4f}")

            # Optional validation -------------------------------------------------
            if valid_loader is not None:
                val_metrics = self.get_metrics(valid_loader)
                print(f"Valid – dest_acc: {val_metrics['dest_acc']:.3f} | eta_rmse: {val_metrics['eta_rmse']:.2f}")

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    @torch.no_grad()
    def predict(self, data_loader: DataLoader):
        """Predict destination class and ETA mean for all samples."""
        self.model.eval()
        dest_preds: List[int] = []
        eta_preds: List[float] = []

        for batch in data_loader:
            x_cat, x_cont, *_ = batch  # ignore labels if present
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            dest_logits, eta_mu, _ = self.model(x_cat, x_cont)
            dest_pred = dest_logits.argmax(dim=1).cpu()
            dest_preds.append(dest_pred)
            eta_preds.append(eta_mu.cpu())

        dest_preds_tensor = torch.cat(dest_preds)
        eta_preds_tensor = torch.cat(eta_preds)
        return dest_preds_tensor.numpy(), eta_preds_tensor.numpy()

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------
    @torch.no_grad()
    def get_metrics(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        eta_sq_error_sum = 0.0
        n_samples = 0

        for batch in data_loader:
            x_cat, x_cont, dest_true, eta_true = batch
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            dest_true = dest_true.to(self.device)
            eta_true = eta_true.to(self.device)

            dest_logits, eta_mu, _ = self.model(x_cat, x_cont)

            # Classification accuracy -------------------------------------
            preds = dest_logits.argmax(dim=1)
            correct += (preds == dest_true).sum().item()
            total += dest_true.size(0)

            # Regression RMSE ---------------------------------------------
            eta_sq_error_sum += ((eta_true - eta_mu) ** 2).sum().item()
            n_samples += eta_true.size(0)

        dest_acc = correct / total if total else 0.0
        eta_rmse = (eta_sq_error_sum / n_samples) ** 0.5 if n_samples else 0.0
        return {"dest_acc": dest_acc, "eta_rmse": eta_rmse}

    # -------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------
    def save(self, path: str):
        """Save model state-dict and hyper-parameters to *path* (torch)."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "hps": self.hps,
            "cat_dims": self.cat_dims,
            "cont_dim": self.cont_dim,
            "num_stations": self.num_stations,
        }
        torch.save(checkpoint, path)
        print(f"Model checkpoint saved to {path}")

    def load(self, path: str):
        """Load model state-dict from *path* and replace current state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.hps = checkpoint.get("hps", self.hps)
        print(f"Model checkpoint loaded from {path}")
        return self.model 