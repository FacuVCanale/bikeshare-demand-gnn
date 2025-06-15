import torch
import torch.nn as nn
import torch.nn.functional as F

class BikeDestETAEnhanced(nn.Module):
    """
    Multi-task network with categorical embeddings,
    learned task-uncertainty weights and ETA variance output.
    """
    def __init__(
        self,
        cat_dims,              # dict: {feature_name: num_categories}
        cont_dim,              # Nº de features continuas
        num_stations,          # clases destino
        emb_dim_rule=lambda n: min(50, round(1.6 * n**0.25)),
        shared_dims=[256, 128],
        dropout=0.2
    ):
        super().__init__()

        # --- Embedding layers ---------------
        self.emb_layers = nn.ModuleDict({
            k: nn.Embedding(num, emb_dim_rule(num))
            for k, num in cat_dims.items()
        })
        total_emb = sum(e.embedding_dim for e in self.emb_layers.values())

        # --- Normalización de continuas -----
        self.cont_norm = nn.BatchNorm1d(cont_dim, affine=False)

        # --- Bloque compartido --------------
        layers = []
        in_dim = total_emb + cont_dim
        for h in shared_dims:
            layers += [nn.Linear(in_dim, h),
                       nn.ReLU(),
                       nn.LayerNorm(h),
                       nn.Dropout(dropout)]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # --- Cabeza destino -----------------
        self.dest_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_stations)       # logits
        )

        # --- Cabeza ETA (μ y logσ²) ---------
        self.eta_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)                   # [μ, logσ2]
        )

        # --- Log-sigmas (tarea) -------------
        self.log_vars = nn.Parameter(torch.zeros(2))  # [s_dest, s_eta]

    # ---------- Forward --------------------
    def forward(self, x_cat, x_cont):
        # Embeddings
        emb = [layer(x_cat[:, i])                 # i-ésima columna categórica
               for i, layer in enumerate(self.emb_layers.values())]
        emb = torch.cat(emb, dim=1)

        # Continuas
        cont = self.cont_norm(x_cont.float())

        # Mezclar y pasar por bloque compartido
        shared = self.shared(torch.cat([emb, cont], dim=1))

        # Predicciones
        dest_logits = self.dest_head(shared)      # (B, num_stations)
        eta_mu, eta_logvar = torch.chunk(self.eta_head(shared), 2, dim=1)
        eta_sigma = torch.exp(0.5 * eta_logvar)

        return dest_logits, eta_mu.squeeze(1), eta_sigma.squeeze(1)

    # ---------- Pérdida multitarea ---------
    def loss(self, dest_logits, dest_true, eta_mu, eta_sigma, eta_true,
             class_weights=None, label_smoothing=0.05):
        # Destino
        cls_loss = F.cross_entropy(
            dest_logits, dest_true,
            weight=class_weights,
            label_smoothing=label_smoothing
        )

        # ETA: NLL de Gauss parciales
        nll = 0.5 * torch.log(eta_sigma**2) + \
              0.5 * ((eta_true - eta_mu)**2) / (eta_sigma**2)

        eta_loss = nll.mean()

        # Ponderación aprendida
        precision_dest = torch.exp(-self.log_vars[0])
        precision_eta  = torch.exp(-self.log_vars[1])
        loss = precision_dest * cls_loss + self.log_vars[0] + \
               precision_eta  * eta_loss + self.log_vars[1]

        return loss, cls_loss.detach(), eta_loss.detach()
