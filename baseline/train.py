import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParticleDisplacementDataset(Dataset):
    """
    Each sample is one timestep transition: given positions at step t,
    predict displacement to step t+1.

    Returns:
        pos:      (P, 3)  positions at timestep t
        disp:     (P, 3)  displacement from t to t+1
        mat_ids:  (P,)    remapped material IDs (contiguous 0..M-1)
    """
    def __init__(self, positions, mat_ids, timestep_indices):
        """
        positions:        (T, P, 3) full position tensor
        mat_ids:          (P,)      material IDs
        timestep_indices: list of t values to include (each gives t -> t+1)
        """
        self.positions = positions
        self.mat_ids   = mat_ids
        self.indices   = timestep_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        pos  = self.positions[t]            # (P, 3)
        disp = self.positions[t+1] - pos    # (P, 3)
        return pos, disp, self.mat_ids


def remap_material_ids(mat_ids_raw):
    """Remap arbitrary integer IDs to contiguous 0..M-1."""
    unique_ids = mat_ids_raw.unique(sorted=True)
    remap = {old.item(): new for new, old in enumerate(unique_ids)}
    mat_ids = torch.tensor(
        [remap[i.item()] for i in mat_ids_raw], dtype=torch.long
    )
    print(f"Material ID remap: {remap}")
    return mat_ids, len(unique_ids)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ParticleTransformer(nn.Module):
    """
    Transformer that predicts per-particle displacement.

    Input per particle:  position (3) + material embedding (material_embed_dim)
    Output per particle: displacement (3)

    Particles attend to each other via self-attention, conditioned on
    material type through the embedding.
    """
    def __init__(self, hidden_size, num_heads, num_layers, dropout,
                 num_materials, material_embed_dim):
        super().__init__()

        self.material_embed = nn.Embedding(num_materials, material_embed_dim)

        input_dim = 3 + material_embed_dim
        self.input_proj = nn.Linear(input_dim, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_size, 3)

    def forward(self, pos, mat_ids):
        """
        pos:     (B, P, 3)
        mat_ids: (P,) or (B, P)
        Returns: (B, P, 3) predicted displacements
        """
        if mat_ids.dim() == 1:
            mat_ids = mat_ids.unsqueeze(0).expand(pos.shape[0], -1)  # (B, P)

        mat_emb = self.material_embed(mat_ids)          # (B, P, material_embed_dim)
        x = torch.cat([pos, mat_emb], dim=-1)           # (B, P, 3 + material_embed_dim)
        x = self.input_proj(x)                          # (B, P, hidden_size)
        x = self.transformer(x)                         # (B, P, hidden_size)
        disp_pred = self.output_proj(x)                 # (B, P, 3)
        return disp_pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg_path="default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    save_dir = train_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # --- Logging ---
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Config: {cfg}")

    # --- Load data ---
    positions = torch.load(
        os.path.join(data_cfg["data_path"], "GtX.pt"),
        map_location="cpu"
    ).float()                                           # (T, P, 3)
    mat_ids_raw = torch.load(
        os.path.join(data_cfg["data_path"], "MaterialID.pt"),
        map_location="cpu"
    ).long()                                            # (P,)

    T, P, _ = positions.shape
    mat_ids, num_materials = remap_material_ids(mat_ids_raw)
    logger.info(f"Data: T={T}, P={P}, num_materials={num_materials}")

    N_train = train_cfg["N_train"]
    use_val = train_cfg["use_validation"]
    N_val   = train_cfg["N_val"] if use_val else 0

    assert N_train + N_val < T, (
        f"N_train ({N_train}) + N_val ({N_val}) must be < T ({T})"
    )

    # Timestep indices: t gives transition t -> t+1
    train_indices = list(range(0, N_train))
    val_indices   = list(range(N_train, N_train + N_val)) if use_val else []

    train_dataset = ParticleDisplacementDataset(positions, mat_ids, train_indices)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    if use_val and len(val_indices) > 0:
        val_dataset = ParticleDisplacementDataset(positions, mat_ids, val_indices)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            drop_last=False,
        )
    else:
        val_loader = None

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = ParticleTransformer(
        hidden_size=model_cfg["hidden_size"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_materials=num_materials,
        material_embed_dim=model_cfg["material_embed_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    loss_fn   = nn.MSELoss()

    mat_ids_device = mat_ids.to(device)

    train_losses = []
    val_losses   = []
    epochs_logged = []

    # --- Training loop ---
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for pos_batch, disp_batch, _ in train_loader:
            pos_batch  = pos_batch.to(device)   # (B, P, 3)
            disp_batch = disp_batch.to(device)  # (B, P, 3)

            optimizer.zero_grad()
            disp_pred = model(pos_batch, mat_ids_device)
            loss = loss_fn(disp_pred, disp_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # --- Validation ---
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for pos_batch, disp_batch, _ in val_loader:
                    pos_batch  = pos_batch.to(device)
                    disp_batch = disp_batch.to(device)
                    disp_pred  = model(pos_batch, mat_ids_device)
                    val_loss_total += loss_fn(disp_pred, disp_batch).item()
                    val_batches += 1
            avg_val_loss = val_loss_total / val_batches

        # --- Logging ---
        if epoch % train_cfg["log_every"] == 0:
            if avg_val_loss is not None:
                logger.info(
                    f"Epoch {epoch:4d} | train_loss={avg_train_loss:.6f} "
                    f"| val_loss={avg_val_loss:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch:4d} | train_loss={avg_train_loss:.6f}"
                )

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epochs_logged.append(epoch)

        # --- Checkpoint ---
        if epoch % train_cfg["checkpoint_every"] == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_materials": num_materials,
                "mat_ids": mat_ids,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # --- Save final checkpoint ---
    final_ckpt_path = os.path.join(save_dir, "checkpoint_final.pt")
    torch.save({
        "epoch": train_cfg["max_epochs"] - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_materials": num_materials,
        "mat_ids": mat_ids,
    }, final_ckpt_path)
    logger.info(f"Saved final checkpoint: {final_ckpt_path}")

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_logged, train_losses, label="Train loss", color="steelblue")
    if any(v is not None for v in val_losses):
        val_losses_clean = [v if v is not None else float("nan") for v in val_losses]
        ax.plot(epochs_logged, val_losses_clean, label="Val loss", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved loss plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
