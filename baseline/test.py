import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

from train import ParticleTransformer, remap_material_ids


def test(cfg_path="default.yaml", checkpoint_path=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    save_dir = train_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    positions = torch.load(
        os.path.join(data_cfg["data_path"], "GtX.pt"),
        map_location="cpu"
    ).float()                                           # (T, P, 3)
    mat_ids_raw = torch.load(
        os.path.join(data_cfg["data_path"], "MaterialID.pt"),
        map_location="cpu"
    ).long()

    T, P, _ = positions.shape
    mat_ids, num_materials = remap_material_ids(mat_ids_raw)

    N_train = train_cfg["N_train"]
    N_val   = train_cfg["N_val"] if train_cfg["use_validation"] else 0
    N_test_start = 0 # N_train + N_val

    assert N_test_start < T - 1, (
        f"No test timesteps available: N_train+N_val={N_test_start} >= T-1={T-1}"
    )

    print(f"Data: T={T}, P={P}, num_materials={num_materials}")
    print(f"Test rollout: timesteps {N_test_start} -> {T-1} "
          f"({T - 1 - N_test_start} steps)")

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ParticleTransformer(
        hidden_size=model_cfg["hidden_size"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_materials=num_materials,
        material_embed_dim=model_cfg["material_embed_dim"],
    ).to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir, "checkpoint_final.pt")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    mat_ids_device = mat_ids.to(device)

    # --- Autoregressive rollout ---
    # Start from ground truth position at N_test_start
    current_pos = positions[N_test_start].to(device)  # (P, 3)

    gt_positions   = positions[N_test_start:]          # (T_test+1, P, 3)
    T_test         = gt_positions.shape[0] - 1         # number of prediction steps

    pred_positions = torch.zeros(T_test + 1, P, 3)
    pred_positions[0] = current_pos.cpu()

    with torch.no_grad():
        for t in range(T_test):
            pos_input = current_pos.unsqueeze(0)       # (1, P, 3)
            disp_pred = model(pos_input, mat_ids_device)  # (1, P, 3)
            current_pos = current_pos + disp_pred.squeeze(0)
            pred_positions[t + 1] = current_pos.cpu()

    pred_np = pred_positions.numpy()                   # (T_test+1, P, 3)
    gt_np   = gt_positions.numpy()                     # (T_test+1, P, 3)

    # --- Compute errors ---
    # Per-timestep mean L2 error across all particles
    error = np.linalg.norm(pred_np - gt_np, axis=-1)  # (T_test+1, P)
    mean_error_per_step = error.mean(axis=1)           # (T_test+1,)

    # Per-material error
    mat_ids_np = mat_ids.numpy()
    unique_mats = np.unique(mat_ids_np)
    per_mat_error = {}
    for mat_id in unique_mats:
        mask = mat_ids_np == mat_id
        per_mat_error[mat_id] = error[:, mask].mean(axis=1)

    mean_error_overall = mean_error_per_step.mean()
    final_error        = mean_error_per_step[-1]

    print(f"\nTest rollout error summary:")
    print(f"  Mean position error (averaged over rollout): {mean_error_overall:.6f}")
    print(f"  Final frame position error:                  {final_error:.6f}")
    for mat_id in unique_mats:
        mat_mean  = per_mat_error[mat_id].mean()
        mat_final = per_mat_error[mat_id][-1]
        print(f"  Material {mat_id} | mean: {mat_mean:.6f}, final: {mat_final:.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test Rollout: Mean Particle Position Error", fontsize=13)

    timesteps = np.arange(T_test + 1)

    axes[0].plot(timesteps, mean_error_per_step, color="steelblue", linewidth=1.5)
    axes[0].set_title("All particles")
    axes[0].set_xlabel("Rollout step")
    axes[0].set_ylabel("Mean L2 error")
    axes[0].grid(True, alpha=0.3)

    colors = ["royalblue", "tomato", "green", "purple", "orange"]
    for i, mat_id in enumerate(unique_mats):
        axes[1].plot(
            timesteps, per_mat_error[mat_id],
            label=f"Material {mat_id}",
            color=colors[i % len(colors)],
            linewidth=1.5,
        )
    axes[1].set_title("Per material")
    axes[1].set_xlabel("Rollout step")
    axes[1].set_ylabel("Mean L2 error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "test_rollout_error.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\nSaved error plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Defaults to checkpoint_final.pt in save_dir.")
    args = parser.parse_args()
    test(args.config, args.checkpoint)
