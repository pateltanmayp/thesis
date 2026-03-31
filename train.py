from collections import OrderedDict
import numpy as np
import hydra
import omegaconf
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
from einops.layers.torch import Rearrange
import sys
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import time
import random

try:
    sys.path.insert(0, "/home/tanmay/thesis/ICKANs/")
    from core import *
    import drivers.config as c
    from ickan import *
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("Warning: KAN not available, falling back to MLP stress model.")

torch.autograd.set_detect_anomaly(True)


class MaterialHyperNet(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
            nn.Softplus()
        )

    def forward(self, z):
        return self.net(z)


class BranchingConstitutiveStress(nn.Module):
    """
    Branching elastic/plastic constitutive model that replaces the MLP
    stress_model in FprojNN_StressNN.

    Args:
        hidden_size:  width of the MLP fallback and hypernetwork layers
        embed_dim:    dimensionality of the material latent z
        n_hidden:     KAN width list e.g. [3, 8, 8, 1] (only used if use_kan)
        grid_range:   KAN grid range (only used if use_kan)
        use_kan:      if True use KAN elastic branch, else use MLP
        seed:         random seed for KAN initialisation
    """
    def __init__(self, hidden_size, embed_dim, n_hidden=None,
                 grid_range=None, use_kan=True, seed=0):
        super().__init__()
        self.use_kan = use_kan
        self.embed_dim = embed_dim

        # --- Elastic branch ---
        # Input to elastic branch is 3 invariants (K1, K2, K3)
        if use_kan:
            assert KAN_AVAILABLE, "KAN requested but not importable."
            assert n_hidden is not None and grid_range is not None
            self.elastic_nn = KAN(
                width=n_hidden, grid=c.grid, k=c.spline_order,
                seed=seed, device='cuda', base_fun='zero', grid_eps=1.0,
                grid_range_0=grid_range, sp_trainable=c.sp_trainable,
                sb_trainable=c.sb_trainable,
                symbolic_enabled=c.symbolic_enabled,
                auto_save=False
            )
        else:
            # MLP fallback with hidden_size matching the rest of the network
            self.elastic_nn = nn.Sequential(
                nn.Linear(9, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 9),
                nn.SiLU(),
                nn.Linear(9, 1),
                nn.Softplus()
            )

        self.plastic_nn = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 9),
            nn.SiLU(),
            nn.Linear(9, 1),
            nn.Softplus()
        )

        self.elastic_scale  = MaterialHyperNet(embed_dim, out_dim=1)

        self.plastic_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        self.branch_weights = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.Softplus()
        )

    def compute_invariants(self, F_flat):
        """
        Compute 3 polyconvex invariants from a (P, 9) flattened 3D F.
        Uses the upper-left 2x2 block for the 2D-style invariants,
        which is a reasonable approximation and keeps the convexity
        properties of the original formulation.
        """
        F_flat = torch.clamp(F_flat, min=-5.0, max=5.0)

        # Extract 2D block from 3D F (row-major: F = [[F00,F01,F02],[F10,...],...])
        F00 = F_flat[:, 0:1]  # F[0,0]
        F01 = F_flat[:, 1:2]  # F[0,1]
        F10 = F_flat[:, 3:4]  # F[1,0]
        F11 = F_flat[:, 4:5]  # F[1,1]

        C00 = F00**2 + F10**2
        C01 = F00*F01 + F10*F11
        C11 = F01**2 + F11**2

        I1 = C00 + C11 + 1.0
        I3 = C00*C11 - C01**2
        eps = 1e-6
        I3_safe = torch.clamp(I3, min=eps)

        J  = torch.sqrt(I3_safe)
        K1 = I1 * torch.pow(I3_safe, -1.0/3.0) - 3.0
        K2 = torch.pow(
            (I1 + I3_safe - 1.0) * torch.pow(I3_safe, -2.0/3.0), 1.5
        ) - 3.0 * (3.0 ** 0.5)
        K3 = (J - 1.0)**2

        return torch.cat([K1, K2, K3], dim=1).double()  # (P, 3)

    def forward(self, F_flat, z):
        """
        Args:
            F_flat: (P, 9) flattened deformation gradient, requires_grad=True
            z:      (P, embed_dim) per-particle material latent
        Returns:
            stress_symmetric: (P, 3, 3)
        """
        F_flat = F_flat.double()
        z = z.double()

        W_elastic     = self.elastic_nn(F_flat)         # (P, 1)
        elastic_scale = self.elastic_scale(z)      # (P, 1)
        W_elastic     = elastic_scale * W_elastic

        W_plastic      = self.plastic_nn(F_flat)
        plastic_factor = self.plastic_gate(z)      # (P, 1)
        W_plastic      = plastic_factor * W_plastic

        weights = self.branch_weights(z)           # (P, 2)
        alpha   = weights / weights.sum(dim=1, keepdim=True)

        W = alpha[:, 0:1] * W_elastic + alpha[:, 1:2] * W_plastic  # (P, 1)

        # Stress via autodiff: dW/dF
        create_graph = torch.is_grad_enabled()
        P_flat = torch.autograd.grad(
            W.sum(), F_flat,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]                                       # (P, 9)

        stress = P_flat.view(-1, 3, 3)
        stress_symmetric = 0.5 * (stress + stress.permute(0, 2, 1))
        # stress_symmetric = torch.clamp(stress_symmetric, -1e4, 1e4)
        return stress_symmetric


class FprojNN_StressNN(nn.Module):
    def __init__(self, activation, Ftransform, hidden_size, embed_dim,
                 use_kan=False, n_hidden=None, grid_range=None, seed=0):
        super().__init__()

        print("NN Activation:", activation)
        if activation == "gelu":
            self.activation = nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.use_kan = use_kan

        self.fproj_model = nn.Sequential(OrderedDict([
            ('fc1_fproj', nn.Linear(27 + embed_dim, hidden_size, bias=True)),
            ('act1_fproj', self.activation),
            ('fc2_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
            ('act2_fproj', self.activation),
            ('fc3_fproj', nn.Linear(hidden_size, 9, bias=True)),
        ]))

        self.stress_model = BranchingConstitutiveStress(
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            n_hidden=n_hidden,
            grid_range=grid_range,
            use_kan=use_kan,
            seed=seed,
        )

    def Ftmp_U_sigma_Vt_transform(self, Ftmp):
        U, sigma, Vt = torch.linalg.svd(Ftmp)
        U_flatten    = self.flatten(U)
        Vt_flatten   = self.flatten(Vt)
        Ftmp_flatten = self.flatten(Ftmp)
        return torch.cat([Ftmp_flatten, U_flatten, Vt_flatten], dim=-1)

    def FFt_logJ_sigma_J_logJ1_J1_transform(self, F):
        Ft  = F.transpose(1, 2)
        FtF = torch.matmul(Ft, F)
        J   = torch.max(torch.det(F), torch.tensor([1e-6], device=F.device))
        J1  = torch.max(F[:, 0, 0], torch.tensor([1e-6], device=F.device))
        U, sigma, Vt = torch.svd(F)
        FtF_flatten  = self.flatten(FtF)
        J  = J.unsqueeze(-1)
        J1 = J1.unsqueeze(-1)
        R  = torch.matmul(U, Vt)
        strain = torch.cat(
            [sigma, FtF_flatten, J, torch.log(J), J1, torch.log(J1)], dim=-1
        )
        return strain, R

    def forward(self, Ftmp, F, C, latent, traj_ids):
        latent_particles = latent(traj_ids)  # (P, embed_dim)

        # --- Fproj (identical to original) ---
        Ftmp_flatten = self.Ftmp_U_sigma_Vt_transform(Ftmp)
        out_fproj    = self.fproj_model(
            torch.cat([Ftmp_flatten, latent_particles], dim=-1)
        )
        out_fproj = torch.clamp(out_fproj, -1.0, 1.0) # TODO: verify this doesn't break stuff
        Fproj = Ftmp + out_fproj.view(out_fproj.shape[0], 3, 3)

        # KAN path: pass flat F and latent to BranchingConstitutiveStress
        F_flat = self.flatten(F).float().detach().requires_grad_(True)
        stress_symmetric = self.stress_model(F_flat, latent_particles)
        # R correction is skipped for KAN — energy-based stress is
        # already frame-indifferent by construction via the invariants
        # else:
        #     # Original MLP path (unchanged)
        #     strain, R    = self.FFt_logJ_sigma_J_logJ1_J1_transform(F)
        #     C_flatten    = self.flatten(C)
        #     out_stress   = self.stress_model(
        #         torch.cat([strain, C_flatten, latent_particles], dim=-1)
        #     )
        #     stress       = out_stress.view(out_stress.shape[0], 3, 3)
        #     stress_symmetric = 0.5 * (stress + stress.permute(0, 2, 1))
        #     stress_symmetric = torch.matmul(R, stress_symmetric)

        return Fproj, stress_symmetric


class F_dataset(torch.utils.data.Dataset):
    def __init__(self, file_dir_list, local_dir):
        self.file_dir_list = file_dir_list
        self.local_dir = local_dir
        self.load_data()

    def load_data(self):
        self.all_data_dict = {}
        self.traj_id_dict = {}

        start_time = time.time()
        for idx, file_dir in enumerate(self.file_dir_list):
            print ("Load data: ", idx)

            idx_l = list(np.arange(0, 500))
            random.shuffle(idx_l)

            cur_Ftmp = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "GtFtmp.pt"), map_location='cpu')[1:]
            cur_F = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}" ,"GtF.pt"), map_location='cpu')[1:]
            cur_stress = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "GtStress.pt"), map_location='cpu')[1:]
            cur_C = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "GtC.pt"), map_location='cpu')[1:]
            if cur_C.shape[0] == 960:
                cur_Ftmp = torch.cat((cur_Ftmp, cur_Ftmp, cur_Ftmp[:80, :, :, :]), dim=0)
                cur_F = torch.cat((cur_F, cur_F, cur_F[:80, :, :, :]), dim=0)
                cur_stress = torch.cat((cur_stress, cur_stress, cur_stress[:80, :, :, :]), dim=0)
                cur_C = torch.cat((cur_C, cur_C, cur_C[:80, :, :, :]), dim=0)
            cur_traj_id = torch.full((cur_F.shape[0], cur_F.shape[1]), idx)

            self.sim_timesteps = cur_Ftmp.shape[0]
            self.sim_num_particles = cur_Ftmp.shape[1]

            input_Ftmp_tensor = cur_Ftmp.reshape(-1, 3, 3)
            input_F_tensor = cur_F.reshape(-1, 3, 3)
            gt_stress_tensor = cur_stress.reshape(-1, 3, 3)
            input_C_tensor = cur_C.reshape(-1, 3, 3)
            traj_ids = cur_traj_id.reshape(-1)
            self.traj_id_dict[idx] = f"{file_dir}"

            self.all_data_dict[idx] = {'input_F': input_F_tensor, 'gt_stress': gt_stress_tensor, 'traj_ids': traj_ids,
                                       'input_Ftmp': input_Ftmp_tensor, 'input_C': input_C_tensor}

        print ("Loaded data in: ", time.time() - start_time, " seconds.")

    def __len__(self):
        return len(self.file_dir_list) * self.sim_num_particles * self.sim_timesteps

    def __getitem__(self, idx):
        traj_idx = idx // (self.sim_num_particles * self.sim_timesteps)
        sample_idx = idx % (self.sim_num_particles * self.sim_timesteps)

        sample = {'input_F': self.all_data_dict[traj_idx]['input_F'][sample_idx],
                  'stress_target': self.all_data_dict[traj_idx]['gt_stress'][sample_idx],
                  'input_C': self.all_data_dict[traj_idx]['input_C'][sample_idx],
                  'traj_ids': self.all_data_dict[traj_idx]['traj_ids'][sample_idx],
                  'input_Ftmp': self.all_data_dict[traj_idx]['input_Ftmp'][sample_idx],
                  }
        return sample

@hydra.main(config_path='UniPhy/configs', config_name='default')
def main(cfg: omegaconf.DictConfig):

    ##### Environment and Logging Setup #####
    save_dir = cfg['train_cfg']['save_dir']
    local_dir = cfg['train_cfg']['local_dir']
    logger = logging.getLogger()

    os.makedirs(f"{local_dir}/{save_dir}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_dir}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_dir}/log.txt")
    with open(f"{local_dir}/{save_dir}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    fh.setLevel(logging.DEBUG) # or any level you want
    logger.addHandler(fh)

    ##### Load Dataset #####
    traj_l = ["elastic_diverse", "plasticine_diverse", "sand_diverse"] # Should have "non_newtonian_diverse" and "newtonian_diverse" after "elastic_diverse"
    traj_dir_list = []
    for tl in traj_l:
        for _dir_idx, _dir in enumerate(os.listdir(os.path.join(f"{local_dir}/dataset/", tl))):
            traj_dir_list.append(f"{tl}/{_dir}")

    count_traj = len(traj_dir_list)
    train_data = F_dataset(file_dir_list=traj_dir_list, local_dir=local_dir)
    train_loader = DataLoader(train_data, batch_size=cfg['train_cfg']['batch_size'], shuffle=False, drop_last=False)
    print ("Dataset size: ", len(train_data), "Number of trajectories: ", count_traj)
    with open(os.path.join(local_dir, save_dir, f"traj_name_id_{count_traj}.pkl"), "wb") as f:
        pkl.dump(train_data.traj_id_dict, f)

    ##### Model Setup #####

    # Define Latent
    embed_dim = cfg['train_cfg']['embed_dim']
    trajectory_latent = torch.nn.Embedding(count_traj, embed_dim).cuda()
    traj_mean, traj_std = 0., 1.
    torch.nn.init.normal_(trajectory_latent.weight, mean=traj_mean, std=traj_std)

    # Define Model
    use_kan = cfg['train_cfg']['use_kan']
    if use_kan:
        n_hidden = OmegaConf.to_container(cfg['train_cfg']['n_hidden'], resolve=True)
        grid_range = OmegaConf.to_container(cfg['train_cfg']['grid_range'], resolve=True)
    else:
        n_hidden = cfg['train_cfg']['n_hidden_ann']
        grid_range = None

    model = FprojNN_StressNN(
        activation=cfg['train_cfg']['nn_activation'],
        Ftransform=cfg['train_cfg']['Ftransform'],
        hidden_size=cfg['train_cfg']['hidden_size'],
        embed_dim=cfg['train_cfg']['embed_dim'],
        use_kan=use_kan,
        n_hidden=n_hidden,
        grid_range=grid_range,
        seed=0,
    ).cuda()
    
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": cfg['train_cfg']['lr']},
                                   {"params": trajectory_latent.parameters(), "lr": cfg['train_cfg']['lr'] * 10.}])
    total_epochs = cfg['train_cfg']['epochs']
    step_lr_step_size=cfg['train_cfg']['step_lr_step_size']
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr_step_size, gamma=0.9)

    loss_mse = nn.MSELoss()
    loss_mse_fproj = nn.MSELoss()
    for epoch in range(total_epochs):
        print ("Starting epoch: ", epoch)
        epoch_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0

        for idx, data in enumerate(train_loader):
            input_F, stress_target, traj_ids, input_Ftmp, input_C = data['input_F'].cuda(), data['stress_target'].cuda(), data['traj_ids'].cuda(), data['input_Ftmp'].cuda(), data['input_C'].cuda()

            optimizer.zero_grad()

            pred_Fproj, pred_stress = model(input_Ftmp, input_F, input_C, trajectory_latent, traj_ids)

            if len((torch.isnan(pred_stress) == True).nonzero()) > 0 or len((torch.isnan(pred_Fproj) == True).nonzero()) > 0:
                import ipdb; ipdb.set_trace()

            loss_l2 = loss_mse(pred_stress, stress_target)
            loss_l2_fproj = loss_mse_fproj(pred_Fproj, input_F)

            reg_loss = torch.norm(trajectory_latent.weight, dim=-1)
            loss_reg = 0.0001 * torch.mean(reg_loss)
            loss = loss_l2 + loss_l2_fproj + loss_reg

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reg_loss += loss_reg.item()
            num_batches += 1

            # logger.info(f"Epoch: {epoch}, Iter/Total: {idx}/{len(train_loader)}, Loss: {loss}, LR1: {scheduler1.get_last_lr()}")
            # if idx % 1000 == 0:
            #     writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + idx)
            #     writer.add_scalar('Stress Loss/train', loss_l2, epoch * len(train_loader) + idx)
            #     writer.add_scalar('Fproj Loss/train', loss_l2_fproj, epoch * len(train_loader) + idx)
            #     writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + idx)

            # if epoch < 2:
            #     scheduler1.step()

        if epoch % 1 == 0:
            torch.save(trajectory_latent, f"{local_dir}/{save_dir}/traj_latent_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{local_dir}/{save_dir}/model_{epoch}.pth")
        
        avg_loss = epoch_loss / num_batches
        avg_reg_loss = epoch_reg_loss / num_batches

        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Regression Loss/train_epoch', avg_reg_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        logger.info(f"Epoch: {epoch}, Avg Loss: {avg_loss}, Reg Loss: {avg_reg_loss}")

        train_loader.dataset.load_data()

if __name__=='__main__':
    main()








