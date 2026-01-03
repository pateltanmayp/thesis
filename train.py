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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl
import time
import random
import sys, ipdb, traceback, subprocess
from ICKANs.drivers.model_branching import BranchingConvexKAN
torch.set_default_dtype(torch.float32)

torch.autograd.set_detect_anomaly(True)

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
            cur_material_id = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "MaterialID.pt"), map_location='cpu') # shape: (num_particles,)

            if cur_C.shape[0] == 960:
                cur_Ftmp = torch.cat((cur_Ftmp, cur_Ftmp, cur_Ftmp[:80, :, :, :]), dim=0)
                cur_F = torch.cat((cur_F, cur_F, cur_F[:80, :, :, :]), dim=0)
                cur_stress = torch.cat((cur_stress, cur_stress, cur_stress[:80, :, :, :]), dim=0)
                cur_C = torch.cat((cur_C, cur_C, cur_C[:80, :, :, :]), dim=0)
            cur_traj_id = torch.full((cur_F.shape[0], cur_F.shape[1]), idx)

            # Repeat over timesteps
            material_ids = cur_material_id.unsqueeze(0).repeat(cur_F.shape[0], 1)
            material_ids = material_ids.reshape(-1)

            self.sim_timesteps = cur_Ftmp.shape[0]
            self.sim_num_particles = cur_Ftmp.shape[1]

            input_Ftmp_tensor = cur_Ftmp.reshape(-1, 3, 3)
            input_F_tensor = cur_F.reshape(-1, 3, 3)
            gt_stress_tensor = cur_stress.reshape(-1, 3, 3)
            input_C_tensor = cur_C.reshape(-1, 3, 3)
            traj_ids = cur_traj_id.reshape(-1)
            self.traj_id_dict[idx] = f"{file_dir}"

            self.all_data_dict[idx] = {
                'input_F': input_F_tensor,
                'gt_stress': gt_stress_tensor,
                'input_C': input_C_tensor,
                'material_ids': material_ids,
                'traj_ids': traj_ids,
            }

        print ("Loaded data in: ", time.time() - start_time, " seconds.")

    def __len__(self):
        return len(self.file_dir_list) * self.sim_num_particles * self.sim_timesteps

    def __getitem__(self, idx):
        traj_idx = idx // (self.sim_num_particles * self.sim_timesteps)
        sample_idx = idx % (self.sim_num_particles * self.sim_timesteps)

        sample = {
            'input_F': self.all_data_dict[traj_idx]['input_F'][sample_idx],
            'stress_target': self.all_data_dict[traj_idx]['gt_stress'][sample_idx],
            'input_C': self.all_data_dict[traj_idx]['input_C'][sample_idx],
            'material_ids': self.all_data_dict[traj_idx]['material_ids'][sample_idx],
        }
        return sample

@hydra.main(config_path='UniPhy/configs', config_name='default')
def main(cfg: omegaconf.DictConfig):

    # Environment and logging setup
    save_dir = cfg['train_cfg']['save_dir']
    local_dir = cfg['train_cfg']['local_dir']
    logger = logging.getLogger()

    os.makedirs(f"{local_dir}/{save_dir}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_dir}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_dir}/log.txt")
    with open(f"{local_dir}/{save_dir}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Load dataset
    traj_l = ["elastic_diverse", "non_newtonian_diverse", "plasticine_diverse", "sand_diverse"] # Should have "newtonian_diverse" after "elastic_diverse"
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

    # Model setup

    # Define latent
    z_dim = cfg['train_cfg']['z_dim']
    num_materials = cfg['train_cfg']['num_materials']
    material_latent = torch.nn.Embedding(num_materials, z_dim).cuda()
    torch.nn.init.normal_(material_latent.weight, mean=0.0, std=0.1)
    traj_mean, traj_std = 0., 1.
    torch.nn.init.normal_(material_latent.weight, mean=traj_mean, std=traj_std)

    # Define model
    n_hidden = OmegaConf.to_container(cfg['train_cfg']['n_hidden'], resolve=True)
    grid_range = OmegaConf.to_container(cfg['train_cfg']['grid_range'], resolve=True)
    
    model = BranchingConvexKAN(
        n_hidden=n_hidden,
        grid_range=grid_range,
        z_dim=z_dim,
        seed=0
    ).cuda()
    
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg['train_cfg']['lr']},
        {"params": material_latent.parameters(), "lr": cfg['train_cfg']['lr'] * 10}
    ])

    total_epochs = 10
    step_lr_step_size=cfg['train_cfg']['step_lr_step_size']
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr_step_size, gamma=0.9)

    loss_mse = nn.MSELoss()
    loss_mse_fproj = nn.MSELoss()
    for epoch in range(total_epochs):
        print ("Epoch: ", epoch)
        for idx, data in enumerate(train_loader):
            # input_F, stress_target, traj_ids, input_Ftmp, input_C = data['input_F'].cuda(), data['stress_target'].cuda(), data['traj_ids'].cuda(), data['input_Ftmp'].cuda(), data['input_C'].cuda()

            optimizer.zero_grad()

            input_F = data['input_F'].cuda()
            stress_target = data['stress_target'].cuda()

            # Should do this based on segmentation of scene
            material_ids = data['material_ids'].cuda()

            # Flatten F to match your KAN interface
            F_flat = input_F[:, :3, :3].reshape(-1, 9)  # if using 3D
            F_flat = F_flat.clone().requires_grad_(True).cuda()
            z = material_latent(material_ids).cuda()

            F_flat = F_flat.float()
            z = z.float()

            # Energy prediction
            W = model(F_flat, z)

            # Stress via autodiff
            P = torch.autograd.grad(
                W.sum(), F_flat, create_graph=True
            )[0]

            # Compare to target stress (mapped consistently)
            stress_target_flat = stress_target[:, :3, :3].reshape(-1, 9).cuda()  # if using 3D
            loss = loss_mse(P, stress_target_flat)

            loss_reg = 1e-4 * torch.mean(torch.norm(material_latent.weight, dim=-1))
            loss = loss + loss_reg

            loss.backward()
            optimizer.step()

            logger.info(f"Epoch: {epoch}, Iter/Total: {idx}/{len(train_loader)}, Loss: {loss}, LR1: {scheduler1.get_last_lr()}")
            if idx % 1000 == 0:
                writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + idx)
                writer.add_scalar('Regression Loss/train', loss_reg, epoch * len(train_loader) + idx)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + idx)

            if epoch < 2:
                scheduler1.step()

        if epoch % 1 == 0:
            torch.save(material_latent, f"{local_dir}/{save_dir}/material_latent_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{local_dir}/{save_dir}/model_{epoch}.pth")

        train_loader.dataset.load_data()

if __name__=='__main__':
    main()
