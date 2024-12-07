import os
import random
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from ocpmodels.datasets import LmdbDataset
from ocpmodels.models.equiformer_v2.equiformer_v2_oc20_seq import EquiformerV2_OC20, EquiformerV2_seq
from tqdm import trange
from datetime import datetime
random.seed(114)


def set_para_requires_grad(model, feature_frozen):
    if feature_frozen:
        for para in model.parameters():
            para.requires_grad = False


def set_para_requires_grad_unfreeze(model, keys):
    for model_key in keys:
        for name, parameter in model.named_parameters():
            if model_key in name:
                parameter.requires_grad = True


def get_dataloader(data, batch_size, num_workers, shuffle: bool):
    batch = data
    random.shuffle(batch)
    split_idx = int(len(batch) * 0.9)
    train_data = batch[:split_idx]
    val_data = batch[split_idx:]
    dataloader_train = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    dataloader_valid = DataLoader(val_data,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return dataloader_train, dataloader_valid


def get_loader(path, num_workers, batch_size, shuffle: bool):
    batch_slab = []
    data_train = LmdbDataset({"src": path})
    for i in range(len(data_train) // 4):
        batch_data = [data_train[i * 4], data_train[i * 4 + 1], data_train[i * 4 + 2], data_train[i * 4 + 3]]
        batch_slab.append(batch_data)
    train_dataloader_slab, valid_dataloader_slab = get_dataloader(batch_slab, batch_size, num_workers, shuffle)
    return train_dataloader_slab, valid_dataloader_slab


def train(model, data_loader, seq_optimizer, num_epochs, seq_criterion, seq_normalizers, seq_device, save_best: bool):
    train_losses = []
    valid_losses = []
    best_loss = 1000000.0
    train_num = 1
    model = model.to(seq_device)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d-%H-%M")
    if not os.path.exists(f'train_seq/{date_str}'):
        os.makedirs(f'train_seq/{date_str}')
    for epoch in trange(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                with open(f'train_seq/{date_str}/record{epoch}.txt', 'a') as f:
                    f.write(f'train_epoch: {epoch} start \n')
                model.train()
            else:
                with open(f'train_seq/{date_str}/record{epoch}.txt', 'a') as f:
                    f.write(f'vaild:  start \n')
                model.eval()
            running_loss = 0.0
            num_samples = 0
            for data in data_loader[phase]:
                optimizer.zero_grad()
                data4 = [data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)]
                energy = model(data4)
                energy_hat = (energy * seq_normalizers["target"]["std"] + seq_normalizers["target"]["mean"])
                if phase == 'train':
                    with open(f'train_seq/{date_str}/record{epoch}.txt', 'a') as f:
                        f.write(f'Train Energy: {energy_hat[0].item()}, {energy_hat[1].item()}, '
                                f'{energy_hat[2].item()}, {energy_hat[3].item()}'
                                f' Y_relaxed:{data4[0].y_relaxed.item()}, {data4[1].y_relaxed.item()}, '
                                f'{data4[2].y_relaxed.item()}, {data4[3].y_relaxed.item()}\n')
                if phase == 'valid':
                    with open(f'train_seq/{date_str}/record{epoch}.txt', 'a') as f:
                        f.write(f'Valid Energy: {energy_hat[0].item()}, {energy_hat[1].item()}, '
                                f'{energy_hat[2].item()}, {energy_hat[3].item()}'
                                f' Y_relaxed:{data4[0].y_relaxed.item()}, {data4[1].y_relaxed.item()}, '
                                f'{data4[2].y_relaxed.item()}, {data4[3].y_relaxed.item()}\n')
                energy_target = torch.stack((data4[0].y_relaxed, data4[1].y_relaxed, data4[2].y_relaxed,
                                             data4[3].y_relaxed))
                energy_target = (
                        (energy_target.detach() - seq_normalizers["target"]["mean"]) / seq_normalizers["target"]["std"]
                ).to(device)
                weights = torch.tensor([1, 1, 1, 1], dtype=torch.float32).to(device)
                loss = (seq_criterion(energy, energy_target) * weights).mean()
                if phase == 'train':
                    loss.backward()
                    seq_optimizer.step()
                running_loss += loss.item() * len(data4)
                num_samples += len(data4)
                #print("{}_num = {}, loss = {} ".format(phase, train_num, running_loss))
                with open(f'train_seq/{date_str}/record{epoch}loss.txt', 'a') as f:
                    f.write(f'{phase}  {train_num}, Epoch Loss: {running_loss}\n')
                train_num += 1
            epoch_loss = running_loss / num_samples
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                if save_best is True:
                    state = {
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer': seq_optimizer.state_dict(),
                        'normalizers': seq_normalizers
                    }
                    filename = f'../checkpoint/Seq_Equiformer_{date_str}.pt'
                    torch.save(state, filename)

            print(f"{phase}  Epoch [{epoch + 1}/{num_epochs}], Epoch Loss: {epoch_loss}")

            if phase == 'valid':
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_losses.append(epoch_loss)

    if save_best is True:
        print("Train_loss: ", train_losses)
        with open(f'train_seq/{date_str}/train_losses', 'w') as file:
            for item in train_losses:
                file.write(str(item) + '\n')

        print("Valid_losses: ", valid_losses)
        with open(f'train_seq/{date_str}/valid_losses', 'w') as file:
            for item in valid_losses:
                file.write(str(item) + '\n')

        print("Best_loss: ", best_loss, "\t", valid_losses.index(best_loss))
        os.makedirs(f'train_seq/{date_str}/{str(best_loss)}-{str(valid_losses.index(best_loss))}')
    else:
        print("Best_loss: ", best_loss)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = {
        "use_pbc": True,
        "regress_forces": True,
        "otf_graph": True,
        "enforce_max_neighbors_strictly": False,
        "max_neighbors": 20,
        "max_radius": 12.0,
        "max_num_elements": 90,
        "num_layers": 18,
        "sphere_channels": 128,
        "attn_hidden_channels": 64,
        # [64, 96] This determines the hidden size of message passing. Do not necessarily use 96.
        "num_heads": 8,
        "attn_alpha_channels": 64,  # Not used when `use_s2_act_attn` is True.
        "attn_value_channels": 16,
        "ffn_hidden_channels": 128,
        "norm_type": "layer_norm_sh",  # ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
        "lmax_list": [6],
        "mmax_list": [2],
        "grid_resolution": 18,  # [18, 16, 14, None] For `None`, simply comment this line.
        "num_sphere_samples": 128,
        "edge_channels": 128,
        "use_atom_edge_embedding": True,
        "share_atom_edge_embedding": False,
        # If `True`, `use_atom_edge_embedding` must be `True` and the atom edge embedding will be shared across all blocks.
        "use_m_share_rad": False,
        "distance_function": "gaussian",
        "num_distance_basis": 512,  # not used
        "attn_activation": "silu",
        "use_s2_act_attn": False,
        # [False, True] Switch between attention after S2 activation or the original EquiformerV1 attention.
        "use_attn_renorm": True,  # Attention re-normalization. Used for ablation study.
        "ffn_activation": "silu",  # ['silu', 'swiglu']
        "use_gate_act": False,  # [True, False] Switch between gate activation and S2 activation
        "use_grid_mlp": True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        "use_sep_s2_act": True,  # Separable S2 activation. Used for ablation study.
        "alpha_drop": 0.1,  # [0.0, 0.1]
        "drop_path_rate": 0.1,  # [0.0, 0.05]
        "proj_drop": 0.0,
        "weight_init": "uniform",  # ['uniform', 'normal']
        "load_energy_lin_ref": True,
        # Set to `True` for the test set or when loading a checkpoint that has `energy_lin_ref` parameters,
        # `False` for training and val.
        "use_energy_lin_ref": True,  # Set to `True` for the test set, `False` for training and val.
    }

    equiformer = EquiformerV2_OC20(num_atoms=-1, bond_feat_dim=-1, num_targets=-1, **model_config)
    equiformer.regress_forces = False
    checkpoint_path = '../checkpoint/eq2_121M_e4_f100_oc22_s2ef.pt'
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt["state_dict"]
    state_dict = {k[2 * len("module."):]: v for k, v in state_dict.items()}
    normalizers = ckpt.get("normalizers")
    equiformer.load_state_dict(state_dict)
    model_seq = EquiformerV2_seq(equiformer, 256, dropout_lstm=0.07582982402684453,
                                 dropout_out=0.09432734725379567)
    set_para_requires_grad(model_seq, True)
    set_para_requires_grad_unfreeze(model_seq, ['lstm', 'lstm2input'])
    dir_path = "../Seq-Equiformer/data/demo.lmdb"
    train_dataloader, valid_dataloader = get_loader(dir_path, 1, 1, False)
    dataloader = {'train': train_dataloader, 'valid': valid_dataloader}
    params = [parameter for parameter in model_seq.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': 0.001564290299277057}])
    criterion = nn.MSELoss()
    num_epoch = 100
    train(model_seq, dataloader, optimizer, num_epoch, criterion, normalizers, device, True)
