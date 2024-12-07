import os
import random
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from ocpmodels.datasets import LmdbDataset
from ocpmodels.models.equiformer_v2.equiformer_v2_oc20_seq import EquiformerV2_OC20, EquiformerV2_seq
from tqdm import trange, tqdm
from datetime import datetime
random.seed(114)


def set_para_requires_grad(model, feature_frozen):
    if feature_frozen:
        for para in model.parameters():
            para.requires_grad = False


def get_test_dataloader(data, batch_size, num_workers, shuffle: bool):
    batch = data
    tset_dataloader = DataLoader(batch,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return tset_dataloader


def get_loader(path, num_workers, batch_size, shuffle: bool):
    batch_slab = []
    data_train = LmdbDataset({"src": path})
    for i in range(len(data_train) // 4):
        batch_data = [data_train[i * 4], data_train[i * 4 + 1], data_train[i * 4 + 2], data_train[i * 4 + 3]]
        batch_slab.append(batch_data)
    test_dataloader_slab = get_test_dataloader(batch_slab, batch_size, num_workers, shuffle)
    return test_dataloader_slab


def test(model, data_loader, normalizers, device):
    model = model.to(device)
    log = []
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d-%H-%M")
    if not os.path.exists(f'test_seq/{date_str}'):
        os.makedirs(f'test_seq/{date_str}')
    with open(f'test_seq/{date_str}/record.txt', 'a') as f:
        f.write(f'test:  start \n')
    model.eval()
    for data in tqdm(data_loader):
        data4 = [data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)]
        energy = model(data4)
        energy_hat = (energy * normalizers["target"]["std"] + normalizers["target"]["mean"])
        slab = energy_hat[0].item() + 2 * (-14.22)
        OH = energy_hat[1].item() + 0.35 + (-14.22) + 0.5 * (-6.82)
        O = energy_hat[2].item() + 0.04 + (-14.22) + (-6.82)
        OOH = energy_hat[3].item() + 0.35 + 1.5 * (-6.82)
        DOH = OH - slab
        DO = O - OH
        DOOH = OOH - O
        DO2 = slab + 4.92 - OOH
        MAX = max(DOH, DO, DOOH, DO2)
        overpotential = MAX - 1.23
        with open(f'test_seq/{date_str}/record.txt', 'a') as f:
            f.write(f'{energy_hat[0].item()}, {energy_hat[1].item()}, '
                    f'{energy_hat[2].item()}, {energy_hat[3].item()}, {slab}, {OH}, {O}, {OOH}, {DOH}, {DO}, {DOOH},'
                    f'{DO2}, {MAX}, {overpotential}\n')
        log.append([energy_hat[0].item(), energy_hat[1].item(), energy_hat[2].item(),
                    energy_hat[3].item(), slab, OH, O, OOH, DOH, DO, DOOH, DO2, MAX, overpotential])

    data_log = pd.DataFrame(data=log, columns=['predict_slab', 'predict_OH', 'predict_O', 'predict_OOH',
                                               'slab', 'OH', 'O', 'OOH', 'DOH', 'DO', 'DOOH', 'DO2', 'MAX',
                                               'overpotential'])
    data_log.to_csv(f'test_seq/{date_str}/log.csv')


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
    model_seq = EquiformerV2_seq(equiformer, 256, dropout_lstm=0.07582982402684453,
                                 dropout_out=0.09432734725379567)
    checkpoint_path = f'../checkpoint/Seq_Equiformer.pt'
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt["state_dict"]
    model_seq.load_state_dict(state_dict)
    set_para_requires_grad(model_seq, True)
    checkpoint_path_equiformer = "../checkpoint/eq2_121M_e4_f100_oc22_s2ef.pt"
    ckpt_equiformer = torch.load(checkpoint_path_equiformer)
    normalizers = ckpt_equiformer.get("normalizers")
    dir_path = "../Seq-Equiformer/data/demo.lmdb"
    dataloader_test = get_loader(dir_path, 1, 1, False)
    test(model_seq, dataloader_test, normalizers, device)
