from rdkit.Chem import AllChem
from rdmc.mol import RDKitMol
import torch
import numpy as np
from copy import deepcopy


def hard_sphere_loss_fn(data, predicted_ts_coords):
    hard_sphere_radii = torch.tensor([0, 0.37 / 1.2, 0, 0, 0, 0, 0.69 / 1.2, 0.71 / 2, 0.66 / 1.2],
                                     dtype=torch.float32, device=predicted_ts_coords.device)
    start, end = data.edge_index
    distances = torch.norm(predicted_ts_coords[start] - predicted_ts_coords[end], dim=-1)
    min_dist = hard_sphere_radii[data.z[start].long()] + hard_sphere_radii[data.z[end].long()]

    hard_sphere_loss = torch.square(min_dist - distances)
    hard_sphere_mask = min_dist < distances
    hard_sphere_loss[hard_sphere_mask] = hard_sphere_loss[hard_sphere_mask] * 0

    return hard_sphere_loss.sum() / (~hard_sphere_mask).float().sum().clamp(min=1e-8)


def eval_stats(data, predicted_ts_coords):
    n_curr = 0
    maes, rmses, rmsds = [], [], []
    for mols in data.mols:
        true_ts = RDKitMol.FromMol(mols[1])
        new_ts = deepcopy(true_ts)
        n_atoms = new_ts.GetNumAtoms()
        new_ts_coords = predicted_ts_coords[n_curr:n_curr + n_atoms].cpu().detach().numpy()
        new_ts.SetPositions(np.array(new_ts_coords, dtype=float))
        n_curr = n_curr + n_atoms

        rmsd = AllChem.GetBestRMS(true_ts.ToRWMol(), new_ts.ToRWMol())
        rmsds.append(rmsd)

        mae = np.abs(true_ts.GetDistanceMatrix() - new_ts.GetDistanceMatrix()).sum() / (n_atoms ** 2 - n_atoms)
        maes.append(mae)

        rmse = np.square(true_ts.GetDistanceMatrix() - new_ts.GetDistanceMatrix()).sum() / (n_atoms ** 2 - n_atoms)
        rmses.append(rmse)

    return maes, rmses, rmsds


def radius_graph(pos, cutoff, edge_index, edge_attr):
    row, col = edge_index
    dist = (pos[row] - pos[col]).norm(dim=-1)
    mask = dist < cutoff
    return edge_index[:, mask], edge_attr[mask]
