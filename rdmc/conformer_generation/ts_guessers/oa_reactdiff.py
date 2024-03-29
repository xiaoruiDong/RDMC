import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from rdmc.conformer_generation.comp_env.software import try_import
from rdmc.conformer_generation.comp_env.torch import F, torch
from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser
from rdtools.mol import get_atomic_nums


package_name = "OA-ReactDiff"
namespace = globals()

modules = [
    "oa_reactdiff",
    "oa_reactdiff.trainer.pl_trainer.DDPMModule",
    "oa_reactdiff.dataset.base_dataset.ATOM_MAPPING",
    "oa_reactdiff.dataset.base_dataset.n_element",
    "oa_reactdiff.diffusion._schedule.DiffSchedule",
    "oa_reactdiff.diffusion._schedule.PredefinedNoiseSchedule",
    "oa_reactdiff.diffusion._normalizer.FEATURE_MAPPING",
]

for module in modules:
    try_import(module, namespace=namespace, package_name=package_name)


class OAReactDiffGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the OA-ReactDiff model. This is currently not working as expected as
    the atom order "changes" during the generation.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self,
        device: str = "cpu",
        track_stats: Optional[bool] = False,
    ):
        """
        Initialize the OAReactDiff guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super().__init__(track_stats)

        self.device = device

        self.model_path = (
            Path(oa_reactdiff.__file__).parents[1] / "pretrained-ts1x-diff.ckpt"
        )  # Hard-coded model path this is the only model we found so far

        ddpm_trainer = DDPMModule.load_from_checkpoint(
            checkpoint_path=self.model_path,
            map_location=self.device,
        )

        self.module = set_new_schedule(
            ddpm_trainer,
            timesteps=150,
            device=self.device,
            noise_schedule="polynomial_2",
        )  # according to the notebook

    def is_available(self):
        """
        Check whether the OA-ReactDiff model is available.

        Returns:
            bool: Whether the OA-ReactDiff model is available.
        """
        return package_available["OA-ReactDiff"]

    def generate_ts_guesses(self, mols: list, **kwargs):
        """
        Generate TS guesses.

        Args:
            mols (list): A list of reactant and product pairs.

        Returns:
            Tuple[dict, dict]: The generated guesses positions and the success status.
                The keys are the conformer IDs. The values are the positions and the success status.
        """

        batch = mols_to_batch(
            mols,
            device=self.device,
        )

        out_samples, xh_fixed, fragments_nodes = inplaint_batch(
            batch,
            self.module,
            resamplings=5,
            jump_length=5,
            frag_fixed=[0, 2],
        )  # according to the notebook
        pos, _, _ = samples_to_pos_charge(out_samples, fragments_nodes)

        print(out_samples)

        positions = {
            i: p.astype("float") for i, p in enumerate(pos["transition_state"])
        }
        successes = {i: True for i in range(len(positions))}

        return positions, successes


def mols_to_batch(mols, device: str = "cpu"):
    """
    A helper function for our embedder. This is greatly simplify compared to the original
    featurizer due to all the mol pairs corresponding to the same reaction.
    """

    n_samples = len(mols)
    dtype = torch.int64

    n_atoms = mols[0][0].GetNumAtoms()
    atomic_nums = get_atomic_nums(mols[0][0])

    r_poss, ts_poss, p_poss = [], [], []
    for rmol, pmol in mols:

        r_pos = torch.from_numpy(rmol.GetPositions().astype("float32"))
        r_pos = r_pos - torch.mean(r_pos, dim=0)
        p_pos = torch.from_numpy(pmol.GetPositions().astype("float32"))
        p_pos = p_pos - torch.mean(p_pos, dim=0)
        ts_pos = (r_pos + p_pos) / 2
        r_poss.append(r_pos)
        ts_poss.append(ts_pos)
        p_poss.append(p_pos)

    batch = []
    for poss in [r_poss, ts_poss, p_poss]:
        batch.append(
            {
                "size": torch.tensor([n_atoms] * n_samples, dtype=dtype, device=device),
                "pos": torch.concat(poss).to(device),
                "charge": torch.tensor(
                    atomic_nums * n_samples, dtype=dtype, device=device
                ).unsqueeze(1),
                "mask": torch.repeat_interleave(
                    torch.arange(n_samples, device=device),
                    n_atoms,
                ),
                "one_hot": torch.concat(
                    [
                        F.one_hot(
                            torch.tensor([ATOM_MAPPING[anum] for anum in atomic_nums]),
                            num_classes=n_element,
                        )
                        for _ in range(n_samples)
                    ]
                ).to(device),
            }
        )
    conditions = torch.zeros((n_samples, 1), dtype=dtype, device=device)

    return batch, conditions


# Following are copied from oa_reactdiff.evaluate.utils to avoid import of pyscf


def samples_to_pos_charge(out_samples, fragments_nodes):
    x_r = torch.tensor_split(
        out_samples[0], torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    x_ts = torch.tensor_split(
        out_samples[1], torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    x_p = torch.tensor_split(
        out_samples[2], torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    pos = {
        "reactant": [_x[:, :3].cpu().numpy() for _x in x_r],
        "transition_state": [_x[:, :3].cpu().numpy() for _x in x_ts],
        "product": [_x[:, :3].cpu().numpy() for _x in x_p],
    }
    z = [_x[:, -1].long().cpu().numpy() for _x in x_r]
    natoms = [f.cpu().item() for f in fragments_nodes[0]]
    return pos, z, natoms


def set_new_schedule(
    ddpm_trainer: "DDPMModule",
    timesteps: int = 250,
    device: torch.device = torch.device("cuda"),
    noise_schedule: str = "polynomial_2",
) -> DDPMModule:
    precision: float = 1e-5

    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=precision,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module, norm_values=ddpm_trainer.ddpm.norm_values
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    return ddpm_trainer.to(device)


def inplaint_batch(
    batch: List,
    ddpm_trainer: DDPMModule,
    resamplings: int = 1,
    jump_length: int = 1,
    frag_fixed: List = [0, 2],
):
    representations, conditions = batch
    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations
    ]
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [repre["size"] for repre in representations]
    out_samples, _ = ddpm_trainer.ddpm.inpaint(
        n_samples=n_samples,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=resamplings,
        jump_length=jump_length,
        timesteps=None,
        xh_fixed=xh_fixed,
        frag_fixed=frag_fixed,
    )
    return out_samples[0], xh_fixed, fragments_nodes
