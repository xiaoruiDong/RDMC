from abc import abstractmethod
import pickle
from typing import List

import numpy as np

from rdmc import Mol
from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.verifiers.base import FreqVerifier


class TSFreqVerifier(FreqVerifier):

    allowed_number_negative_frequencies = 1
    default_cutoff_frequency = -10.0


class IRCVerifier(BaseTask):

    path_prefix = "irc"

    def run(
        self,
        ts_mol,
        rxn_smiles: str,
        **kwargs,
    ):
        """
        Verifying TS guesses or optimized TS geometries.

        Args:
            ts_mol (RDKitMol): The TS in RDKitMol object with 3D geometries embedded.
            rxn_smiles (str): The reaction Smiles.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        for i in range(ts_mol.GetNumConformers()):
            if not ts_mol.KeepIDs[i]:
                # Bypass previously discarded conformers
                continue

            try:
                irc_adj_mats = self.run_irc(
                    ts_mol,
                    conf_id=i,
                    **kwargs,
                )
            except BaseException as e:
                print("IRC verification fails")
                print(e)
                ts_mol.KeepIDs[i] = False
                continue

            r_smi, p_smi = rxn_smiles.split(">>")
            r_adj = Mol.FromSmiles(r_smi).GetAdjacencyMatrix()
            p_adj = Mol.FromSmiles(p_smi).GetAdjacencyMatrix()

            ts_mol.keepIDs[i] = check_adjacency_matrixes_pairs(
                (r_adj, p_adj),
                irc_adj_mats,
            )

        if self.save_dir:
            with open(self.save_dir / "irc_check_ids.pkl", "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol

    @abstractmethod
    def run_irc(
        self,
        ts_mol,
        conf_id: int,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Run IRC.

        Args:
            ts_mol (RDKitMol): The TS in RDKitMol object with 3D geometries embedded.

        Returns:
            List[np.ndarray]: The forward and backward adjacency matrixes.
        """
        raise NotImplementedError


def check_adjacency_matrixes_pairs(
    adj_mats1: List[np.ndarray],
    adj_mats2: List[np.ndarray],
):
    """
    Check if two adjacency matrix pairs are the equivalent.

    Args:
        adj_mats1 (List[np.ndarray]): The forward and backward adjacency matrixes.
        adj_mats2 (List[np.ndarray]): The forward and backward adjacency matrixes.

    Returns:
        bool: True if two adjacency matrix pairs are the equivalent.
    """
    if len(adj_mats1) != len(adj_mats2):
        return False

    f_adj, b_adj = adj_mats1
    r_adj, p_adj = adj_mats2

    # check one direction
    try:
        if np.array_equal(f_adj, r_adj) and np.array_equal(b_adj, p_adj):
            return True
        elif np.array_equal(f_adj, p_adj) and np.array_equal(b_adj, r_adj):
            return True
        return False
    except ValueError:
        print(
            "The reactant/products have different numbers of atoms"
            " than the TS unexpectedly"
        )  # change it to log
        return False
