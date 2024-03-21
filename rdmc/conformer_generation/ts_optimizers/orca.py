import os
import subprocess
from typing import Optional

import numpy as np

from rdmc import RDKitMol
from rdmc.conformer_generation.ts_optimizers.base import TSOptimizer
from rdmc.external.inpwriter import write_orca_opt


class OrcaOptimizer(TSOptimizer):
    """
    The class to optimize TS geometries using the Berny algorithm built in Orca.
    You have to have the Orca package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                Defaults to ``"XTB2"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self,
        method: str = "XTB2",
        nprocs: int = 1,
        track_stats: bool = False,
    ):
        """
        Initiate the Orca berny optimizer.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                    If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                    Defaults to ``"XTB2"``.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(OrcaOptimizer, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs

        ORCA_BINARY = os.environ.get("ORCA")
        if not ORCA_BINARY:
            raise RuntimeError("No Orca binary is found in the PATH.")
        else:
            self.orca_binary = ORCA_BINARY

    def extract_frequencies(self, save_dir: str, n_atoms: int):
        """
        Extract frequencies from the Orca opt job.

        Args:
            save_dir (str): Path where Orca logs are saved.
            n_atoms (int): The number of atoms in the molecule.

        Returns:
            np.ndarray: The frequencies in cm-1.
        """

        log_file = os.path.join(save_dir, "orca_opt.log")
        with open(log_file, "r") as f:
            orca_data = f.readlines()

        dof = 3 * n_atoms
        orca_data.reverse()
        freq_idx = None
        for i, line in enumerate(orca_data):
            if "VIBRATIONAL FREQUENCIES" in line:
                freq_idx = i
                break
        if freq_idx:
            freqs = orca_data[freq_idx - 4 - dof : freq_idx - 4]
            freqs.reverse()
            return np.array([float(line.split()[1]) for line in freqs])
        else:
            return None

    def optimize_ts_guesses(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Optimize the TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            multiplicity (int): The multiplicity of the molecule. Defaults to ``1``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        opt_mol = mol.Copy(quickCopy=True, copy_attrs=["KeepIDs"])
        opt_mol.energy = {}  # TODO: add orca energies
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}
        for i in range(mol.GetNumConformers()):

            if not opt_mol.KeepIDs[i]:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                continue

            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"orca_opt{i}")
                os.makedirs(ts_conf_dir, exist_ok=True)

            # Generate and save the ORCA input file
            orca_str = write_orca_opt(
                mol,
                conf_id=i,
                ts=True,
                method=self.method,
                mult=multiplicity,
                nprocs=self.nprocs,
            )

            orca_input_file = os.path.join(ts_conf_dir, "orca_opt.inp")
            with open(orca_input_file, "w") as f:
                f.writelines(orca_str)

            # Run the optimization using subprocess
            with open(os.path.join(ts_conf_dir, "orca_opt.log"), "w") as f:
                orca_run = subprocess.run(
                    [self.orca_binary, orca_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )

            # Check the Orca results
            if orca_run.returncode == 0:
                try:
                    new_mol = RDKitMol.FromFile(
                        os.path.join(ts_conf_dir, "orca_opt.xyz"), sanitize=False
                    )
                    opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                    opt_mol.frequency.update(
                        {
                            i: self.extract_frequencies(
                                ts_conf_dir, opt_mol.GetNumAtoms()
                            )
                        }
                    )
                except Exception as e:
                    opt_mol.AddNullConformer(confId=i)
                    opt_mol.energy.update({i: np.nan})
                    opt_mol.KeepIDs[i] = False
                    print(f"Cannot read Orca output, got: {e}")
            else:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                opt_mol.KeepIDs[i] = False

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol, opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol
