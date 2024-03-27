import os
import subprocess
from typing import Optional

import numpy as np

from rdmc.conformer_generation.task.gaussian_task import GaussianTask
from rdmc.conformer_generation.ts_optimizers.base import TSOptimizer
from rdmc.external.inpwriter import write_gaussian_opt
from rdmc.external.logparser import GaussianLog


class GaussianOptimizer(GaussianTask, TSOptimizer):
    """
    The class to optimize TS geometries using the Berny algorithm built in Gaussian.
    You have to have the Gaussian package installed to run this optimizer

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do.
                                Defaults to ``"GFN2-xTB"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

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
        opt_mol.energy = {}
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}
        for i in range(mol.GetNumConformers()):

            if not opt_mol.KeepIDs[i]:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                continue

            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"gaussian_opt{i}")
                os.makedirs(ts_conf_dir, exist_ok=True)

            # Generate and save the gaussian input file
            gaussian_str = write_gaussian_opt(
                mol,
                conf_id=i,
                ts=True,
                method=self.method,
                mult=multiplicity,
                nprocs=self.nprocs,
                memory=self.memory,
            )
            gaussian_input_file = os.path.join(ts_conf_dir, "gaussian_opt.gjf")
            with open(gaussian_input_file, "w") as f:
                f.writelines(gaussian_str)

            # Run the gaussian via subprocess
            with open(os.path.join(ts_conf_dir, "gaussian_opt.log"), "w") as f:
                gaussian_run = subprocess.run(
                    [self.binary_path, gaussian_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            # Check the output of the gaussian
            if gaussian_run.returncode == 0:
                try:
                    g16_log = GaussianLog(os.path.join(ts_conf_dir, "gaussian_opt.log"))
                    if g16_log.success:
                        new_mol = g16_log.get_mol(
                            embed_conformers=False, sanitize=False
                        )
                        opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                        opt_mol.energy.update(
                            {i: g16_log.get_scf_energies(relative=False)[-1]}
                        )
                        opt_mol.frequency.update({i: g16_log.freqs})
                except Exception as e:
                    opt_mol.AddNullConformer(confId=i)
                    opt_mol.energy.update({i: np.nan})
                    opt_mol.KeepIDs[i] = False
                    print(f"Got an error when reading the Gaussian output: {e}")
            else:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                opt_mol.KeepIDs[i] = False

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol, opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol
