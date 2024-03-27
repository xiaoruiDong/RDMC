import os
import pickle
from time import time
from typing import Optional
import subprocess

import numpy as np
from rdkit import Chem

from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.task.gaussian import GaussianTask

from rdmc.external.logparser import GaussianLog
from rdmc.external.inpwriter import write_gaussian_opt


class GaussianOptimizer(GaussianTask, ConfGenOptimizer):
    """
    Optimizer using the Gaussian.

    Args:
        method (str, optional): The method to be used for species optimization. You can use the level of theory available in Gaussian.
                                Defaults to ``"GFN2-xTB"``, which is realized by additional scripts provided in the ``rdmc`` package.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def run(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> "RDKitMol":
        """
        Optimize the conformers.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            multiplicity (int): The multiplicity of the molecule. Defaults to ``1``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The optimized molecule as RDKitMol with 3D geometries embedded.
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
                conf_dir = os.path.join(save_dir, f"gaussian_opt{i}")
                os.makedirs(conf_dir, exist_ok=True)

            # Generate and save the gaussian input file
            gaussian_str = write_gaussian_opt(
                mol,
                conf_id=i,
                method=self.method,
                mult=multiplicity,
                nprocs=self.nprocs,
                memory=self.memory,
            )
            gaussian_input_file = os.path.join(conf_dir, "gaussian_opt.gjf")
            with open(gaussian_input_file, "w") as f:
                f.writelines(gaussian_str)

            # Run the gaussian via subprocess
            with open(os.path.join(conf_dir, "gaussian_opt.log"), "w") as f:
                gaussian_run = subprocess.run(
                    [self.binary_path, gaussian_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            # Check the output of the gaussian
            if gaussian_run.returncode == 0:
                try:
                    g16_log = GaussianLog(os.path.join(conf_dir, "gaussian_opt.log"))
                    pre_adj_mat = mol.GetAdjacencyMatrix()
                    post_adj_mat = g16_log.get_mol(
                        refid=g16_log.num_all_geoms - 1,  # The last geometry in the job
                        converged=False,
                        sanitize=False,
                        backend="openbabel",
                    ).GetAdjacencyMatrix()
                    if g16_log.success and (pre_adj_mat == post_adj_mat).all():
                        new_mol = g16_log.get_mol(embed_conformers=True, sanitize=False)
                        opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                        opt_mol.energy.update(
                            {i: g16_log.get_scf_energies(relative=False)[-1]}
                        )
                        opt_mol.frequency.update({i: g16_log.freqs})
                    else:
                        opt_mol.AddNullConformer(confId=i)
                        opt_mol.energy.update({i: np.nan})
                        opt_mol.KeepIDs[i] = False
                        print(
                            "Error! Likely that the smiles doesn't correspond to this species."
                        )
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

    def save_opt_mols(
        self,
        save_dir: str,
        opt_mol: "RDKitMol",
        keep_ids: dict,
        energies: dict,
    ):
        """
        Save the information of the optimized stable species into the directory.

        Args:
            save_dir (str): The path to the directory to save the results.
            opt_mol (RDKitMol): The optimized stable species in RDKitMol with 3D conformer saved with the molecule.
            keep_ids (dict): Dictionary of which opts succeeded and which failed.
            energies (dict): Dictionary of energies for each conformer.
        """
        # Save optimized stable species mols
        path = os.path.join(save_dir, "optimized_confs.sdf")
        try:
            writer = Chem.rdmolfiles.SDWriter(path)
            for i in range(opt_mol.GetNumConformers()):
                opt_mol.SetProp("Energy", str(energies[i]))
                writer.write(opt_mol, confId=i)
        except Exception:
            raise
        finally:
            writer.close()

        # save ids
        with open(os.path.join(save_dir, "opt_check_ids.pkl"), "wb") as f:
            pickle.dump(keep_ids, f)
