import os
import subprocess
from typing import Optional

import numpy as np

from rdmc.conformer_generation.ts_optimizers.base import TSOptimizer
from rdmc.conformer_generation.task.qchem import QChemTask
from rdmc.external.inpwriter import write_qchem_opt
from rdmc.external.logparser import QChemLog


class QChemOptimizer(QChemTask, TSOptimizer):
    """
    The class to optimize TS geometries using the Baker's eigenvector-following (EF) algorithm built in QChem.
    You have to have the QChem package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the method available in QChem.
                                Defaults to ``"wB97x-d3"``.
        basis (str, optional): The method to be used for TS optimization. you can use the basis available in QChem.
                                Defaults to ``"def2-tzvp"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
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
                opt_mol.KeepIDs[i] = False
                continue

            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"qchem_opt{i}")
                os.makedirs(ts_conf_dir, exist_ok=True)

            # Generate and save the qchem input file
            qchem_str = write_qchem_opt(
                mol,
                conf_id=i,
                ts=True,
                method=self.method,
                basis=self.basis,
                mult=multiplicity,
            )

            qchem_input_file = os.path.join(ts_conf_dir, "qchem_opt.qcin")
            with open(qchem_input_file, "w") as f:
                f.writelines(qchem_str)

            # Run the qchem via subprocess
            with open(os.path.join(ts_conf_dir, "qchem_opt.log"), "w") as f:
                qchem_run = subprocess.run(
                    [self.binary_path, "-nt", str(self.nprocs), qchem_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            # Check the output of the qchem
            if qchem_run.returncode == 0:
                try:
                    qchem_log = QChemLog(os.path.join(ts_conf_dir, "qchem_opt.log"))
                    if qchem_log.success:
                        new_mol = qchem_log.get_mol(
                            embed_conformers=False, sanitize=False
                        )
                        opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                        opt_mol.energy.update(
                            {i: qchem_log.get_scf_energies(relative=False)[-1]}
                        )
                        opt_mol.frequency.update({i: qchem_log.freqs})
                except Exception as e:
                    opt_mol.AddNullConformer(confId=i)
                    opt_mol.energy.update({i: np.nan})
                    opt_mol.KeepIDs[i] = False
                    print(f"Got an error when reading the QChem output: {e}")
            else:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                opt_mol.KeepIDs[i] = False

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol, opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol
