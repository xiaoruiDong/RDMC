#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for optimizing initial guess geometries.
"""

from rdmc.forcefield import RDKitFF
from time import time
import numpy as np
import os
import subprocess
from typing import List, Tuple, Optional, Union

from rdkit import Chem
from rdmc.conformer_generation.utils import *
from rdmc.external.logparser import GaussianLog
from rdmc.external.inpwriter import write_gaussian_opt
try:
    from rdmc.external.xtb_tools.opt import run_xtb_calc
except ImportError:
    pass


class ConfGenOptimizer:
    """
    Base class for the geometry optimizers used in conformer generation.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    def __init__(self, track_stats=False):

        self.iter = 0
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_conformers(self,
                            mol_data: List[dict]):
        """
        Optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Raises:
            NotImplementedError: This function should be implemented in the child class.
        """
        raise NotImplementedError

    def __call__(self,
                 mol_data: List[dict],
                 ) -> List[dict]:
        """
        Run the workflow to optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Returns:
            List[dict]: The list of optimized conformers.
        """

        self.iter += 1
        time_start = time()
        mol_data = self.optimize_conformers(mol_data)

        if not self.track_stats:
            return mol_data

        time_end = time()
        stats = {"iter": self.iter,
                 "time": time_end - time_start,
                 "n_failures": self.n_failures,
                 "percent_failures": self.percent_failures,
                 "n_opt_cycles": self.n_opt_cycles}
        self.stats.append(stats)
        return mol_data


class MMFFOptimizer(ConfGenOptimizer):
    """
    Optimizer using the MMFF force field.

    Args:
        method (str, optional): The method to be used for stable species optimization. Defaults to ``"rdkit"``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    def __init__(self,
                 method: str = "rdkit",
                 track_stats: bool = False):
        super(MMFFOptimizer, self).__init__(track_stats)
        if method == "rdkit":
            self.ff = RDKitFF()
        elif method == "openbabel":
            raise NotImplementedError

    def optimize_conformers(self,
                            mol_data: List[dict],
                            ) -> List[dict]:
        """
        Optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Returns:
            List[dict]: The list of optimized conformers sorted by energy.
        """
        if len(mol_data) == 0:
            return mol_data

        # Everytime calling dict_to_mol create a new molecule object
        # No need to Copy the molecule object in this function
        mol = dict_to_mol(mol_data)
        self.ff.setup(mol)
        results = self.ff.optimize_confs()
        _, energies = zip(*results)  # kcal/mol
        opt_mol = self.ff.get_optimized_mol()

        for c_id, energy in zip(range(len(mol_data)), energies):
            conf = opt_mol.GetConformer(c_id)
            positions = conf.GetPositions()
            mol_data[c_id].update({"positions": positions,  # issues if not all opts succeeded?
                                   "conf": conf,  # all confs share the same owning molecule `opt_mol`
                                   "energy": energy})

        if self.track_stats:
            self.n_failures = np.sum([r[0] == 1 for r in results])
            self.percent_failures = self.n_failures / len(mol_data) * 100

        return sorted(mol_data, key=lambda x: x["energy"])


class XTBOptimizer(ConfGenOptimizer):
    """
    Optimizer using the xTB.

    Args:
        method (str, optional): The method to be used for species optimization. Defaults to ``"gff"``.
        level (str, optional): The level of theory. Defaults to ``"normal"``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    def __init__(self,
                 method: str = "gff",
                 level: str = "normal",
                 track_stats: bool = False):
        super(XTBOptimizer, self).__init__(track_stats)
        self.method = method
        self.level = level

    def optimize_conformers(self,
                            mol_data: List[dict],
                            ) -> List[dict]:
        """
        Optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Returns:
            List[dict]: The list of optimized conformers sorted by energy.
        """
        if len(mol_data) == 0:
            return mol_data

        new_mol = dict_to_mol(mol_data)
        uhf = new_mol.GetSpinMultiplicity() - 1
        correct_atom_mapping = new_mol.GetAtomMapNumbers()

        failed_ids = set()
        all_props = []
        for c_id in range(len(mol_data)):
            try:
                props, opt_mol = run_xtb_calc(new_mol, confId=c_id, job="--opt", return_optmol=True,
                                              method=self.method, level=self.level, uhf=uhf)
                all_props.append(props)
            except ValueError as e:
                failed_ids.add(c_id)
                print(e)
                continue

            opt_mol.SetAtomMapNumbers(correct_atom_mapping)
            # Renumber the molecule based on the atom mapping just set
            opt_mol.RenumberAtoms()
            positions = opt_mol.GetPositions()
            conf = new_mol.GetConformer(id=c_id)
            conf.SetPositions(positions)
            energy = float(opt_mol.GetProp('total energy / Eh'))  # * HARTREE_TO_KCAL_MOL  # kcal/mol (TODO: check)
            mol_data[c_id].update({"positions": positions,  # issues if not all opts succeeded?
                                   "conf": conf,
                                   "energy": energy})

        final_mol_data = [c for i, c in enumerate(mol_data) if i not in failed_ids]

        if self.track_stats:
            self.n_failures = len(failed_ids)
            self.percent_failures = self.n_failures / len(mol_data) * 100
            self.n_opt_cycles = [p["n_opt_cycles"] if "n_opt_cycles" in p else -1 for p in all_props]

        return sorted(final_mol_data, key=lambda x: x["energy"])

class GaussianOptimizer(ConfGenOptimizer):
    """
    Optimizer using the Gaussian.

    Args:
        method (str, optional): The method to be used for species optimization. You can use the level of theory available in Gaussian.
                                Defaults to ``"GFN2-xTB"``, which is realized by additional scripts provided in the ``rdmc`` package.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 memory: int = 1,
                 track_stats: bool = False):
        """
        Initiate the Gaussian optimizer.

        Args:
            method (str, optional): The method to be used for stable species optimization. you can use the level of theory available in Gaussian.
                                    We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(GaussianOptimizer, self).__init__(track_stats)
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

        for version in ['g16', 'g09', 'g03']:
            GAUSSIAN_ROOT = os.environ.get(f"{version}root")
            if GAUSSIAN_ROOT:
                break
        else:
            raise RuntimeError('No Gaussian installation found.')

        self.gaussian_binary = os.path.join(GAUSSIAN_ROOT, version, version)

    def optimize_conformers(self,
                            mol: 'RDKitMol',
                            multiplicity: int = 1,
                            save_dir: Optional[str] = None,
                            **kwargs,
                            ) -> 'RDKitMol':
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
            gaussian_str = write_gaussian_opt(mol,
                                              conf_id=i,
                                              method=self.method,
                                              mult=multiplicity,
                                              nprocs=self.nprocs,
                                              memory=self.memory)
            gaussian_input_file = os.path.join(conf_dir, "gaussian_opt.gjf")
            with open(gaussian_input_file, "w") as f:
                f.writelines(gaussian_str)

            # Run the gaussian via subprocess
            with open(os.path.join(conf_dir, "gaussian_opt.log"), "w") as f:
                gaussian_run = subprocess.run(
                    [self.gaussian_binary, gaussian_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            # Check the output of the gaussian
            if gaussian_run.returncode == 0:
                try:
                    g16_log = GaussianLog(os.path.join(conf_dir, "gaussian_opt.log"))
                    pre_adj_mat = mol.GetAdjacencyMatrix()
                    post_adj_mat = g16_log.get_mol(refid=g16_log.num_all_geoms-1,  # The last geometry in the job
                                                   converged=False,
                                                   sanitize=False,
                                                   backend='openbabel').GetAdjacencyMatrix()
                    if g16_log.success and (pre_adj_mat == post_adj_mat).all():
                        new_mol = g16_log.get_mol(embed_conformers=True, sanitize=False)
                        opt_mol.AddConformer(new_mol.GetConformer().ToConformer(), assignId=True)
                        opt_mol.energy.update({i: g16_log.get_scf_energies(relative=False)[-1]})
                        opt_mol.frequency.update({i: g16_log.freqs})
                    else:
                        opt_mol.AddNullConformer(confId=i)
                        opt_mol.energy.update({i: np.nan})
                        opt_mol.KeepIDs[i] = False
                        print("Error! Likely that the smiles doesn't correspond to this species.")
                except Exception as e:
                    opt_mol.AddNullConformer(confId=i)
                    opt_mol.energy.update({i: np.nan})
                    opt_mol.KeepIDs[i] = False
                    print(f'Got an error when reading the Gaussian output: {e}')
            else:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                opt_mol.KeepIDs[i] = False

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol(), opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol

    def save_opt_mols(self,
                      save_dir: str,
                      opt_mol: 'RDKitMol',
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

    def __call__(self,
                 mol: 'RDKitMol',
                 save_dir: str,
                 **kwargs):
        """
        Run the workflow to generate optimize stable species guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            save_dir (str, optional): The path to save results. Defaults to ``None``.

        Returns:
            'RDKitMol': The optimized molecule as RDKitMol with 3D geometries embedded.
        """
        time_start = time()
        opt_mol = self.optimize_conformers(mol=mol, save_dir=save_dir, **kwargs)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return opt_mol
