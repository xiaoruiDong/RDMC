#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for optimizing initial guess geometries
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
    from rdmc.external.xtb_tools.run_xtb import run_xtb_calc
except ImportError:
    pass


class ConfGenOptimizer:
    def __init__(self, track_stats=False):

        self.iter = 0
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_conformers(self, mol_data):
        raise NotImplementedError

    def __call__(self, mol_data):

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


class GaussianOptimizer(ConfGenOptimizer):
    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 memory: int = 1,
                 track_stats: bool = False):
        """
        Initiate the Gaussian berny optimizer.

        Args:
            method (str, optional): The method to be used for stable species optimization. you can use the level of theory available in Gaussian.
                                    We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
            track_stats (bool, optional): Whether to track the status. Defaults to False.
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
                            **kwargs):
        """
        Optimize the conformers.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            multiplicity (int): The multiplicity of the molecule. Defaults to 1.
            save_dir (Optional[str], optional): The path to save the results. Defaults to None.

        Returns:
            RDKitMol
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
            keep_ids (dict): Dictionary of which opts succeeded and which failed
            energies (dict): Dictionary of energies for each conformer
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
            save_dir (str, optional): The path to save results. Defaults to None.
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
