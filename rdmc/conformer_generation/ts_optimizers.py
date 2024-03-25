#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for optimizing transition state geometries.
"""

# Import RDKit and RDMC first to avoid unexpected errors
from rdkit import Chem
from rdkit.Chem.rdchem import Conformer
from rdmc import RDKitMol
import numpy as np
import pickle

import os
import subprocess
from time import time
from typing import List, Optional

from rdmc.external.inpwriter import (write_gaussian_opt,
                                     write_orca_opt,
                                     write_qchem_opt)
from rdmc.external.logparser import (GaussianLog, ORCALog, QChemLog)

try:
    from rdmc.external.sella import run_sella_opt
except BaseException:
    print("No Sella installation deteced. Skipping import...")


class TSOptimizer:
    """
    The abstract class for TS optimizer.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self,
                 track_stats: Optional[bool] = False):
        """
        Initialize the TS optimizer.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_ts_guesses(self,
                            mol: 'RDKitMol',
                            save_dir: str,
                            **kwargs):
        """
        The abstract method for optimizing TS guesses. It will be implemented in actual classes.
        The method needs to take ``mol`` in ``RDKitMol`` and ``save_dir`` as ``str`` as input arguments, and
        return the optimized molecule as ``RDKitMol``.

        Args:
            mol (RDKitMol): The TS in RDKitMol object with geometries embedded as conformers.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        raise NotImplementedError

    def save_opt_mols(self,
                      save_dir: str,
                      opt_mol: 'RDKitMol',
                      keep_ids: dict,
                      energies: dict,
                      ):
        """
        Save the information of the optimized TS geometries into the directory.

        Args:
            save_dir (str): The path to the directory to save the results.
            opt_mol (RDKitMol): The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
            keep_ids (dict): Dictionary of which opts succeeded and which failed.
            energies (dict): Dictionary of energies for each conformer.
        """
        # Save optimized ts mols
        ts_path = os.path.join(save_dir, "ts_optimized_confs.sdf")
        try:
            ts_writer = Chem.rdmolfiles.SDWriter(ts_path)
            for i in range(opt_mol.GetNumConformers()):
                opt_mol.SetProp("Energy", str(energies[i]))
                ts_writer.write(opt_mol, confId=i)
        except Exception:
            raise
        finally:
            ts_writer.close()

        # save ids
        with open(os.path.join(save_dir, "opt_check_ids.pkl"), "wb") as f:
            pickle.dump(keep_ids, f)

    def __call__(self,
                 mol: 'RDKitMol',
                 save_dir: Optional[str] = None,
                 **kwargs):
        """
        Run the workflow to generate optimize TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            save_dir (str, optional): The path to save results. Defaults to ``None``.

        Returns:
            'RDKitMol': The optimized TS molecule as ``RDKitMol`` with 3D geometries embedded.
        """
        time_start = time()

        if not save_dir:
            save_dir = os.getcwd()

        opt_mol = self.optimize_ts_guesses(mol=mol, save_dir=save_dir, **kwargs)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return opt_mol


class SellaOptimizer(TSOptimizer):
    """
    The class to optimize TS geometries using the Sella algorithm.
    It uses XTB as the backend calculator, ASE as the interface, and Sella module from the Sella repo.

    Args:
        method (str, optional): The method in XTB used to optimize the geometry. Options are
                                ``'GFN1-xTB'`` and ``'GFN2-xTB'``. Defaults to ``"GFN2-xTB"``.
        fmax (float, optional): The force threshold used in the optimization. Defaults to ``1e-3``.
        steps (int, optional): Max number of steps allowed in the optimization. Defaults to ``1000``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self,
                 method: str = "GFN2-xTB",
                 fmax: float = 1e-3,
                 steps: int = 1000,
                 track_stats: bool = False):
        """
        Initiate the Sella optimizer.

        Args:
            method (str, optional): The method in XTB used to optimize the geometry. Options are 'GFN1-xTB' and 'GFN2-xTB'. Defaults to "GFN2-xTB".
            fmax (float, optional): The force threshold used in the optimization. Defaults to 1e-3.
            steps (int, optional): Max number of steps allowed in the optimization. Defaults to 1000.
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(SellaOptimizer, self).__init__(track_stats)

        self.method = method
        self.fmax = fmax
        self.steps = steps

    def optimize_ts_guesses(self,
                            mol: 'RDKitMol',
                            save_dir: Optional[str] = None,
                            **kwargs):
        """
        Optimize the TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            save_dir (str, optional): The path to save results. Defaults to ``None``.

        Returns:
            RDKitMol: The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        opt_mol = mol.Copy(copy_attrs=["KeepIDs"])
        opt_mol.energy = {}
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}
        for i in range(mol.GetNumConformers()):

            if not opt_mol.KeepIDs[i]:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                continue

            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"sella_opt{i}")
                os.makedirs(ts_conf_dir, exist_ok=True)

            opt_mol = run_sella_opt(opt_mol,
                                    method=self.method,
                                    confId=i,
                                    fmax=self.fmax,
                                    steps=self.steps,
                                    save_dir=ts_conf_dir,
                                    copy_attrs=["KeepIDs", "energy", "frequency"],
                                    )
        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol(), opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol


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

    def __init__(self,
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
            raise RuntimeError('No Orca binary is found in the PATH.')
        else:
            self.orca_binary = ORCA_BINARY

    def extract_frequencies(self,
                            save_dir: str,
                            n_atoms: int):
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
            freqs = orca_data[freq_idx - 4 - dof: freq_idx - 4]
            freqs.reverse()
            return np.array([float(line.split()[1]) for line in freqs])
        else:
            return None

    def optimize_ts_guesses(self,
                            mol: 'RDKitMol',
                            multiplicity: int = 1,
                            save_dir: Optional[str] = None,
                            **kwargs):
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
            orca_str = write_orca_opt(mol,
                                      conf_id=i,
                                      ts=True,
                                      method=self.method,
                                      mult=multiplicity,
                                      nprocs=self.nprocs)

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
                    new_mol = RDKitMol.FromFile(os.path.join(ts_conf_dir, "orca_opt.xyz"), sanitize=False)
                    opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                    opt_mol.frequency.update({i: self.extract_frequencies(ts_conf_dir, opt_mol.GetNumAtoms())})
                except Exception as e:
                    opt_mol.AddNullConformer(confId=i)
                    opt_mol.energy.update({i: np.nan})
                    opt_mol.KeepIDs[i] = False
                    print(f'Cannot read Orca output, got: {e}')
            else:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                opt_mol.KeepIDs[i] = False

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol(), opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol


class GaussianOptimizer(TSOptimizer):
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

    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 memory: int = 1,
                 track_stats: bool = False):
        """
        Initiate the Gaussian berny optimizer.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                    We provided a script to run XTB using Gaussian, but there are some extra steps to do.
                                    Defaults to ``"GFN2-xTB"``.
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

    def optimize_ts_guesses(self,
                            mol: 'RDKitMol',
                            multiplicity: int = 1,
                            save_dir: Optional[str] = None,
                            **kwargs):
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
            gaussian_str = write_gaussian_opt(mol,
                                              conf_id=i,
                                              ts=True,
                                              method=self.method,
                                              mult=multiplicity,
                                              nprocs=self.nprocs,
                                              memory=self.memory)
            gaussian_input_file = os.path.join(ts_conf_dir, "gaussian_opt.gjf")
            with open(gaussian_input_file, "w") as f:
                f.writelines(gaussian_str)

            # Run the gaussian via subprocess
            with open(os.path.join(ts_conf_dir, "gaussian_opt.log"), "w") as f:
                gaussian_run = subprocess.run(
                    [self.gaussian_binary, gaussian_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            # Check the output of the gaussian
            if gaussian_run.returncode == 0:
                try:
                    g16_log = GaussianLog(os.path.join(ts_conf_dir, "gaussian_opt.log"))
                    if g16_log.success:
                        new_mol = g16_log.get_mol(embed_conformers=False, sanitize=False)
                        opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                        opt_mol.energy.update({i: g16_log.get_scf_energies(relative=False)[-1]})
                        opt_mol.frequency.update({i: g16_log.freqs})
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


class QChemOptimizer(TSOptimizer):
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

    def __init__(self,
                 method: str = "wB97x-d3",
                 basis: str = "def2-tzvp",
                 nprocs: int = 1,
                 track_stats: bool = False):
        """
        Initiate the QChem EF optimizer.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the method available in QChem.
                                    Defaults to ``"wB97x-d3"``.
            basis (str, optional): The method to be used for TS optimization. you can use the basis available in QChem.
                                    Defaults to ``"def2-tzvp"``.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(QChemOptimizer, self).__init__(track_stats)

        self.method = method
        self.basis = basis
        self.nprocs = nprocs

        QCHEM_BINARY = os.environ.get("qchem")
        if not QCHEM_BINARY:
            raise RuntimeError('No QChem binary is found in the PATH.')
        else:
            self.qchem_binary = QCHEM_BINARY

    def optimize_ts_guesses(self,
                            mol: 'RDKitMol',
                            multiplicity: int = 1,
                            save_dir: Optional[str] = None,
                            **kwargs):
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
            qchem_str = write_qchem_opt(mol,
                                        conf_id=i,
                                        ts=True,
                                        method=self.method,
                                        basis=self.basis,
                                        mult=multiplicity)

            qchem_input_file = os.path.join(ts_conf_dir, "qchem_opt.qcin")
            with open(qchem_input_file, "w") as f:
                f.writelines(qchem_str)

            # Run the qchem via subprocess
            with open(os.path.join(ts_conf_dir, "qchem_opt.log"), "w") as f:
                qchem_run = subprocess.run(
                    [self.qchem_binary, "-nt", str(self.nprocs), qchem_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            # Check the output of the qchem
            if qchem_run.returncode == 0:
                try:
                    qchem_log = QChemLog(os.path.join(ts_conf_dir, "qchem_opt.log"))
                    if qchem_log.success:
                        new_mol = qchem_log.get_mol(embed_conformers=False, sanitize=False)
                        opt_mol.AddConformer(new_mol.GetConformer(), assignId=True)
                        opt_mol.energy.update({i: qchem_log.get_scf_energies(relative=False)[-1]})
                        opt_mol.frequency.update({i: qchem_log.freqs})
                except Exception as e:
                    opt_mol.AddNullConformer(confId=i)
                    opt_mol.energy.update({i: np.nan})
                    opt_mol.KeepIDs[i] = False
                    print(f'Got an error when reading the QChem output: {e}')
            else:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                opt_mol.KeepIDs[i] = False

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol(), opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol
