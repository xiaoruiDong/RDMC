#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for verifying optimized ts
"""

# Import RDMC first to avoid unexpected errors
from rdmc import RDKitMol

import os
import pickle
import subprocess
from time import time
from typing import Optional

from rdmc.external.xtb_tools.opt import run_xtb_calc
from rdmc.external.orca import write_orca_irc
from rdmc.external.gaussian import GaussianLog, write_gaussian_irc


class TSVerifier:
    """
    The abstract class for TS verifiers.
    """
    def __init__(self,
                 track_stats: bool = False):
        """
        Initialize the TS verifier.

        Args:
            track_stats (bool, optional): Whether to track status. Defaults to False.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          keep_ids: list,
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        The abstract method for verifying TS guesses (or optimized TS geometries). The method need to take
        `ts_mol` in RDKitMol, `keep_ids` in list, `multiplicity` in int, and `save_dir` in str, and returns
        a list indicating the ones passing the check.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            keep_ids (list): A list of Trues and Falses.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def __call__(self,
                 ts_mol: 'RDKitMol',
                 keep_ids: list,
                 multiplicity: int = 1,
                 save_dir: Optional[str] = None,
                 **kwargs):
        """
        Run the workflow for verifying the TS guessers (or optimized TS conformers).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            keep_ids (list): A list of Trues and Falses.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            list: a list of true and false
        """
        time_start = time()
        keep_ids = self.verify_ts_guesses(ts_mol, keep_ids, save_dir, multiplicity=multiplicity, **kwargs)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return keep_ids


class XTBFrequencyVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its frequencies using XTB.
    """

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          keep_ids: list,
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            keep_ids (list): A list of Trues and Falses.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            list
        """
        freq_checks = []
        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                props = run_xtb_calc(ts_mol, confId=i, job="--hess", uhf=multiplicity - 1)
                # Check if the number of negative frequencies is equal to 1
                freq_checks.append(sum(props["frequencies"] < 0) == 1)
            else:
                freq_checks.append(False)

        if save_dir:
            with open(os.path.join(save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(freq_checks, f)

        return freq_checks


class OrcaIRCVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Orca.
    """

    def __init__(self,
                 method: str = "XTB2",
                 nprocs: int = 1,
                 track_stats: bool = False):
        """
        Initiate the Orca IRC verifier.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                    If you want to use XTB methods, you need to put the xtb binary into the Orca directory. Defaults to XTB2.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(OrcaIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs

        ORCA_BINARY = os.environ.get("ORCA")
        if not ORCA_BINARY:
            raise RuntimeError('No Orca binary is found in the PATH.')
        else:
            self.orca_binary = ORCA_BINARY

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          keep_ids: list,
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            keep_ids (list): A list of Trues and Falses.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.
        """
        irc_checks = []
        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:

                # Create and save the Orca input file
                orca_str = write_orca_irc(ts_mol,
                                          confId=i,
                                          method=self.method,
                                          mult=multiplicity,
                                          nprocs=self.nprocs)
                orca_dir = os.path.join(save_dir, f"orca_irc{i}")
                os.makedirs(orca_dir)

                orca_input_file = os.path.join(orca_dir, "orca_irc.inp")
                with open(orca_input_file, "w") as f:
                    f.writelines(orca_str)

                # Run the Orca IRC using subprocess
                with open(os.path.join(orca_dir, "orca_irc.log"), "w") as f:
                    orca_run = subprocess.run(
                        [self.orca_binary, orca_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )
                if orca_run.returncode != 0:
                    irc_checks.append(False)
                    continue

                # Generate the adjacency matrix from the SMILES
                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()

                # Read the terminal geometries from the IRC analysis into RDKitMol
                try:
                    irc_f_mol = RDKitMol.FromFile(os.path.join(orca_dir, "orca_irc_IRC_F.xyz"), sanitize=False)
                    irc_b_mol = RDKitMol.FromFile(os.path.join(orca_dir, "orca_irc_IRC_B.xyz"), sanitize=False)
                except FileNotFoundError:
                    irc_checks.append(False)
                    continue

                # Generate the adjacency matrix from the mols in the IRC analysis
                f_adj = irc_f_mol.GetAdjacencyMatrix()
                b_adj = irc_b_mol.GetAdjacencyMatrix()

                # Comparing the adjacency matrix
                try:
                    rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                    rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
                except AttributeError:
                    print("Error! Likely that the reaction smiles doesn't correspond to this reaction.")

                irc_checks.append(rf_pb_check or rb_pf_check)

            else:
                irc_checks.append(False)

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(irc_checks, f)

        return irc_checks


class GaussianIRCVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Gaussian.
    """

    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 track_stats: bool = False):
        """
        Initiate the Gaussian IRC verifier.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                    We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(GaussianIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs

        for version in ['g16', 'g09', 'g03']:
            GAUSSIAN_ROOT = os.environ.get(f"{version}root")
            if GAUSSIAN_ROOT:
                break
        else:
            raise RuntimeError('No Gaussian installation found.')

        self.gaussian_binary = os.path.join(GAUSSIAN_ROOT, version, version)

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          keep_ids: list,
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            keep_ids (list): A list of Trues and Falses.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.
        """
        irc_checks = []
        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:

                adj_mat = []
                irc_check = True

                gaussian_dir = os.path.join(save_dir, f"gaussian_irc{i}")
                os.makedirs(gaussian_dir, exist_ok=True)

                for direction in ['forward', 'reverse']:
                    gaussian_str = write_gaussian_irc(ts_mol,
                                                        confId=i,
                                                        method=self.method,
                                                        direction=direction,
                                                        mult=multiplicity,
                                                        nprocs=self.nprocs)

                    gaussian_input_file = os.path.join(gaussian_dir, f"gaussian_irc_{direction}.gjf")
                    with open(gaussian_input_file, "w") as f:
                        f.writelines(gaussian_str)

                    # Run IRC using subprocess
                    with open(os.path.join(gaussian_dir, f"gaussian_irc_{direction}.log"), "w") as f:
                        gaussian_run = subprocess.run(
                            [self.gaussian_binary, gaussian_input_file],
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            cwd=os.getcwd(),
                        )

                    # Extract molecule adjacency matrix from IRC results
                    # TBD: We can stop running IRC if one side of IRC fails
                    # I personally think it is worth to continue to run the other IRC just to provide more sights
                    if gaussian_run.returncode == 0:
                        try:
                            glog = GaussianLog(os.path.join(gaussian_dir, f"gaussian_irc_{direction}.log"))
                            irc_mol = glog.get_mol(converged=False, sanitize=False)
                            n_confs = irc_mol.GetNumConformers()
                            adj_mat.append(RDKitMol.FromXYZ(irc_mol.ToXYZ(confId=n_confs-1), sanitize=False).GetAdjacencyMatrix())
                        except:
                            irc_check = False
                    else:
                        irc_check = False

                # Bypass the further steps if IRC job fails
                if not irc_check and len(adj_mat) != 2:
                    irc_checks.append(False)
                    continue
                f_adj, b_adj = adj_mat

                # Generate the adjacency matrix from the SMILES
                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()

                try:
                    rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                    rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
                except AttributeError:
                    print("Error! Likely that the reaction smiles doesn't correspond to this reaction.")

                irc_checks.append(rf_pb_check or rb_pf_check)

            else:
                irc_checks.append(False)

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(irc_checks, f)

        return irc_checks
