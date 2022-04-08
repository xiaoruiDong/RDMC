#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for verifying optimized ts
"""

from rdmc import RDKitMol
from rdmc.external.xtb_tools.opt import run_xtb_calc
from rdmc.external.orca import write_orca_irc
from rdmc.external.gaussian import GaussianLog, write_gaussian_irc
import os
import pickle
import subprocess
from time import time


class TSVerifier:
    def __init__(self, track_stats=False):
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def verify_ts_guesses(self, ts_mol, keep_ids, save_dir=None, **kwargs):
        raise NotImplementedError

    def __call__(self, ts_mol, keep_ids, save_dir=None, **kwargs):
        time_start = time()
        keep_ids = self.verify_ts_guesses(ts_mol, keep_ids, save_dir, **kwargs)

        if not self.track_stats:
            return keep_ids

        time_end = time()
        stats = {"time": time_end - time_start}
        self.stats.append(stats)

        return keep_ids


class XTBFrequencyVerifier(TSVerifier):
    def __init__(self, track_stats=False):
        super(XTBFrequencyVerifier, self).__init__(track_stats)

    def verify_ts_guesses(self, ts_mol, keep_ids, save_dir=None, **kwargs):

        r_smi, _ = kwargs["rxn_smiles"].split(">>")
        r_mol = RDKitMol.FromSmiles(r_smi)
        uhf = r_mol.GetSpinMultiplicity() - 1

        freq_checks = []
        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                props = run_xtb_calc(ts_mol, confId=i, job="--hess", uhf=uhf)
                if sum(props["frequencies"] < 0) == 1:
                    freq_checks.append(True)
                else:
                    freq_checks.append(False)
            else:
                freq_checks.append(False)

        if save_dir:
            with open(os.path.join(save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(freq_checks, f)

        return freq_checks


class OrcaIRCVerifier(TSVerifier):
    def __init__(self, method="XTB2", multiplicity=1, nprocs=1, track_stats=False):
        super(OrcaIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.multiplicity = multiplicity
        self.nprocs = nprocs

    def verify_ts_guesses(self, ts_mol, keep_ids, save_dir=None, **kwargs):

        ORCA_BINARY = os.environ.get("ORCA")
        irc_checks = []

        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                orca_str = write_orca_irc(ts_mol, confId=i, method=self.method, mult=self.multiplicity, 
                                          nprocs=self.nprocs)
                orca_dir = os.path.join(save_dir, f"orca_irc{i}")
                os.makedirs(orca_dir)

                orca_input_file = os.path.join(orca_dir, "orca_irc.inp")
                with open(orca_input_file, "w") as f:
                    f.writelines(orca_str)

                with open(os.path.join(orca_dir, "orca_irc.log"), "w") as f:
                    orca_run = subprocess.run(
                        [ORCA_BINARY, orca_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )
                if orca_run.returncode != 0:
                    irc_checks.append(False)
                    continue

                # do irc
                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()

                try:
                    irc_f_mol = RDKitMol.FromFile(os.path.join(orca_dir, "orca_irc_IRC_F.xyz"), sanitize=False)
                    irc_b_mol = RDKitMol.FromFile(os.path.join(orca_dir, "orca_irc_IRC_B.xyz"), sanitize=False)
                except FileNotFoundError:
                    irc_checks.append(False)
                    continue

                f_adj = irc_f_mol.GetAdjacencyMatrix()
                b_adj = irc_b_mol.GetAdjacencyMatrix()

                try:
                    rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                    rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
                except AttributeError:
                    print("Error! Likely that the reaction smiles doesn't correspond to this reaction.")

                if rf_pb_check or rb_pf_check:
                    irc_checks.append(True)
                else:
                    irc_checks.append(False)

            else:
                irc_checks.append(False)

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(irc_checks, f)

        return irc_checks


class GaussianIRCVerifier(TSVerifier):
    def __init__(self, method="GFN2-xTB", multiplicity=1, nprocs=1, track_stats=False):
        super(GaussianIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.multiplicity = multiplicity
        self.nprocs = nprocs

    def verify_ts_guesses(self, ts_mol, keep_ids, save_dir=None, **kwargs):

        GAUSSIAN_BINARY = os.path.join(os.environ.get("g16root"), "g16", "g16")
        irc_checks = []

        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                gaussian_f_str = write_gaussian_irc(ts_mol, confId=i, method=self.method, direction="forward",
                                                    mult=self.multiplicity, nprocs=self.nprocs)
                gaussian_r_str = write_gaussian_irc(ts_mol, confId=i, method=self.method, direction="reverse",
                                                    mult=self.multiplicity, nprocs=self.nprocs)
                gaussian_dir = os.path.join(save_dir, f"gaussian_irc{i}")
                os.makedirs(gaussian_dir)

                gaussian_f_input_file = os.path.join(gaussian_dir, "gaussian_irc_forward.gjf")
                with open(gaussian_f_input_file, "w") as f:
                    f.writelines(gaussian_f_str)

                gaussian_r_input_file = os.path.join(gaussian_dir, "gaussian_irc_reverse.gjf")
                with open(gaussian_r_input_file, "w") as f:
                    f.writelines(gaussian_r_str)

                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()

                # do irc
                with open(os.path.join(gaussian_dir, "gaussian_irc_forward.log"), "w") as f:
                    gaussian_f_run = subprocess.run(
                        [GAUSSIAN_BINARY, gaussian_f_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )

                # extract forward irc mol
                g16_f_log = GaussianLog(os.path.join(gaussian_dir, "gaussian_irc_forward.log"))
                irc_f_mol = g16_f_log.get_mol(converged=False, sanitize=False)
                n_f_confs = irc_f_mol.GetNumConformers()
                f_adj = RDKitMol.FromXYZ(irc_f_mol.ToXYZ(confId=n_f_confs-1), sanitize=False).GetAdjacencyMatrix()

                with open(os.path.join(gaussian_dir, "gaussian_irc_reverse.log"), "w") as f:
                    gaussian_r_run = subprocess.run(
                        [GAUSSIAN_BINARY, gaussian_r_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )

                # check reverse irc mol and full irc
                g16_b_log = GaussianLog(os.path.join(gaussian_dir, "gaussian_irc_reverse.log"))
                irc_b_mol = g16_b_log.get_mol(converged=False, sanitize=False)
                n_b_confs = irc_b_mol.GetNumConformers()
                b_adj = RDKitMol.FromXYZ(irc_b_mol.ToXYZ(confId=n_b_confs-1), sanitize=False).GetAdjacencyMatrix()

                try:
                    rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                    rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
                except AttributeError:
                    print("Error! Likely that the reaction smiles doesn't correspond to this reaction.")

                if rf_pb_check or rb_pf_check:
                    irc_checks.append(True)
                else:
                    irc_checks.append(False)

            else:
                irc_checks.append(False)

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(irc_checks, f)

        return irc_checks
