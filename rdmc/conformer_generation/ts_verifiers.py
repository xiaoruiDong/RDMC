#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for verifying optimized ts
"""

import os
import pickle
import subprocess
from rdmc import RDKitMol
from rdmc.external.xtb_tools.opt import run_xtb_calc
from rdmc.external.orca import write_orca_irc
from rdmc.external.gaussian import GaussianLog, write_gaussian_irc


class XTBFrequencyVerifier:
    def __init__(self, track_stats=False):
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def __call__(self, ts_mol, keep_ids, save_dir=None, **kwargs):

        freq_checks = []
        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                props = run_xtb_calc(ts_mol, confId=i, job="--hess")
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


class OrcaIRCVerifier:
    def __init__(self, method="XTB2", track_stats=False):
        self.track_stats = track_stats
        self.method = method

    def __call__(self, ts_mol, keep_ids, save_dir, **kwargs):

        ORCA_BINARY = os.environ.get("ORCA")
        irc_checks = []

        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                orca_str = write_orca_irc(ts_mol, confId=i, method=self.method)
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


class GaussianIRCVerifier:
    def __init__(self, method="GFN2-xTB", track_stats=False):
        self.track_stats = track_stats
        self.method = method

    def __call__(self, ts_mol, keep_ids, save_dir, **kwargs):

        GAUSSIAN_BINARY = os.path.join(os.environ.get("g16root"), "g16", "g16")
        irc_checks = []

        for i in range(ts_mol.GetNumConformers()):
            if keep_ids[i]:
                gaussian_f_str = write_gaussian_irc(ts_mol, confId=i, method=self.method, direction="forward")
                gaussian_r_str = write_gaussian_irc(ts_mol, confId=i, method=self.method, direction="reverse")
                gaussian_dir = os.path.join(save_dir, f"gaussian_irc{i}")
                os.makedirs(gaussian_dir)

                gaussian_f_input_file = os.path.join(gaussian_dir, "gaussian_irc_forward.gjf")
                with open(gaussian_f_input_file, "w") as f:
                    f.writelines(gaussian_f_str)

                gaussian_r_input_file = os.path.join(gaussian_dir, "gaussian_irc_reverse.gjf")
                with open(gaussian_r_input_file, "w") as f:
                    f.writelines(gaussian_r_str)

                # do irc
                with open(os.path.join(gaussian_dir, "gaussian_irc_forward.log"), "w") as f:
                    gaussian_f_run = subprocess.run(
                        [GAUSSIAN_BINARY, gaussian_f_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )
                if gaussian_f_run.returncode != 0:
                    irc_checks.append(False)
                    continue

                with open(os.path.join(gaussian_dir, "gaussian_irc_reverse.log"), "w") as f:
                    gaussian_r_run = subprocess.run(
                        [GAUSSIAN_BINARY, gaussian_r_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )
                if gaussian_r_run.returncode != 0:
                    irc_checks.append(False)
                    continue

                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()

                g16_f_log = GaussianLog(os.path.join(gaussian_dir, "gaussian_irc_forward.log"))
                g16_r_log = GaussianLog(os.path.join(gaussian_dir, "gaussian_irc_reverse.log"))

                if g16_f_log.success and g16_r_log.success:
                    irc_f_mol = g16_f_log.get_mol(embed_conformers=False, sanitize=False)
                    irc_r_mol = g16_r_log.get_mol(embed_conformers=False, sanitize=False)
                else:
                    irc_checks.append(False)
                    continue

                f_adj = irc_f_mol.GetAdjacencyMatrix()
                b_adj = irc_r_mol.GetAdjacencyMatrix()

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
