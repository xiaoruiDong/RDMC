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
                orca_dir = os.path.join(save_dir, f"orca_conf{i}")
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

                rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
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
