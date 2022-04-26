#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
CREST wrappers.
Taken from Simon Axelrod/RGB
"""

import os
from shutil import rmtree
import subprocess
import tempfile
from rdmc.external.xtb_tools.utils import CREST_BINARY, XTB_ENV


def make_xyz_text(rd_mol,
                  comment):
    atoms = [i for i in rd_mol.GetAtoms()]
    num_atoms = len(atoms)
    pos = rd_mol.GetConformers()[0].GetPositions()

    lines = [str(num_atoms), comment]

    for atom, this_pos in zip(atoms, pos):
        line = "%s %.8f %.8f %.8f " % (atom.GetSymbol(),
                                       this_pos[0], this_pos[1], this_pos[2])
        lines.append(line)

    text = "\n".join(lines)
    return text


def write_confs_xyz(confs, path):
    text = ""
    for i, conf in enumerate(confs):
        rd_mol = conf["conf"].GetOwningMol()
        energy = conf["energy"]
        comment = "%.8f !CONF%d" % (energy, i + 1)

        this_text = make_xyz_text(rd_mol=rd_mol,
                                  comment=comment)

        if i != 0:
            text += "\n"
        text += this_text

    with open(path, 'w') as f:
        f.write(text)


def read_unique(job_dir):
    path = os.path.join(job_dir, "enso.tags")
    with open(path, 'r') as f:
        lines = f.readlines()
    unique_idx = []
    for line in lines:
        split = line.strip().split()
        if not split:
            continue

        idx = split[-1].split("!CONF")[-1]
        # means something went wrong
        if not idx.isdigit():
            return

        unique_idx.append(int(idx) - 1)

    return unique_idx


def run_cre_check(confs, ethr=0.15, rthr=0.125, bthr=0.01, ewin=10000):
    temp_dir = tempfile.mkdtemp()

    logfile = os.path.join(temp_dir, "xtb.log")
    confs_path = os.path.join(temp_dir, "confs.xyz")
    conf_0_path = os.path.join(temp_dir, "conf_0.xyz")
    cregen_out = os.path.join(temp_dir, "cregen.out")

    write_confs_xyz(confs, path=confs_path)
    write_confs_xyz(confs[:1], path=conf_0_path)

    with open(logfile, "w") as f:
        xtb_run = subprocess.run(
            [
                CREST_BINARY,
                conf_0_path,
                "--cregen",
                confs_path,
                "--ethr", str(ethr), "--rthr", str(rthr), "--bthr", str(bthr), "--ewin", str(ewin), "--enso",
                ">",
                cregen_out,
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=temp_dir,
            env=XTB_ENV,
        )

    if xtb_run.returncode != 0:
        error_out = os.path.join(temp_dir, "xtb.log")
        raise ValueError(f"xTB calculation failed. See {error_out} for details.")

    unique_ids = read_unique(temp_dir)
    updated_confs = [confs[i] for i in unique_ids]
    rmtree(temp_dir)

    ### DEBUG ###
    # num_removed = len(confs) - len(unique_ids)
    # plural = 's' if num_removed > 1 else ''
    # print("Removed %d duplicate conformer%s with cregen" % (num_removed, plural))
    ### DEBUG ###

    return updated_confs, unique_ids
