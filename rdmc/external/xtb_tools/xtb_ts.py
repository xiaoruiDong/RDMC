#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
XTB + gaussian optimization. Modified based on https://github.com/jensengroup/ReactionDiscovery/blob/main/xtb_ts_test/run_xtb_ts.py
"""

import os
import sys
import subprocess
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem import rdchem

from rdmc import RDKitMol
from rdmc.ts import get_formed_and_broken_bonds
from rdmc.external.gaussian import GaussianLog


XTB_GAUSSIAN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xtb_gaussian.py')
GAUSSIAN_CMD = "g16"
GAUSSIAN_INPUT_TEMPLATE = """%nprocshared={cpus}
%mem={mem}GB
{scheme}
external={gaus_xtb_path}

title

{chrg} {spin}
{ts_xyz}


"""

SCHEMES = {'irc_forward': '#irc=(forward, calcfc, maxpoint=100, stepsize=5) ',
           'irc_reverse': '#irc=(reverse, calcfc, maxpoint=100, stepsize=5) ',
           'opt': '#opt ',
           'opt_ts': '#opt=(calcall, ts, noeigen, nomicro, maxstep=15) ',
           'opt_constraint': '#opt geom(modredundant, gic) ',
    }


def run_cmd(cmd):
    """
    Run command line
    """
    cmd = cmd.split()
    print(cmd)
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, _ = p.communicate()
    return output.decode('utf-8')


# I don't think it is needed
def reorder_atoms_to_map(mol):

    """
    Reorders the atoms in a mol objective to match that of the mapping
    """

    atom_map_order = np.zeros(mol.GetNumAtoms()).astype(int)
    for atom in mol.GetAtoms():
        map_number = atom.GetAtomMapNum()-1
        atom_map_order[map_number] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, atom_map_order.tolist())
    return mol


def atom_information(xyz_file):
    """
    extract information about system: number of atoms and atom numbers
    """
    atom_numbers = []
    with open(xyz_file, 'r') as f:
        line = f.readline()
        n_atoms = int(line.split()[0])
        f.readline()
        for _ in range(n_atoms):
            line = f.readline().split()
            atom_number = line[0]
            atom_numbers.append(atom_number)

    return n_atoms, atom_numbers


def bonds_getting_formed_or_broken(rsmi, psmi):
    """
    Get bonds getting formed or broken
    """
    rmol = RDKitMol.FromSmiles(rsmi)
    pmol = RDKitMol.FromSmiles(psmi)

    formed, broken = get_formed_and_broken_bonds(rmol, pmol)
    return formed + broken

def get_gaussian_input_content(xyz_file,
                               scheme: str,
                               chrg: int = 0,
                               spin: int = 1,
                               cpus: int = 1,
                               mem: float = 1.,
                               ):
    with open(xyz_file, 'r') as file_in:
        lines = file_in.readlines()[2:]
    ts_xyz = '\n'.join(lines)

    return GAUSSIAN_INPUT_TEMPLATE.format(
                cpus=cpus,
                mem=mem,
                scheme=scheme,
                gaus_xtb_path=XTB_GAUSSIAN_FILE,
                chrg=chrg,
                spin=spin,
                ts_xyz=ts_xyz)


def write_ts_com_file(xyz_file,
                      chrg: int = 0,
                      spin: int = 1,
                      cpus: int = 1,
                      mem: float = 1.,
                      scheme: str = None):
    """ prepares com file for gaussian """
    if scheme is None:
        scheme = SCHEMES.get('opt_ts')

    com_file = xyz_file[:-4]+'_ts.com'
    with open(com_file, 'w') as f:
        f.write(get_gaussian_input_content(
            xyz_file=xyz_file,
            scheme=scheme,
            chrg=chrg,
            spin=spin,
            cpus=cpus,
            mem=mem,
        ))

    return com_file


def write_irc_com_file(xyz_file,
                       direction,
                       chrg: int = 0,
                       spin: int = 1,
                       cpus: int = 1,
                       mem: float = 1.,
                       scheme: str = None,):
    """ prepares com file for gaussian """
    if scheme is None:
        if direction == 'forward':
            scheme = SCHEMES.get('irc_forward')
        elif direction == 'reverse':
            scheme = SCHEMES.get('irc_reverse')

    com_file = xyz_file[:-4]+'_irc_'+str(direction)+'.com'
    with open(com_file, 'w') as f:
        f.write(get_gaussian_input_content(
            xyz_file=xyz_file,
            scheme=scheme,
            chrg=chrg,
            spin=spin,
            cpus=cpus,
            mem=mem,
        ))

    return com_file


def write_opt_com_file(xyz_file,
                       chrg: int = 0,
                       spin: int = 1,
                       cpus: int = 1,
                       mem: float = 1.,
                       scheme: str = None,):
    """ prepares com file for gaussian """
    if scheme is None:
        scheme = SCHEMES.get('opt')

    com_file = xyz_file[:-4]+'_opt.com'
    with open(com_file, 'w') as f:
        f.write(get_gaussian_input_content(
            xyz_file=xyz_file,
            scheme=scheme,
            chrg=chrg,
            spin=spin,
            cpus=cpus,
            mem=mem,
        ))

    return com_file


def write_constrained_opt_com_file(xyz_file,
                                   bond_pairs_changed: list,
                                   chrg: int = 0,
                                   spin: int = 1,
                                   cpus: int = 1,
                                   mem: float = 1.,
                                   scheme: str = None,):
    """
    This function prepares a Gaussian input file calculating forces from xTB for a constrained
    optimization freezing the bonds broken or created during the reaction.
    """
    if scheme is None:
        scheme = SCHEMES.get('opt_constraint')

    content = get_gaussian_input_content(
            xyz_file=xyz_file,
            scheme=scheme,
            chrg=chrg,
            spin=spin,
            cpus=cpus,
            mem=mem,
        )
    lines = content.splitlines()[:-1]  # remove the last empty line
    for bond_pair in bond_pairs_changed:
        lines.append(f'B {bond_pair[0]+1} {bond_pair[1]+1} F')

    lines.append(''); lines.append('')  # Gaussian input ends with two empty lines

    com_file = xyz_file[:-4]+'_opt.com'
    with open(com_file, 'w') as f:
        f.writelines(lines)

    return com_file


def calc_gaussian(com_file_writer, *args):
    """ Do gaussian-xtb ts optimization """
    com_file = com_file_writer(*args)
    output = run_cmd(f"srun {GAUSSIAN_CMD} {com_file}")
    with open(com_file[:-4]+'.out', 'w') as f:
        f.write(output)

    return com_file[:-4]+'.out'


def get_frequencies(out_file):
    """
    This function extracts the calculated normal modes from a frequency
    calculation with the corresponding frequencies including the optimized
    geometry
    """
    glog = GaussianLog(out_file)

    frequencies = glog.freqs
    vibration_matrices = list(glog.cclib_results.vibdisps)
    coordinates = glog.converged_geometries[-1]

    return frequencies, vibration_matrices, coordinates

def check_imaginary_frequencies(frequencies,
                                vibration_matrices,
                                bond_breaking_pairs,
                                coordinates,):
    """
    This function checks imaginary frequencies by projecting them onto each of
    the atom pairs that have bonds being formed or broken.
    """
    n_atoms = coordinates.shape[0]

    bond_matrices = []
    n_imag_freqs = np.sum(frequencies < 0)
    print(f"Number of imaginary frequencies = {n_imag_freqs}")
    if n_imag_freqs == 0:
        print("No imaginary Frequencies")
        lowest_freq_active = None
    else:
        for pair in bond_breaking_pairs:
            atom_1_coord, atom_2_coord = [coordinates[atom, :] for atom in pair]
            transition_direction = atom_2_coord - atom_1_coord
            transition_matrix = np.zeros((n_atoms, 3))
            transition_matrix[pair[0], :] = transition_direction
            transition_matrix[pair[1], :] = -transition_direction
            bond_matrices.append(transition_matrix)

        for i in range(n_imag_freqs):
            if i == 0:
                lowest_freq_active = 0
            print(f"transition: {i+1}")
            frequency_vector = np.ravel(vibration_matrices[i])
            for count, bond_matrix in enumerate(bond_matrices):
                transition_vector = np.ravel(bond_matrix)
                overlap = \
                (transition_vector/np.linalg.norm(transition_vector)) @ frequency_vector
                print(bond_breaking_pairs[count], overlap)
                if abs(overlap) > 0.33:
                    print("Vibration along the bond")
                    if i == 0:
                        lowest_freq_active += 1
                else:
                    print("Vibration not along bond")
        print(f"Lowest imaginary frequency active along: {lowest_freq_active} bonds")

    return n_imag_freqs, lowest_freq_active


def extract_optimized_structure(out_file, n_atoms, atom_labels):
    """
    After waiting for the constrained optimization to finish, the
    resulting structure from the constrained optimization is
    extracted and saved as .xyz file ready for TS optimization.
    """
    optimized_xyz_file = out_file[:-4]+".xyz"
    optimized_energy = None
    with open(out_file, 'r') as ofile:
        line = ofile.readline()
        while line:
            if 'Recovered energy=' in line:
                optimized_energy = line.split()[2]
            if 'Standard orientation' in line or 'Input orientation' in line:
                coordinates = np.zeros((n_atoms, 3))
                for i in range(5):
                    line = ofile.readline()
                for i in range(n_atoms):
                    coordinates[i, :] = np.array(line.split()[-3:])
                    line = ofile.readline()
            line = ofile.readline()
    with open(optimized_xyz_file, 'w') as _file:
        _file.write(str(n_atoms)+'\n\n')
        for i in range(n_atoms):
            _file.write(atom_labels[i])
            for j in range(3):
                _file.write(' '+"{:.5f}".format(coordinates[i, j]))
            _file.write('\n')

    print(f"optimized energy ({out_file}) = {optimized_energy}")

    return optimized_xyz_file, optimized_energy


def chiral_tags(mol):
    """
    Tag methylene and methyl groups with a chiral tag priority defined
    from the atom index of the hydrogens
    """
    li_list = []
    smarts_ch2 = '[!#1][*]([#1])([#1])([!#1])'
    atom_sets = mol.GetSubstructMatches(RDKitMol.FromSmarts(smarts_ch2))
    for atoms in atom_sets:
        atoms = sorted(atoms[2:4])
        prioritized_H = atoms[-1]
        li_list.append(prioritized_H)
        mol.GetAtoms()[prioritized_H].SetAtomicNum(9)  # ? Hard coded by the original writer
    smarts_ch3 = '[!#1][*]([#1])([#1])([#1])'
    atom_sets = mol.GetSubstructMatches(RDKitMol.FromSmarts(smarts_ch3))
    for atoms in atom_sets:
        atoms = sorted(atoms[2:])
        H1 = atoms[-1]
        H2 = atoms[-2]
        li_list.append(H1)
        li_list.append(H2)
        mol.GetAtoms()[H1].SetAtomicNum(9)  # ? Hard coded by the original writer
        mol.GetAtoms()[H2].SetAtomicNum(9)  # ? Hard coded by the original writer

    Chem.AssignAtomChiralTagsFromStructure(mol._mol, -1)
    Chem.rdmolops.AssignStereochemistry(mol._mol)
    for atom_idx in li_list:
        mol.GetAtoms()[atom_idx].SetAtomicNum(1)

    return mol


def choose_resonance_structure(mol):
    """
    This function creates all resonance structures of the mol object, counts
    the number of rotatable bonds for each structure and chooses the one with
    fewest rotatable bonds (most 'locked' structure)
    """
    resonance_mols = Chem.rdchem.ResonanceMolSupplier(mol._mol,
                                                      Chem.rdchem.ResonanceFlags.ALLOW_CHARGE_SEPARATION)
    res_status = True
    new_mol = None
    if not resonance_mols:
        print("using input mol")
        new_mol = mol
        res_status = False
    for res_mol in resonance_mols:
        Chem.SanitizeMol(res_mol)
        n_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(res_mol)
        if new_mol is None:
            smallest_rot_bonds = n_rot_bonds
            new_mol = RDKitMol(res_mol)
        if n_rot_bonds < smallest_rot_bonds:
            smallest_rot_bonds = n_rot_bonds
            new_mol = RDKitMol(res_mol)

    Chem.DetectBondStereochemistry(new_mol._mol, -1)
    Chem.rdmolops.AssignStereochemistry(new_mol._mol,
                                        flagPossibleStereoCenters=True,
                                        force=True)
    Chem.AssignAtomChiralTagsFromStructure(new_mol._mol, -1)
    return new_mol, res_status


def extract_smiles(xyz_file, charge, allow_charge=True, backend='openbabel'):
    """
    uses xyz2mol to extract smiles with as much 3d structural information as
    possible
    """
    with open(xyz_file, "r") as f:
        xyz = f.read()
    input_mol = RDKitMol.FromXYZ(xyz, backend=backend)
    # atoms, _, xyz_coordinates = xyz2mol_local.read_xyz_file(xyz_file)
    # try:
    #     input_mol = xyz2mol_local.xyz2mol(atoms, xyz_coordinates, charge=charge,
    #                                       use_graph=True,
    #                                       allow_charged_fragments=allow_charge,
    #                                       use_huckel=True, use_atom_maps=True,
    #                                       embed_chiral=True)
    # except:
    #     input_mol = xyz2mol_local.xyz2mol(atoms, xyz_coordinates, charge=charge,
    #                                       use_graph=True,
    #                                       allow_charged_fragments=allow_charge,
    #                                       use_huckel=False, use_atom_maps=True,
    #                                       embed_chiral=True)

    structure_mol, res_status = choose_resonance_structure(input_mol)
    structure_mol = chiral_tags(structure_mol)
    Chem.rdmolops.AssignStereochemistry(structure_mol._mol)
    structure_smiles = structure_mol.ToSmiles()

    return structure_smiles, structure_mol.GetFormalCharge(), res_status



def get_smiles(xyz_file, charge):
    """
    Try different things to extract sensible smiles using xyz2mol
    """
    print(xyz_file)
    try:
        smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                           allow_charge=True)
        print(smiles)
        if formal_charge != charge:
            smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                               allow_charge=False)
    except:
        try:
            smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                               allow_charge=False)
        except:
            return None, None, None

    return smiles, formal_charge, res_status

def smiles_to_mol(smiles):
    mol = RDKitMol.FromSmiles(smiles, sanitize=False)
    return mol

def is_ts_correct(rsmi, psmi, irc_start_xyz, irc_end_xyz):
    """
    This function compares the input smiles with the smiles of the endpoints of
    the IRC.
    """
    print(rsmi, psmi)
    rmol = smiles_to_mol(rsmi)
    pmol = smiles_to_mol(psmi)

    charge = GetFormalCharge(rmol)

    ts_found = False

    #doing smiles check
    irc_start_smi, _, _ = get_smiles(irc_start_xyz, charge)
    print("reverse SMILES: ", irc_start_smi)
    irc_end_smi, _, _ = get_smiles(irc_end_xyz, charge)
    print("forward smiles: ", irc_end_smi)
    if irc_start_smi == rsmi and irc_end_smi == psmi:
        ts_found = True
        print("SMILES MATCH: TS FOUND: reactant = reverse")

    if irc_start_smi == psmi and irc_end_smi == rsmi:
        ts_found = True
        print("SMILES MATCH: TS FOUND: reactant = forward")


    #doing AC check
    r_ac = rdmolops.GetAdjacencyMatrix(rmol)
    p_ac = rdmolops.GetAdjacencyMatrix(pmol)

    irc_start_mol = smiles_to_mol(irc_start_smi)
    irc_end_mol = smiles_to_mol(irc_end_smi)

    irc_start_ac = rdmolops.GetAdjacencyMatrix(irc_start_mol)
    irc_end_ac = rdmolops.GetAdjacencyMatrix(irc_end_mol)

    if np.all(irc_start_ac == irc_end_ac):
        print("found TS for conformational change")
    else:
        print("found non-coonformational change")

    if np.all(r_ac == irc_start_ac) and np.all(p_ac == irc_end_ac):
        print("AC MATCH: reactant = reverse")
    if np.all(p_ac == irc_start_ac) and np.all(r_ac == irc_end_ac):
        print("AC MATCH: reactant = forward")

    return ts_found


def check_ts(ts_guess, rsmi, psmi, n_atoms, atom_numbers, bond_pairs_changed, cpus, mem):
    """
    do TS optimization using xTB gradients with Gaussians optimizer
    """

    ts_out = calc_gaussian(write_ts_com_file, ts_guess, cpus, mem)
    frequencies, vibrations, coordinates = get_frequencies(ts_out, n_atoms)
    n_imag_freq, lowest_vibration_active = check_imaginary_frequencies(frequencies,
                                                                       vibrations,
                                                                       bond_pairs_changed,
                                                                       coordinates, n_atoms)
    print("# of imaginary frequencies =", n_imag_freq)
    if n_imag_freq != 1:
        print("ERROR: TS not found - check # of imaginary freqs!")
        return False
    print("lowest imag. freq =", frequencies[0])

    ts_xyz, ts_energy = extract_optimized_structure(ts_out, n_atoms,
                                                    atom_numbers)
    forward_irc_out = calc_gaussian(write_irc_com_file, ts_xyz, 'forward', cpus, mem)
    reverse_irc_out = calc_gaussian(write_irc_com_file, ts_xyz, 'reverse', cpus, mem)

    forward_irc_xyz, _ = extract_optimized_structure(forward_irc_out, n_atoms,
                                                     atom_numbers)
    reverse_irc_xyz, _ = extract_optimized_structure(reverse_irc_out, n_atoms,
                                                     atom_numbers)
    ts_found = is_ts_correct(rsmi, psmi, reverse_irc_xyz, forward_irc_xyz)

    if ts_found:
        os.chdir('../')
        return ts_found

    forward_opt_out = calc_gaussian(write_opt_com_file, forward_irc_xyz,
                                    cpus, mem)
    reverse_opt_out = calc_gaussian(write_opt_com_file, reverse_irc_xyz,
                                    cpus, mem)
    forward_opt_xyz, _ = extract_optimized_structure(forward_opt_out, n_atoms,
                                                     atom_numbers)
    reverse_opt_xyz, _ = extract_optimized_structure(reverse_opt_out, n_atoms,
                                                     atom_numbers)

    ts_found = is_ts_correct(rsmi, psmi, reverse_opt_xyz, forward_opt_xyz)

    return ts_found

if __name__ == "__main__":
    TS_GUESS = os.path.basename(sys.argv[1])
    RSMI = sys.argv[2]
    PSMI = sys.argv[3]
    CPUS = sys.argv[4]
    MEM = sys.argv[5]
    os.mkdir("ts_test_xtb")
    shutil.copy(TS_GUESS, "ts_test_xtb")
    os.chdir("ts_test_xtb")
    N_ATOMS, ATOM_NUMBERS = atom_information(TS_GUESS)
    BOND_PAIRS_CHANGED = bonds_getting_formed_or_broken(RSMI, PSMI, N_ATOMS)

    TS_FOUND = check_ts(TS_GUESS, RSMI, PSMI, N_ATOMS, ATOM_NUMBERS,
                        BOND_PAIRS_CHANGED, CPUS, MEM)
    if not TS_FOUND:
        print("---------- do constrained optimization ------------")
        CONSTRAINED_OUT = calc_gaussian(write_constrained_opt_com_file,
                                        TS_GUESS, BOND_PAIRS_CHANGED, CPUS,
                                        MEM)
        CONSTRAINED_XYZ, _ = extract_optimized_structure(CONSTRAINED_OUT,
                                                         N_ATOMS, ATOM_NUMBERS)
        TS_FOUND = check_ts(CONSTRAINED_XYZ, RSMI, PSMI, N_ATOMS, ATOM_NUMBERS,
                            BOND_PAIRS_CHANGED, CPUS, MEM)