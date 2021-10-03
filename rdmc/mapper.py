#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provide method for atom mapping
"""

import rdkit
from rdkit import Chem
import os
import re
import copy
import warnings

RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()
def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:
    """
    Generate Chem.Mol including explicit or implicit hydrogen atoms
    
    :param string: A SMILES or InChI string.
    :param explicit_hydrogens: Whether the explicit is added.
    
    :return: A RDKit Chem.Mol object.
    """
    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        # Set params here so we don't remove hydrogens with atom mapping
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        #return mol
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)
    
def nb_H_Num(atom: Chem.Atom) -> int:
    """
    Calculating the number of hydroegn atom around any atom
    Note: With the SMILES, for example, [C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]
          RDKit Chem.atom.GetNumExplicitHs() cann't count the Hs correctly
    
    :param string: A SMILES or InChI string.
    :param atom: A Chem.atom object
    
    :return: A RDKit Chem.Mol object.
    """
    nb_Hs = 0
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() == 'H':
            nb_Hs = nb_Hs + 1
    return(nb_Hs)

def mol_info(mol: Chem.Mol):
    """
    Generating molecular informations including aotm map numberm, symbol, atom index and neighbor Hs number.
    
    :param mol: A Chem.Mol object.
    
    :return: Two lists including heavy atom infromation and the information of hrdrogen atoms which do not
             connected with heavy atoms.
    """
    map_list = []
    Hmap_list = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            if atom.GetAtomMapNum() ==0:
                raise Exception("Find a heavy atom without atom map number !!!")
            else:
                map_list.append([atom.GetAtomMapNum(), atom.GetSymbol(), nb_H_Num(atom), atom.GetIdx()])
        else:
            atom.SetAtomMapNum(0) # We just focus on whether heavy atoms are matched correctly
            if nb_H_Num(atom) == 0 and len(atom.GetNeighbors()) != 0:
                continue
            else:
                Hmap_list.append([atom.GetAtomMapNum(), atom.GetSymbol(), nb_H_Num(atom), atom.GetIdx()])
    return sorted(map_list, key=lambda x: x[0]), Hmap_list

def Reorder_AtomMapNum(rsmiles:str,
                       psmiles:str,
                       start_num: int = 1):
    """
    This function is to reorder the atom map number
    
    :param rsmiles: The reactant(s) smiles
    :param psmiles: The product(s) smiles
    :param star_num: The start atom number, default = 1
    """
    r_nums = re.findall(r"\d+", rsmiles)
    replace_nums = [[str(i+start_num), num] for i, num in enumerate(r_nums)]
    for replace_num in replace_nums:
        rsmiles = rsmiles.replace(":"+replace_num[1] + ']', ":r" + replace_num[0] + ']')
    rsmiles = rsmiles.replace(":r", ":")
    for replace_num in replace_nums:
        psmiles = psmiles.replace(":"+replace_num[1] + ']', ":r" + replace_num[0] + ']')
    psmiles= psmiles.replace(":r", ":")
    
    return rsmiles, psmiles

def match_Hs(r_info:list,
             p_info:list,
             r_sep_Hs_info:list,
             p_sep_Hs_info:list,
             r_mol:Chem.Mol,
             p_mol:Chem.Mol):
    """
    This function is to match the Hs of reactant(s) and product(s)
    
    :param r_info: Heavy atom information of reactant(s).
    :param p_info: Heavy atom information of product(s).
    :param r_sep_Hs_info: The information of reactant(s) hrdrogen atoms which do not connected with heavy atoms
    :param p_sep_Hs_info: The information of product(s) hrdrogen atoms which do not connected with heavy atoms
    :param r_mol: A Chem.Mol object.
    :param r_mol: A Chem.Mol object.
    
    :return: The atom mapped reactant(s) and product(s) SMILES
    """
    # Check the reactant and product. 
    Hnum_count = []
    if len(r_info) !=len(p_info):
        warnings.warn("Different heavy atom number between reactants and products!")
    for r_atom in r_info:
        for p_atom in p_info:
            if r_atom[0] == p_atom[0] and r_atom[1] == p_atom[1]:
                Hnum_count.append([r_atom[0],r_atom[2],p_atom[2]])
                break
        else:
            raise Exception("Because of the absence of same atoms with same atom map number,reactant smiles doesn't match product smiles")
    # Select the heavy atoms with H migration
    heavy_atoms_with_Hmig = [Hnum_count[i] for i, atom_pair in enumerate(Hnum_count) if atom_pair[1] != atom_pair[2]]
    
    #For 2D model, the hydrogen atoms connected to the same heavy atom are the same
    heavy_atoms_Hs_change = [[a[0], a[2]-a[1]] for a in heavy_atoms_with_Hmig]
    Hs_number = [a[1] for a in heavy_atoms_Hs_change]
    Sep_Hs = False
    if sum(Hs_number) !=0:
        Sep_Hs = True # If not balanced H for heavy atoms, H migrated from/to heavy atoms
        warnings.warn("Some hydrogen atoms might migrate from/to [H] or [H][H]")
    elif r_sep_Hs_info or p_sep_Hs_info:
        Sep_Hs = True # If not balanced H for heavy atoms, H migrated from/to heavy atoms
        warnings.warn("Some hydrogen atoms might migrate from/to [H] or [H][H]!")
    if sum(Hs_number) + len(p_sep_Hs_info) - len(r_sep_Hs_info) != 0:
        raise Exception("Unbalanced Hs!")
    # We obtained a list [heavy_atom_map_num, Hs_num_obtained]
    # Initializing the r_mol
    counter = max([a.GetAtomMapNum() for a in r_mol.GetAtoms()]) + 1
    r_Hs_map_num = {} # atom_map_number_of_heavy_atoms: [atom_map_number_of_Hs]
    for atom_info in r_info:
        heavy_atom = r_mol.GetAtomWithIdx(atom_info[3])#atom_info[3] is atom index
        r_Hs_map_num[atom_info[0]] = [] #atom_info[0] is atom map number
        for nb_atom in heavy_atom.GetNeighbors():
            if nb_atom.GetSymbol() == "H" and nb_atom.GetAtomMapNum() == 0:
                nb_atom.SetAtomMapNum(counter)
                r_Hs_map_num[atom_info[0]].append(counter)
                counter +=1
    r_sep_Hs_map_num = {} # atom_idx_of_H: aotm_map_number_of_H
    if len(r_sep_Hs_info) != 0:
        for r_sep_H_info in r_sep_Hs_info:
            H_atom = r_mol.GetAtomWithIdx(r_sep_H_info[3])
            H_atom.SetAtomMapNum(counter)
            r_sep_Hs_map_num[r_sep_H_info[3]] = counter
            counter +=1
    
    # Assign atom map number for changed Hs
    # We asusume that the seperated H atoms or H2 would change H atoms with the heavy atoms.
    # If Sep_Hs = True(unbalanced Hs of heavy atoms) and H_change < 0. H migrate from heavy atoms to form H2
    # f Sep_Hs = True(unbalanced Hs of heavy atoms) and H_change > 0. H migrate to heavy atoms to form [H]
    p_Hs_map_num = copy.deepcopy(r_Hs_map_num)
    collect_changed_H_num = []
    if Sep_Hs is True:
        collect_sep_H_num = [num for num in r_sep_Hs_map_num.values()]
    for H_change in heavy_atoms_Hs_change:
        if H_change[1] < 0: # H_change[1]: the number of H atom changed
            collect_changed_H_num.extend(r_Hs_map_num[H_change[0]][0: -H_change[1]]) #H_change[0]: the heavy atom num of H changed
            del p_Hs_map_num[H_change[0]][0: -H_change[1]]
    if Sep_Hs is True:
        if len(p_sep_Hs_info)-len(r_sep_Hs_info) > 0:
            collect_sep_H_num.extend(collect_changed_H_num[0: len(p_sep_Hs_info)-len(r_sep_Hs_info)])
            del collect_changed_H_num[0: len(p_sep_Hs_info)-len(r_sep_Hs_info)]
        else:
            collect_changed_H_num.extend(collect_sep_H_num[0: len(r_sep_Hs_info)-len(p_sep_Hs_info)])
            del collect_sep_H_num[0: len(p_sep_Hs_info)-len(r_sep_Hs_info)]
                   
    for H_change in heavy_atoms_Hs_change:
        if H_change[1] > 0:
            if collect_changed_H_num:
                p_Hs_map_num[H_change[0]].extend(collect_changed_H_num[0:H_change[1]])
                del collect_changed_H_num[0:H_change[1]]
            elif Sep_Hs is True and collect_sep_H_num:
                p_Hs_map_num[H_change[0]].extend(collect_sep_H_num[0:H_change[1]])
                del collect_sep_H_num[0:H_change[1]]
            else:
                raise Expect(" Unbalanced Hs during assigning the atom map number for products")
    for atom_info in p_info:
        heavy_atom = p_mol.GetAtomWithIdx(atom_info[3])
        nums = p_Hs_map_num[atom_info[0]]
        i = 0
        for nb in heavy_atom.GetNeighbors():
            if nb.GetSymbol() == 'H' and nb.GetAtomMapNum() == 0:
                nb.SetAtomMapNum(nums[i])
                i += 1
    if Sep_Hs is True and collect_sep_H_num:
        j = 0
        for atom in p_mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                if nb_H_Num(atom) == 0 and len(atom.GetNeighbors()) != 0:
                    continue
                else:
                    atom.SetAtomMapNum(collect_sep_H_num[j])
                    j += 1

    r_smiles = Chem.MolToSmiles(r_mol)
    p_smiles = Chem.MolToSmiles(p_mol)
    r_smiles_new, p_smiles_new = Reorder_AtomMapNum(r_smiles, p_smiles, start_num = 1)
    
    return str_to_mol(r_smiles_new), str_to_mol(p_smiles_new)

def map_reaction_Hs( rxn: str,
                     show_img: bool = True,
                     ):
    """
    This function is to match the Hs of a reaction with heavy atoms already mapped
    
    :param rxn: SMILES of reavtion, for example: react_SMILES>>prod_SMILES.
    :param show_img: Display image of reactants and products
    
    :return: The atom mapped reactant(s) and product(s) SMILES
    """
    reaction_mapped = rxn.split('>>')
    r_mol = str_to_mol(reaction_mapped[0])
    p_mol = str_to_mol(reaction_mapped[1])
    if show_img is True:
        display(r_mol)
        display(p_mol)
    r_info, r_sep_Hs_info = mol_info(r_mol)
    p_info, p_sep_Hs_info = mol_info(p_mol)
    r, p = match_Hs(r_info, p_info, r_sep_Hs_info, p_sep_Hs_info, r_mol, p_mol )
    if show_img is True:
        print("************After Mapping Hs***************")
        print('REACTANT(S):')
        display(r)
        print('PRODUCT(S):')
        display(p)
    return Chem.MolToSmiles(r), Chem.MolToSmiles(p)