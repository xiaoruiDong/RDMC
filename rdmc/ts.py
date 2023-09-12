#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with Transition states.
"""

from typing import List, Set, Tuple, Union

import numpy as np
from scipy import optimize
from scipy.spatial import distance

from rdmc import RDKitMol
from rdmc.mathlib.geom import get_centroid, get_max_distance_from_center, rotate, translate, translate_centroid
from rdmc.utils import PERIODIC_TABLE as PT


def _get_bonds_as_sets(*mols: Union['RDKitMol', 'Mol'],
                       ) -> Tuple[Set]:
    """
    Get the set of bonds for the provided list of mols.

    Args:
        mols ('RDKitMol' or 'Mol'): a RDKit Mol object

    Returns
        Tuple[Set]: (bond set in the reactant, bond set in the product)
    """
    return tuple(set(mol.GetBondsAsTuples()) for mol in mols)


def get_formed_bonds(r_mol: Union['RDKitMol', 'Mol'],
                     p_mol: Union['RDKitMol', 'Mol'],
                     ) -> List:
    """
    Get all bonds formed in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns
        list: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    return list(p_bonds - r_bonds)


def get_broken_bonds(r_mol: Union['RDKitMol', 'Mol'],
                     p_mol: Union['RDKitMol', 'Mol'],
                     ) -> List:
    """
    Get all bonds broken in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    return list(r_bonds - p_bonds)


def get_formed_and_broken_bonds(r_mol: Union['RDKitMol', 'Mol'],
                                p_mol: Union['RDKitMol', 'Mol'],
                                ) -> List:
    """
    Get all bonds broken in the reaction. Both reactant and product complexes
    need to be atom-mapped. This function doesn't count bonds whose bond order
    is lowered but not equal to zero.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: - formed bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
              - broken bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    return (list(p_bonds - r_bonds),
            list(r_bonds - p_bonds))


def get_all_changing_bonds(r_mol: Union['RDKitMol', 'Mol'],
                           p_mol: Union['RDKitMol', 'Mol'],
                           ) -> List:
    """
    Get all bonds changed in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: - formed bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
              - broken bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
              - bonds with BO changed: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    formed_bonds, broken_bonds = p_bonds - r_bonds, r_bonds - p_bonds
    changed_bonds = [bond for bond in (r_bonds & p_bonds)
                     if (r_mol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble()
                         != p_mol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble())]
    return list(formed_bonds), list(broken_bonds), changed_bonds


def clean_ts(r_mol: 'RDKitMol',
             p_mol: 'RDKitMol',
             ts_mol: 'RDKitMol'):
    """
    Cleans transition state ``ts_mol`` by removing all bonds that correspond to broken or formed bonds.
    ``r_mol``, ``p_mol``, and ``ts_mol`` need to be atom mapped. Bond order changes are not considered.

    Args:
        r_mol (RDKitMol): the reactant complex.
        p_mol (RDKitMol): the product complex.
        ts_mol (RDKitMol): the transition state corresponding to ``r_mol`` and ``p_mol``.

    Returns:
        RDKitMol: an edited version of ``ts_mol``, which is the original ``ts_mol`` with cleaned bonding.
        list:

            - broken bonds: A list of length-2 tuples that contains the atom indexes of the bonds broken in the reaction.
            - formed bonds: A list of length-2 tuples that contains the atom indexes of the bonds formed in the reaction.
    """
    r_bonds, p_bonds, ts_bonds = _get_bonds_as_sets(r_mol, p_mol, ts_mol)
    formed_bonds, broken_bonds = p_bonds - r_bonds, r_bonds - p_bonds

    edited_ts_mol = ts_mol.Copy()
    edit_bonds = ts_bonds - (r_bonds & p_bonds)  # bonds in ts but not in r_mol and p_mol simultaneously
    for bond_idxs in edit_bonds:
        edited_ts_mol.RemoveBond(*bond_idxs)
        edited_ts_mol.AddBond(*bond_idxs)

    return edited_ts_mol, list(broken_bonds), list(formed_bonds)


def is_DA_rxn_endo(r_mol: 'RDKitMol',
                   p_mol: 'RDKitMol',
                   embed: bool = False):
    """
    Determine the Diels Alder reaction stereo type (endo or exo),
    based on the provided reactants and products.

    Args:
        r_mol (RDKitMol): the reactant complex.
        p_mol (RDKitMol): the product complex.
        embed (bool): bool. If the DA product has no conformer embedded.
                            Whether to embed a conformer. Defaults to ``False``.
    """
    frags = r_mol.GetMolFrags()

    if len(frags) == 1:
        # This reaction is defined in the reverse direction:
        # DA_product <=> diene + dienophile
        r_mol, p_mol = p_mol, r_mol
        frags = r_mol.GetMolFrags()

    assert len(frags) == 2

    if p_mol.GetNumConformers() == 0 and embed:
        p_mol.EmbedConformer()
    elif p_mol.GetNumConformers() == 0:
        raise ValueError('The provided DA product has no geometry available'
                         'Cannot determine the stereotype of the DA reaction')

    # Analyze the reaction center
    formed, _, changing = get_all_changing_bonds(r_mol, p_mol)
    # `fbond_atoms` are atoms in the formed bonds
    fbond_atoms = set([atom for bond in formed for atom in bond])
    for bond in changing:
        bond = set(bond)
        if len(bond & fbond_atoms) == 0:
            # Find the single bond in the diene
            dien_sb = list(bond)
        elif len(bond & fbond_atoms) == 2:
            # Find the double bond of the dienophile
            dinp_db = list(bond)
    # Make `fbond_atoms` for convenience in slicing
    fbond_atoms = list(fbond_atoms)

    # Find the atom indexes in diene and dienophile
    _, dienophile = frags if dien_sb[0] in frags[0] else frags[::-1]

    # Get the 3D geometry of the DA product
    # Create a reference plane from atoms in formed bonds
    # The reference point is chosen to be the center of the plane
    xyz = p_mol.GetPositions()
    ref_plane = xyz[fbond_atoms]
    ref_pt = ref_plane.mean(axis=0, keepdims=True)

    # Use the total least square algorithm to find
    # the normal vector of the reference plane
    A = ref_plane - ref_pt
    norm_vec = np.linalg.svd(A.T @ A)[0][:, -1].reshape(1, -1)

    # Use the vector between middle point of the diene single point
    # and the reference point as one direction vector
    dien_vec = xyz[dien_sb, :].mean(axis=0, keepdims=True) - ref_pt

    # Use the vector between mass center of the dienophile
    # and the reference point as one direction vector
    # excluding atom in dienophile's double bond
    atom_scope = [atom for atom in dienophile if atom not in dinp_db]
    mass_vec = [r_mol.GetAtomWithIdx(i).GetMass() for i in atom_scope]
    wt_xyz = (xyz[atom_scope, :] * np.reshape(mass_vec, (-1, 1)))
    dinp_vec = wt_xyz.mean(axis=0, keepdims=True) - ref_pt

    # Endo is determined by if dien_vec has the same direction as the dinp_vec
    # using the normal vector of the reference plane as a reference direction
    endo = ((norm_vec @ dien_vec.T) * (norm_vec @ dinp_vec.T)).item() > 0
    return endo


def guess_rxn_from_normal_mode(xyz: np.array,
                               symbols: np.array,
                               disp: np.array,
                               amplitude: Union[float, list] = 0.25,
                               weights: Union[bool, np.array] = True,
                               backend: str = 'openbabel',
                               multiplicity: int = 1):
    """
    Guess reaction according to the normal mode analysis for a transition state.

    Args:
        xyz (np.array): The xyz coordinates of the transition state. It should have a
                        size of :math:`N \\times 3`.
        symbols (np.array): The symbols of each atoms. It should have a size of :math:`N`.
        disp (np.array): The displacement of the normal mode. It should have a size of
                         :math:`N \\times 3`.
        amplitude (float): The amplitude of the motion. If a single value is provided then the guess
                           will be unique (if available). ``0.25`` will be the default. Otherwise, a list
                           can be provided, and all possible results will be returned.
        weights (bool or np.array): If ``True``, use the :math:`\\sqrt(atom mass)` as a scaling factor to the displacement.
                                    If ``False``, use the identity weights. If a :math:`N \\times 1` ``np.ndarray`` is provided,
                                    then use the provided weights. The concern is that light atoms (e.g., H)
                                    tend to have larger motions than heavier atoms.
        backend (str): The backend used to perceive xyz. Defaults to ``'openbabel'``.
        multiplicity (int): The spin multiplicity of the transition states. Defaults to ``1``.

    Returns:
        list: Potential reactants
        list: Potential products
    """
    if isinstance(amplitude, float):
        amplitude = [amplitude]

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array([PT.GetAtomicWeight(PT.GetAtomicNumber(symbol)) for symbol in symbols]).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((xyz.shape[0], 1))

    mols = {'r': [], 'p': []}
    for amp in amplitude:
        xyzs = xyz - amp * disp * weights, xyz + amp * disp * weights

        # Create the reactant complex and the product complex
        for xyz_mod, side in zip(xyzs, ['r', 'p']):
            xyz_str = ''
            for symbol, coords in zip(symbols, xyz_mod):
                xyz_str += f'{symbol:4}{coords[0]:14.8f}{coords[1]:14.8f}{coords[2]:14.8f}\n'
            try:
                mols[side].append(RDKitMol.FromXYZ(xyz_str, header=False, backend=backend))
                mols[side][-1].SaturateMol(multiplicity=multiplicity)
            except BaseException:
                # Need to provide a more precise error in the future
                # Cannot generate the molecule from XYZ
                pass

    # Pairwise isomorphic comparison
    for side, ms in mols.items():
        prune = []
        len_mol = len(ms)
        for i in range(len_mol):
            if i in prune:
                continue
            for j in range(i + 1, len_mol):
                if j not in prune and ms[i].GetSubstructMatch(ms[j]):
                    prune.append(j)
        mols[side] = [mol for i, mol in enumerate(ms) if i not in prune]

    return tuple(mols.values())


def examine_normal_mode(r_mol: RDKitMol,
                        p_mol: RDKitMol,
                        ts_xyz: np.array,
                        disp: np.array,
                        amplitude: Union[float, list] = 0.25,
                        weights: Union[bool, np.array] = True,
                        verbose: bool = True,
                        as_factors: bool = True):
    """
    Examine a TS's imaginary frequency given a known reactant complex and a
    product complex. The function checks if the bond changes are corresponding
    to the most significant change in the normal mode. The reactant and product
    complex need to be atom mapped.

    Args:
        r_mol ('RDKitMol'): the reactant complex.
        p_mol ('RDKitMol'): the product complex.
        ts_xyz (np.array): The xyz coordinates of the transition state. It should have a
                           size of N x 3.
        disp (np.array): The displacement of the normal mode. It should have a size of
                         N x 3.
        amplitude (float): The amplitude of the motion. Defaults to ``0.25``.
        weights (bool or np.array): If ``True``, use the sqrt(atom mass) as a scaling factor to the displacement.
                                    If ``False``, use the identity weights. If a N x 1 ``np.array`` is provided, then
                                    The concern is that light atoms (e.g., H) tend to have larger motions
                                    than heavier atoms.
        verbose (bool): If print detailed information. Defaults to ``True``.
        as_factors (bool): If return the value of factors instead of a judgment.
                           Defaults to ``False``

    Returns:
        - bool: ``True`` for pass the examination, ``False`` otherwise.
        - list: If `as_factors == True`, two factors will be returned.
    """
    # Analyze connectivity
    broken, formed, changed = get_all_changing_bonds(r_mol, p_mol)
    reacting_bonds = broken + formed + changed

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array(r_mol.GetAtomMasses()).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((ts_xyz.shape[0], 1))

    # Generate conformer instance according to the displacement
    xyzs = ts_xyz - amplitude * disp * weights, ts_xyz + amplitude * disp * weights
    r_copy = r_mol.Copy()
    r_copy.SetPositions(xyzs[0])
    p_copy = p_mol.Copy()
    p_copy.SetPositions(xyzs[1])
    r_conf, p_conf = r_copy.GetConformer(), p_copy.GetConformer()

    # Calculate bond distance change
    formed_and_broken_diff = [abs(r_conf.GetBondLength(bond) - p_conf.GetBondLength(bond))
                              for bond in broken + formed]
    changed_diff = [abs(r_conf.GetBondLength(bond) - p_conf.GetBondLength(bond))
                    for bond in changed]
    other_bonds_diff = [abs(r_conf.GetBondLength(bond) - p_conf.GetBondLength(bond))
                        for bond in r_copy.GetBondsAsTuples() if bond not in reacting_bonds]

    # We expect bonds that are formed or broken in the reaction
    # have relatively large changes; For bonds that change their bond order
    # in the reaction may have a smaller factor.
    # In this function, we only use the larger factor as a check.
    # The smaller factor is less deterministic, considering the change in
    # other bonds due to the change of atom hybridization or bond conjugation.
    baseline = np.max(other_bonds_diff)
    std = np.std(other_bonds_diff)
    larger_factor = (np.min(formed_and_broken_diff) - baseline) / std
    if changed_diff:
        # There might be no bond that only changes its order
        smaller_factor = (np.min(changed_diff) - baseline) / std
    else:
        smaller_factor = 0

    if verbose:
        print(f'The min. bond distance change for bonds that are broken or formed'
              f' is {np.min(formed_and_broken_diff)} A and is {larger_factor:.1f} STD off the baseline.')
        if changed_diff:
            print(f'The min. bond distance change for bonds that are changed'
                  f' is {np.min(changed_diff)} A and is {smaller_factor:.1f} STD off the baseline.')

    if as_factors:
        return larger_factor, smaller_factor

    if larger_factor > 3:
        return True
    return False
