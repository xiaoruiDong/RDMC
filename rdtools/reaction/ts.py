from copy import copy
from typing import Union, Sequence, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT

from rdtools.bond import get_formed_and_broken_bonds, get_all_changing_bonds, get_bonds_as_tuples
from rdtools.conversion.xyz import mol_from_xyz
from rdtools.conf import add_conformer, set_conformer_coordinates
from rdtools.compare import is_same_connectivity_mol
from rdtools.element import get_atom_mass, get_covalent_radius
from rdtools.fix import saturate_mol
from rdtools.mathlib import get_magnitude
from rdtools.mol import get_atom_masses


def guess_rxn_from_normal_mode(
    xyz: np.ndarray,
    symbols: Sequence[str],
    disp: np.ndarray,
    amplitude: Union[float, List[float]] = 0.25,
    weights: Union[bool, np.ndarray] = True,
    backend: str = 'openbabel',
    multiplicity: int = 1
) -> Tuple[list, list]:
    """
    Guess reaction according to the normal mode analysis for a transition state.

    Args:
        xyz (np.ndarray): The xyz coordinates of the transition state. It should have a
                         size of :math:`N \\times 3`. You can get it from mol.GetConformer().GetPositions().
        symbols (Sequence): The symbols of each atoms. It should have a size of :math:`N`. You can get the symbols
                            by rdmc.rdtools.mol.get_element_symbols.
        disp (np.ndarray): The displacement of the normal mode. It should have a size of
                           :math:`N \\times 3`. You can get it from cclib.vib
        amplitude (float or List[float]): The amplitude of the motion. If a single value is provided then the guess
                           will be unique (if available). ``0.25`` will be the default. Otherwise, a list
                           can be provided, and all possible results will be returned.
        weights (bool or np.ndarray): If ``True``, use the :math:`\\sqrt(atom mass)` as a scaling factor to the displacement.
                                    If ``False``, use the identity weights. If a :math:`N \\times 1` ``np.ndarray`` is provided,
                                    then use the provided weights. The concern is that light atoms (e.g., H)
                                    tend to have larger motions than heavier atoms.
        backend (str): The backend used to perceive xyz. Defaults to ``'openbabel'``.
        multiplicity (int): The spin multiplicity of the transition states. Defaults to ``1``.

    Returns:
        list: Potential reactants
        list: Potential products
    """
    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array([get_atom_mass(symbol) for symbol in symbols]).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((xyz.shape[0], 1))

    mols = {'r': [], 'p': []}
    amplitude = [amplitude] if isinstance(amplitude, float) else amplitude
    for amp in amplitude:
        xyzs = xyz - amp * disp * weights, xyz + amp * disp * weights

        # Create the reactant complex and the product complex
        for xyz_mod, side in zip(xyzs, ['r', 'p']):
            xyz_str = ''
            for symbol, coords in zip(symbols, xyz_mod):
                xyz_str += f'{symbol:4}{coords[0]:14.8f}{coords[1]:14.8f}{coords[2]:14.8f}\n'
            try:
                mols[side].append(mol_from_xyz(xyz_str, header=False, backend=backend))
                saturate_mol(mols[side][-1], multiplicity=multiplicity)
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


def examine_normal_mode(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
    ts_xyz: np.ndarray,
    disp: np.ndarray,
    amplitude: Union[float, List[float]] = 0.25,
    weights: Union[bool, np.ndarray] = True,
    verbose: bool = True,
    as_factors: bool = True,
):
    """
    Examine a TS's imaginary frequency given a known reactant complex and a
    product complex. The function checks if the bond changes are corresponding
    to the most significant change in the normal mode. The reactant and product
    complex need to be atom mapped.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.
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
    broken, formed, changed = get_all_changing_bonds(rmol, pmol)
    reacting_bonds = broken + formed + changed

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array(rmol.GetAtomMasses()).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((ts_xyz.shape[0], 1))

    # Generate conformer instance according to the displacement
    xyzs = ts_xyz - amplitude * disp * weights, ts_xyz + amplitude * disp * weights
    r_copy, p_copy = copy(rmol), copy(pmol)
    add_conformer(r_copy, coords=xyzs[0])
    add_conformer(p_copy, coords=xyzs[1])
    r_conf, p_conf = r_copy.GetConformer(), p_copy.GetConformer()

    # Calculate bond distance change
    formed_and_broken_diff = [
        abs(rdMT.GetBondLength(r_conf, *bond) - rdMT.GetBondLength(p_conf, *bond))
        for bond in broken + formed
    ]
    changed_diff = [
        abs(rdMT.GetBondLength(r_conf, *bond) - rdMT.GetBondLength(p_conf, *bond))
        for bond in changed
    ]
    other_bonds_diff = [
        abs(rdMT.GetBondLength(r_conf, *bond) - rdMT.GetBondLength(p_conf, *bond))
        for bond in (set(get_bonds_as_tuples(r_copy)) - set(reacting_bonds))
    ]

    # We expect bonds that are formed or broken in the reaction
    # have relatively large changes; For bonds that change their bond order
    # in the reaction may have a smaller factor.
    # In this function, we only use the larger factor as a check.
    # The smaller factor is less deterministic, considering the change in
    # other bonds due to the change of atom hybridization or bond conjugation.
    baseline = np.max(other_bonds_diff)
    std = np.std(other_bonds_diff)
    larger_factor = (np.min(formed_and_broken_diff) - baseline) / std
    # There might be no bond that only changes its order
    smaller_factor = (np.min(changed_diff) - baseline) / std if changed_diff else 0
    other_factor = baseline / std

    if verbose:
        print(
            f'The min. bond distance change for bonds that are broken or formed'
            f' is {np.min(formed_and_broken_diff)} A and is {larger_factor:.1f} STD off the baseline.'
        )
        if changed_diff:
            print(
                f'The min. bond distance change for bonds that are changed'
                f' is {np.min(changed_diff)} A and is {smaller_factor:.1f} STD off the baseline.'
            )

    if as_factors:
        return larger_factor, smaller_factor, other_factor

    if larger_factor > 3:
        return True

    return False


def is_valid_habs_ts(
    rmol: Chem.Mol
    pmol: Chem.Mol,
    ts_xyz: str,
    disp: np.ndarray,
    single_score: bool = True,
    **kwargs,
):
    """
    Check if the TS belonging to a H atom abstraction reactions. This function assumes the TS, rmol,
    and pmol have the same atom order.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.
        ts_xyz (str): The xyz coordinates of the transition state.
        disp (np.array): The displacement of the normal mode. It should have a size of
            N x 3.
        single_score (bool): If return a single score (True or False). Defaults to ``True``.
            Otherwise, return the check results (True or False) for both side of the reaction

    Returns:
        - bool: ``True`` for pass the examination, ``False`` otherwise.
        - tuple: if the reactant side or the product side pass the examination.
    """

    def get_well_xyz(
        atom1_idx: int,
        atom2_idx: int,
        positions: np.ndarray,
        freq_modes: np.ndarray,
        target: float,
    ) -> np.ndarray:
        """
        Adjust the position by the frequency mode, so that the distance between
        the H atom and the bonding atom is equal to the target value.

        Args:
            atom1_idx: atom index of the H atom or its bonding atom
            atom2_idx: atom index of the other atom
            positions: xyz coordinates of the atoms of the TS
            freq_modes: frequency modes of the imaginary frequency
            target: target distance between the H atom and the bonding atom

        Returns:
            np.ndarray: new xyz coordinates of the atoms
        """
        x_AB = positions[atom1_idx] - positions[atom2_idx]
        x_dAB = freq_modes[atom1_idx] - freq_modes[atom2_idx]
        m1, m2 = get_magnitude(x_AB, x_dAB, target)
        magnitude = m1 if abs(m1) < abs(m2) else m2
        return positions - freq_modes * magnitude


    def check_bond_in_mult_three_member_ring(mol: Chem.Mol, bond: tuple):
        """
        Check if the bond is in multiple three member rings. Perception algorithm
        currently is confused at this condition.

        Args:
            mol (Chem.Mol): the molecule
            bond (tuple): the bond

        Returns:
            bool: ``True`` if the bond is in multiple three member rings, ``False`` otherwise
        """
        try:
            bond_idx = mol.GetBondBetweenAtoms(int(bond[0]), int(bond[1])).GetIdx()
        except:
            return False
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        membership = ring_info.BondMembers(bond_idx)
        num_in_three_member_ring = sum([ring_sizes[mem] == 3 for mem in membership])
        return num_in_three_member_ring >= 2


    formed, broken = get_formed_and_broken_bonds(rmol, pmol)
    assert len(formed) == 1 and len(broken) == 1, "The TS should be b1f1."
    formed, broken = set(formed), set(broken)
    H_index = list(formed & broken)[0]
    atom1_idx = list(formed - {H_index})[0]  # the radical site in reactant
    atom2_idx = list(broken - {H_index})[0]  # the radical site in product

    mol = mol_from_xyz(xyz_str, **{**{'sanitize': False}, **kwargs})
    weights = np.sqrt(get_atom_masses(mol)).reshape(-1, 1)
    positions = mol.GetConformer().GetPositions()
    freq_modes = disp * weights

    # Check 1. change position to have H-atom1 == covalent length, see if the
    # molecule is isomorphic to the product
    target1_dist = get_covalent_radius(
        mol.GetAtomWithIdx(atom1_idx).GetAtomicNum()
    ) + get_covalent_radius(1)
    new_pos = get_well_xyz(H_index, atom1_idx, positions, freq_modes, target1_dist)
    set_conformer_coordinates(mol.GetConformer(), new_pos)
    pmol_fake = mol_from_xyz(mol.ToXYZ(), **{**{'sanitize':False, 'header': True}, **kwargs})
    check1 = is_same_connectivity_mol(pmol, pmol_fake)
    if not check1:
        diff = list(zip(*np.where(pmol.GetAdjacencyMatrix() - pmol_fake.GetAdjacencyMatrix())))
        if len(diff) == 2:  # one bond difference (note, diff is symmetric)
            check1 = check_bond_in_mult_three_member_ring(pmol_fake, diff[0])

    # Check 2. change position to have H-atom2 == covalent length, see if the
    # molecule is isomorphic to the reactant
    target2_dist = get_covalent_radius(
        mol.GetAtomWithIdx(atom2_idx).GetAtomicNum()
    ) + get_covalent_radius(1)
    new_pos = get_well_xyz(H_index, atom2_idx, positions, freq_modes, target2_dist)
    set_conformer_coordinates(mol.GetConformer(), new_pos)
    rmol_fake = mol_from_xyz(mol.ToXYZ(), **{**{'sanitize':False, 'header': True}, **kwargs})
    # print(mol2.GetConformer().GetBondLength([H_index, atom2_idx]), target2_dist)
    check2 = is_same_connectivity_mol(rmol, rmol_fake)
    if not check2:
        diff = list(zip(*np.where(rmol.GetAdjacencyMatrix() - rmol_fake.GetAdjacencyMatrix())))
        if len(diff) == 2:  # one bond difference (note, diff is symmetric)
            check1 = check_bond_in_mult_three_member_ring(rmol_fake, diff[0])

    if single_score:
        return check1 and check2
    return check1, check2
