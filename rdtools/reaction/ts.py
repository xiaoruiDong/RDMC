"""Functions for analyzing transition states of reactions."""

from copy import copy
from typing import Any, Literal, Sequence, Union

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT

from rdtools.atommap import reset_atom_map_numbers
from rdtools.bond import (
    get_all_changing_bonds,
    get_bonds_as_tuples,
    get_formed_and_broken_bonds,
)
from rdtools.compare import is_same_connectivity_mol
from rdtools.conf import add_conformer, set_conformer_coordinates
from rdtools.conversion.xyz import mol_from_xyz, mol_to_xyz
from rdtools.dist import get_adjacency_matrix
from rdtools.element import get_atom_mass, get_covalent_radius
from rdtools.fix import saturate_mol
from rdtools.mathlib import get_magnitude
from rdtools.mol import get_atom_masses, get_element_symbols


def guess_rxn_from_normal_mode(
    xyz: npt.NDArray[np.float64],
    symbols: Sequence[str],
    disp: npt.NDArray[np.float64],
    amplitude: Union[int, float, list[float]] = 0.25,
    weights: Union[bool, npt.NDArray[np.float64]] = True,
    backend: Literal["openbabel", "xyz2mol"] = "openbabel",
    multiplicity: int = 1,
) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    r"""Guess reaction according to the normal mode analysis for a transition state.

    Args:
        xyz (npt.NDArray[np.float64]): The xyz coordinates of the transition state. It should have a
            size of :math:`N \\times 3`. You can get it from mol.GetConformer().GetPositions().
        symbols (Sequence[str]): The symbols of each atoms. It should have a size of :math:`N`. You can get the symbols
            by rdmc.rdtools.mol.get_element_symbols.
        disp (npt.NDArray[np.float64]): The displacement of the normal mode. It should have a size of
            :math:`N \\times 3`. You can get it from cclib.vib
        amplitude (Union[int, float, list[float]], optional): The amplitude of the motion. If a single value is provided then the guess
            will be unique (if available). ``0.25`` will be the default. Otherwise, a list
            can be provided, and all possible results will be returned.
        weights (Union[bool, npt.NDArray[np.float64]]): If ``True``, use the :math:`\\sqrt(atom mass)` as a scaling factor to the displacement.
            If ``False``, use the identity weights. If a :math:`N \\times 1` ``np.ndarray`` is provided,
            then use the provided weights. The concern is that light atoms (e.g., H)
            tend to have larger motions than heavier atoms.
        backend (Literal["openbabel", "xyz2mol"], optional): The backend used to perceive xyz. Defaults to ``'openbabel'``.
        multiplicity (int, optional): The spin multiplicity of the transition states. Defaults to ``1``.

    Returns:
        tuple[list[Chem.Mol], list[Chem.Mol]]: Potential reactants and products.
    """
    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array([get_atom_mass(symbol) for symbol in symbols]).reshape(
            -1, 1
        )
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((xyz.shape[0], 1))

    mols: dict[str, list[Chem.Mol]] = {"r": [], "p": []}
    amplitudes = [amplitude] if isinstance(amplitude, (int, float)) else amplitude
    for amp in amplitudes:
        xyzs = xyz - amp * disp * weights, xyz + amp * disp * weights

        # Create the reactant complex and the product complex
        for xyz_mod, side in zip(xyzs, ["r", "p"]):
            xyz_str = ""
            for symbol, coords in zip(symbols, xyz_mod):
                xyz_str += (
                    f"{symbol:4}{coords[0]:14.8f}{coords[1]:14.8f}{coords[2]:14.8f}\n"  # type: ignore
                )
            try:
                mols[side].append(mol_from_xyz(xyz_str, header=False, backend=backend))
                reset_atom_map_numbers(mols[side][-1])
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

    return mols["r"], mols["p"]


def examine_normal_mode(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
    ts_xyz: npt.NDArray[np.float64],
    disp: npt.NDArray[np.float64],
    amplitude: Union[float, list[float]] = 0.25,
    weights: Union[bool, npt.NDArray[np.float64]] = True,
    verbose: bool = True,
    as_factors: bool = True,
) -> Union[bool, tuple[float, float, float]]:
    """Examine a TS's imaginary frequency given known reactant and product complexes.

    The function checks if the bond changes are corresponding to the most
    significant change in the normal mode. The reactant and product complex need to be
    atom mapped.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.
        ts_xyz (npt.NDArray[np.float64]): The xyz coordinates of the transition state. It should have a
            size of N x 3.
        disp (npt.NDArray[np.float64]): The displacement of the normal mode. It should have a size of
            N x 3.
        amplitude (Union[float, list[float]]): The amplitude of the motion. Defaults to ``0.25``.
        weights (Union[bool, npt.NDArray[np.float64]], optional): If ``True``, use the sqrt(atom mass) as a scaling factor to the displacement.
            If ``False``, use the identity weights. If a N x 1 ``npt.NDArray[np.float64]`` is provided, then
            The concern is that light atoms (e.g., H) tend to have larger motions
            than heavier atoms.
        verbose (bool, optional): If print detailed information. Defaults to ``True``.
        as_factors (bool, optional): If return the value of factors instead of a judgment.
            Defaults to ``False``

    Returns:
        Union[bool, tuple[float, float, float]]: The return value depends on the value of `as_factors`.
            If `as_factors` is ``True``, return the factors of the bond distance change.
            Otherwise, return a boolean value.
    """
    # Analyze connectivity
    broken, formed, changed = get_all_changing_bonds(rmol, pmol)
    reacting_bonds = broken + formed + changed

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array(get_atom_masses(rmol)).reshape(-1, 1)
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
    larger_factor = float((np.min(formed_and_broken_diff) - baseline) / std)
    # There might be no bond that only changes its order
    smaller_factor = (
        float((np.min(changed_diff) - baseline) / std) if changed_diff else 0.0
    )
    other_factor = float(baseline / std)

    if verbose:
        print(
            f"The min. bond distance change for bonds that are broken or formed"
            f" is {np.min(formed_and_broken_diff)} A and is {larger_factor:.1f} STD off the baseline."
        )
        if changed_diff:
            print(
                f"The min. bond distance change for bonds that are changed"
                f" is {np.min(changed_diff)} A and is {smaller_factor:.1f} STD off the baseline."
            )

    if as_factors:
        return larger_factor, smaller_factor, other_factor

    if larger_factor > 3:
        return True

    return False


def is_valid_habs_ts(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
    ts_xyz: npt.NDArray[np.float64],
    disp: npt.NDArray[np.float64],
    single_score: bool = True,
    **kwargs: Any,
) -> Union[bool, tuple[bool, bool]]:
    """Check if the TS belonging to a H atom abstraction reactions.

    This function assumes the TS, rmol, and pmol have the same atom order. This may be extended to
    general substitution reaction.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.
        ts_xyz (npt.NDArray[np.float64]): The xyz coordinates of the transition state. It should have a size of
            N x 3.
        disp (npt.NDArray[np.float64]): The displacement of the normal mode. It should have a size of
            N x 3.
        single_score (bool, optional): If return a single score (True or False). Defaults to ``True``.
            Otherwise, return the check results (True or False) for both side of the reaction
        **kwargs (Any): Additional arguments for the molecule generation.

    Returns:
        Union[bool, tuple[bool, bool]]: The return value depends on the value of `single_score`.
            If `single_score` is ``True``, return a single score (``True`` or ``False``).
            Otherwise, return the check results (``True`` or ``False``) for both side of the reaction.
    """

    def get_well_xyz(
        atom1_idx: int,
        atom2_idx: int,
        positions: npt.NDArray[np.float64],
        freq_modes: npt.NDArray[np.float64],
        target: float,
    ) -> npt.NDArray[np.float64]:
        """Get the well molecule atom coordinates.

        Adjust the position by the frequency mode, so that the distance between the H
        atom and the bonding atom is equal to the target value.

        Args:
            atom1_idx (int): atom index of the H atom or its bonding atom
            atom2_idx (int): atom index of the other atom
            positions (npt.NDArray[np.float64]): xyz coordinates of the atoms of the TS
            freq_modes (npt.NDArray[np.float64]): frequency modes of the imaginary frequency
            target (float): target distance between the H atom and the bonding atom

        Returns:
            npt.NDArray[np.float64]: new xyz coordinates of the atoms
        """
        # Knowing the distance between Atom A and B
        # as well as the distance change between Atom A and B due the to the frequency,
        # Compute the magnitude of the distance change.
        # This is solved by solving a quadratic equation
        # The root with smaller absolute value is chosen as the magnitude.
        x_AB = positions[atom1_idx] - positions[atom2_idx]
        x_dAB = freq_modes[atom1_idx] - freq_modes[atom2_idx]
        m1, m2 = get_magnitude(x_AB, x_dAB, target)
        magnitude = m1 if abs(m1) < abs(m2) else m2
        return positions - freq_modes * magnitude

    def check_bond_in_mult_three_member_ring(
        mol: Chem.Mol, bond: tuple[int, int]
    ) -> bool:
        """Check if the bond is in multiple three member rings.

        Perception algorithm currently is confused at this condition.

        Args:
            mol (Chem.Mol): the molecule
            bond (tuple[int, int]): the bond

        Returns:
            bool: ``True`` if the bond is in multiple three member rings, ``False`` otherwise
        """
        try:
            bond_idx = mol.GetBondBetweenAtoms(int(bond[0]), int(bond[1])).GetIdx()
        except TypeError:
            return False
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        membership = ring_info.BondMembers(bond_idx)
        num_in_three_member_ring = sum([ring_sizes[mem] == 3 for mem in membership])
        return num_in_three_member_ring >= 2

    # Obtain the information of the reaction center
    formed, broken = get_formed_and_broken_bonds(rmol, pmol)
    assert len(formed) == 1 and len(broken) == 1, "The TS should be b1f1."
    _formed, _broken = set(formed[0]), set(broken[0])
    H_idx = (_formed & _broken).pop()  # the H atom index in both sides
    atom1_idx = (_formed - {H_idx}).pop()  # the radical site in reactant
    atom2_idx = (_broken - {H_idx}).pop()  # the radical site in product
    atom1_atomic_num = rmol.GetAtomWithIdx(atom1_idx).GetAtomicNum()
    atom2_atomic_num = pmol.GetAtomWithIdx(atom2_idx).GetAtomicNum()

    # Write a XYZ to perceive the transition state connectivity
    symbols = get_element_symbols(rmol)
    xyz_str = ""
    for symbol, coords in zip(symbols, ts_xyz):
        xyz_str += f"{symbol:4}{coords[0]:14.8f}{coords[1]:14.8f}{coords[2]:14.8f}\n"  # type: ignore
    mol = mol_from_xyz(
        xyz_str, **{**{"sanitize": False, "header": False}, **kwargs}
    )  # force sanitize to be False for the TS structure

    # Add weights to the displacement
    weights = np.sqrt(get_atom_masses(mol)).reshape(-1, 1)
    freq_modes = disp * weights

    # Check 1. change the TS geometry to have L(H--atom1) == covalent bond length, see if the
    # molecule is isomorphic to the product
    len_H_atom1 = sum(map(get_covalent_radius, [1, atom1_atomic_num]))
    new_pos = get_well_xyz(H_idx, atom1_idx, ts_xyz, freq_modes, len_H_atom1)
    set_conformer_coordinates(mol.GetConformer(), new_pos)
    product_xyz = mol_to_xyz(mol)
    pmol_fake = mol_from_xyz(
        product_xyz, **{**{"sanitize": False, "header": True}, **kwargs}
    )
    check1 = is_same_connectivity_mol(pmol, pmol_fake)
    if not check1:
        diff = list(
            zip(*np.where(get_adjacency_matrix(pmol) - get_adjacency_matrix(pmol_fake)))
        )
        if len(diff) == 2:  # one bond difference (note, diff is symmetric)
            check1 = check_bond_in_mult_three_member_ring(pmol_fake, diff[0])

    # Check 2. change position to have H-atom2 == covalent length, see if the
    # molecule is isomorphic to the reactant
    len_H_atom2 = sum(map(get_covalent_radius, [1, atom2_atomic_num]))
    new_pos = get_well_xyz(H_idx, atom2_idx, ts_xyz, freq_modes, len_H_atom2)
    set_conformer_coordinates(mol.GetConformer(), new_pos)
    reactant_xyz = mol_to_xyz(mol)
    rmol_fake = mol_from_xyz(
        reactant_xyz, **{**{"sanitize": False, "header": True}, **kwargs}
    )
    check2 = is_same_connectivity_mol(rmol, rmol_fake)
    if not check2:
        diff = list(
            zip(*np.where(get_adjacency_matrix(rmol) - get_adjacency_matrix(rmol_fake)))
        )
        if len(diff) == 2:  # one bond difference (note, diff is symmetric)
            check1 = check_bond_in_mult_three_member_ring(rmol_fake, diff[0])

    if single_score:
        return check1 and check2
    return check1, check2
