from copy import copy
from typing import Union, Sequence, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT

from rdtools.bond import get_all_changing_bonds, get_bonds_as_tuples
from rdtools.conversion.xyz import mol_from_xyz
from rdtools.conf import add_conformer
from rdtools.element import get_atom_mass
from rdtools.fix import saturate_mol


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
