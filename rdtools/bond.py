# -*- coding: utf-8 -*-
"""A module contains functions to manipulate bonds in a molecule."""

import copy
from functools import lru_cache
from typing import Optional

from rdkit import Chem
from rdkit.Chem import BondType

BOND_ORDERS = {
    1: BondType.SINGLE,
    1.5: BondType.AROMATIC,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.QUADRUPLE,
    12: BondType.AROMATIC,
    "S": BondType.SINGLE,
    "D": BondType.DOUBLE,
    "T": BondType.TRIPLE,
    "B": BondType.AROMATIC,
    "Q": BondType.QUADRUPLE,
}


def add_bond(
    mol: Chem.RWMol,
    bond: tuple[int, int],
    bond_type: BondType = BondType.SINGLE,
    update_properties: bool = True,
    inplace: bool = True,
) -> Chem.RWMol:
    """Add a bond to a molecule.

    Args:
        mol (Chem.RWMol): The molecule to be added.
        bond (tuple[int, int]): The atom index of the bond to be added.
        bond_type (BondType, optional): The bond type to be added. Defaults to ``Chem.BondType.SINGLE``.
        update_properties (bool, optional): Whether to update the properties of the molecule. Defaults to ``True``.
        inplace (bool, optional): Whether to modify the molecule in place. Defaults to ``True``.

    Returns:
        Chem.RWMol: The molecule with bonds added.
    """
    return add_bonds(
        mol, [bond], [bond_type], update_properties=update_properties, inplace=inplace
    )


def add_bonds(
    mol: Chem.RWMol,
    bonds: list[tuple[int, int]],
    bond_types: Optional[list[BondType]] = None,
    update_properties: bool = True,
    inplace: bool = True,
) -> Chem.RWMol:
    """Add bonds to a molecule.

    Args:
        mol (Chem.RWMol): The molecule to be added.
        bonds (list[tuple[int, int]]): The atom index of the bond to be added.
        bond_types (Optional[list[BondType]], optional): The bond type to be added. Defaults to ``Chem.BondType.SINGLE``.
        update_properties (bool, optional): Whether to update the properties of the molecule. Defaults to ``True``.
        inplace (bool, optional): Whether to modify the molecule in place. Defaults to ``True``.

    Returns:
        Chem.RWMol: The molecule with bonds added.
    """
    if not inplace:
        mol = copy.copy(mol)
    if bond_types is None:
        bond_types = [Chem.BondType.SINGLE] * len(bonds)
    for bond, bond_type in zip(bonds, bond_types):
        mol.AddBond(*bond, bond_type)
    if update_properties:
        Chem.GetSymmSSSR(mol)
        mol.UpdatePropertyCache(strict=False)
    return mol


def increment_bond_order(bond: Chem.Bond) -> None:
    """Increment the bond order of a bond by one.

    Args:
        bond (Chem.Bond): The bond whose order is to be incremented.
    """
    bond.SetBondType(BOND_ORDERS[bond.GetBondType() + 1])


def decrement_bond_order(bond: Chem.Bond) -> None:
    """Decrement the bond order of a bond by one.

    Args:
        bond (Chem.Bond): The bond whose order is to be decremented.

    Raises:
        ValueError: If bond order is negative.
    """
    new_order = bond.GetBondTypeAsDouble() - 1
    if new_order <= 0.5:  # Also avoid decrementing aromatic bonds with bond order 1.5
        raise ValueError("Bond order cannot be negative")
    bond.SetBondType(BOND_ORDERS[new_order])


def get_bonds_as_tuples(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Get the bonds of a molecule as a list of tuples.

    Args:
        mol (Chem.Mol): The molecule whose bonds are to be returned.

    Returns:
        list[tuple[int, int]]: The bonds of the molecule as a list of tuples.
    """
    return [
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol.GetBonds()  # type: ignore[no-untyped-call]
    ]


@lru_cache(maxsize=10000)
def _get_bonds_as_sets(
    *mols: Chem.Mol,
) -> tuple[set[tuple[int, int]], ...]:
    """Get the set of bonds for the provided list of mols.

    Args:
        *mols (Chem.Mol): a RDKit Mol object

    Returns:
        tuple[set[tuple[int, int]], ...]: (bond set in the reactant, bond set in the product)
    """
    return tuple(set(get_bonds_as_tuples(mol)) for mol in mols)


def get_formed_bonds(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
) -> list[tuple[int, int]]:
    """Get all bonds formed in the reaction.

    Both reactant and product complexes need to be atom-mapped to get the correct
    atom indexes. The formed bonds are the bonds that are present in the product
    complex but not in the reactant complex.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.

    Returns:
        list[tuple[int, int]]: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(rmol, pmol)
    return list(p_bonds - r_bonds)


def get_broken_bonds(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
) -> list[tuple[int, int]]:
    """Get all bonds broken in the reaction.

    Both reactant and product complexes need to be atom-mapped. The broken bonds are the
    bonds that are present in the reactant complex but not in the product complex.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.

    Returns:
        list[tuple[int, int]]: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(rmol, pmol)
    return list(r_bonds - p_bonds)


def get_bonds_with_BO_changed(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
) -> list[tuple[int, int]]:
    """Get all bonds whose bond order is changed in the reaction.

    Both reactant and product complexes need to be atom-mapped. The changed bonds are the
    bonds that are present in both the reactant and product complexes but have a different
    bond order.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.

    Returns:
        list[tuple[int, int]]: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(rmol, pmol)
    changed_bonds = [
        bond
        for bond in (r_bonds & p_bonds)
        if (
            rmol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble()
            != pmol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble()
        )
    ]
    return changed_bonds


def get_formed_and_broken_bonds(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Get all bonds broken in the reaction.

    Both reactant and product complexes need to be atom-mapped.
    This function doesn't count bonds whose bond order is lowered but
    not equal to zero.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.

    Returns:
        tuple[list[tuple[int, int]], list[tuple[int, int]]]:
            - formed bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
            - broken bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    return get_formed_bonds(rmol, pmol), get_broken_bonds(rmol, pmol)


def get_all_changing_bonds(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Get all bonds changed in the reaction.

    Both reactant and product complexes need to be atom-mapped.

    Args:
        rmol (Chem.Mol): the reactant complex.
        pmol (Chem.Mol): the product complex.

    Returns:
        tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
            - formed bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
            - broken bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
            - bonds with BO changed: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    return (
        get_formed_bonds(rmol, pmol),
        get_broken_bonds(rmol, pmol),
        get_bonds_with_BO_changed(rmol, pmol),
    )


def get_atoms_in_bonds(bonds: list[tuple[int, int]], sorted: bool = False) -> list[int]:
    """Get unique atoms in a list of bonds.

    Args:
        bonds (list[tuple[int, int]]): A list of length-2 tuples that contain the atom indexes of the bonded atoms.
        sorted (bool, optional): Whether to sort the atom indexes. Defaults to ``False``.

    Returns:
        list[int]: A list of the atom indexes of the atoms in the bonds.
    """
    atoms: list[int] = list(set(sum(bonds, ())))
    if sorted:
        atoms.sort()
    return atoms
