from typing import List, Tuple, Optional
import copy

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
    bond: Tuple[int, int],
    bond_type: BondType = BondType.SINGLE,
    update_properties: bool = True,
    inplace: bool = True,
) -> Chem.RWMol:
    """
    Add bonds to a molecule.

    Args:
        mol (Chem.RWMol): The molecule to be added.
        bond (tuple): The atom index of the bond to be added.
        bond_type (Chem.BondType, optional): The bond type to be added. Defaults to ``Chem.BondType.SINGLE``.
        update_properties (bool, optional): Whether to update the properties of the molecule. Defaults to ``True``.

    Returns:
        Chem.RWMol: The molecule with bonds added.
    """
    return add_bonds(
        mol, [bond], [bond_type], update_properties=update_properties, inplace=inplace
    )


def add_bonds(
    mol: Chem.RWMol,
    bonds: List[Tuple[int, int]],
    bond_types: Optional[List[BondType]] = None,
    update_properties: bool = True,
    inplace: bool = True,
) -> Chem.RWMol:
    """
    Add bonds to a molecule.

    Args:
        mol (Chem.RWMol): The molecule to be added.
        bond (tuple): The atom index of the bond to be added.
        bond_type (Chem.BondType, optional): The bond type to be added. Defaults to ``Chem.BondType.SINGLE``.
        update_properties (bool, optional): Whether to update the properties of the molecule. Defaults to ``True``.

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


def increment_order(bond: Chem.Bond):
    """
    Increment the bond order of a bond by one.

    Args:
        bond (Bond): The bond whose order is to be incremented.
    """
    bond.SetBondType(BOND_ORDERS[bond.GetBondType() + 1])


def decrement_order(bond: Chem.Bond):
    """
    Decrement the bond order of a bond by one.

    Args:
        bond (Bond): The bond whose order is to be decremented.
    """
    new_order = bond.GetBondTypeAsDouble() - 1
    if new_order <= 0.5:  # Also avoid decrementing aromatic bonds with bond order 1.5
        raise ValueError("Bond order cannot be negative")
    bond.SetBondType(BOND_ORDERS[new_order])


def get_bonds_as_tuples(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """
    Get the bonds of a molecule as a list of tuples.

    Args:
        mol (Mol): The molecule whose bonds are to be returned.

    Returns:
        List[Tuple[int, int]]: The bonds of the molecule as a list of tuples.
    """
    return [tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()]
