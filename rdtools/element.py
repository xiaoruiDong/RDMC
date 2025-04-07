"""Functions to get information about elements in the periodic table."""

from typing import Union

from rdkit.Chem import GetPeriodicTable

PERIODIC_TABLE = GetPeriodicTable()

electronegativity = {
    1: 2.20,
    3: 0.98,
    4: 1.57,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    13: 1.61,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    19: 0.82,
    20: 1.00,
    21: 1.36,
    22: 1.54,
    23: 1.63,
    24: 1.66,
    25: 1.55,
    26: 1.83,
    27: 1.91,
    29: 1.90,
    30: 1.65,
    31: 1.81,
    32: 2.01,
    33: 2.18,
    34: 2.55,
    35: 2.96,
    53: 2.66,
}


def get_electronegativity(element: Union[int, str]) -> float:
    """Get the electronegativity of an element.

    Currently, only supports atom 1-35 and 53. Others will return 1.0.

    Args:
        element (Union[int, str]): The atomic number or the symbol of the element whose electronegativity is to be returned.

    Returns:
        float: The electronegativity of the atom.
    """
    if isinstance(element, str):
        element = get_atomic_num(element)
    return electronegativity.get(element, 1.0)


def get_n_outer_electrons(element: Union[int, str]) -> int:
    """Get the number of outer electrons of an element.

    Args:
        element (Union[int, str]): The atomic number or the symbol of the element whose number of outer electrons is to be returned.

    Returns:
        int: The number of outer electrons of the element.
    """
    return PERIODIC_TABLE.GetNOuterElecs(element)


def get_atomic_num(symbol: str) -> int:
    """Get the atomic number of an atom given its symbol.

    Args:
        symbol (str): The symbol of the atom.

    Returns:
        int: The atomic number of the atom.
    """
    return PERIODIC_TABLE.GetAtomicNumber(symbol)


def get_element_symbol(atomic_num: int) -> str:
    """Get the symbol of an atom given its atomic number.

    Args:
        atomic_num (int): The atomic number of the atom.

    Returns:
        str: The symbol of the atom.
    """
    return PERIODIC_TABLE.GetElementSymbol(atomic_num)


def get_atom_mass(element: Union[int, str]) -> float:
    """Get the mass of an element.

    Args:
        element (Union[int, str]): The atomic number or the symbol of the element.

    Returns:
        float: The mass of the atom.
    """
    return PERIODIC_TABLE.GetAtomicWeight(element)


def get_covalent_radius(element: Union[int, str]) -> float:
    """Get the covalent radius of an element.

    Args:
        element (Union[int, str]): The atomic number or the symbol of the element.

    Returns:
        float: The covalent radius of the atom.
    """
    return PERIODIC_TABLE.GetRcovalent(element)


def get_vdw_radius(element: Union[int, str]) -> float:
    """Get the van der Waals radius of an element.

    Args:
        element (Union[int, str]): The atomic number or the symbol of the element.

    Returns:
        float: The van der Waals radius of the atom.
    """
    return PERIODIC_TABLE.GetRvdw(element)


def get_bond_radius(element: Union[int, str]) -> float:
    """Get the bond radius of an element.

    Args:
        element (Union[int, str]): The atomic number or the symbol of the element.

    Returns:
        float: The bond radius of the atom.
    """
    return PERIODIC_TABLE.GetRb0(element)
