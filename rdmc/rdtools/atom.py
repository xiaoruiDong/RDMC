import math

from rdkit import Chem

PERIODIC_TABLE = Chem.GetPeriodicTable()

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


def decrement_radical(atom: Chem.Atom):
    """
    Decrement the number of radical electrons on an atom by one.

    Args:
        atom (Atom): The atom whose radical count is to be decremented.
    """
    new_radical_count = atom.GetNumRadicalElectrons() - 1
    if new_radical_count < 0:
        raise ValueError("Radical count cannot be negative")
    atom.SetNumRadicalElectrons(new_radical_count)


def get_electronegativity(atom: Chem.Atom) -> float:
    """
    Get the electronegativity of an atom. Currently, only supports atom 1-35 and 53. Others will
    return 1.0.

    Args:
        atom (Atom): The atom whose electronegativity is to be returned.

    Returns:
        float: The electronegativity of the atom.
    """
    return electronegativity.get(atom.GetAtomicNum(), 1.0)


def get_total_bond_order(atom: Chem.Atom) -> float:
    """
    Get the total bond order of an atom.

    Args:
        atom (Atom): The atom whose total bond order is to be returned.

    Returns:
        float: The total bond order of the atom.
    """
    # Note 1:
    # b.GetValenceContrib(atom) is more robust than b.GetBondTypeAsDouble()
    # as it considers cases like dative bonds

    # Note 2:
    # About hydrogen and bond:
    # SMILES C, 4 implicit Hs, no bond
    # SMILES [CH3], 3 explicit Hs, no bond
    # SMILES [CH]([H])[H], 1 explicit Hs, 2 bond
    return (
        sum([b.GetValenceContrib(atom) for b in atom.GetBonds()]) + atom.GetTotalNumHs()
    )


# RDKit / RDMC compatible
def get_lone_pair(atom: Chem.Atom) -> int:
    """
    Get the number of lone pairs on an atom.

    Args:
        atom (Atom): The atom whose lone pair is to be returned.

    Returns:
        int: The number of lone pairs on the atom.
    """
    order = get_total_bond_order(atom)
    return (
        PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetNumRadicalElectrons()
        - atom.GetFormalCharge()
        - int(order)
    ) // 2


def get_num_occupied_orbitals(atom: Chem.Atom) -> int:
    """
    Get the number of occupied orbitals on an atom.

    Args:
        atom (Atom): The atom whose number of occupied orbitals is to be returned.

    Returns:
        int: The number of occupied orbitals on the atom.
    """
    order = get_total_bond_order(atom)
    return math.ceil(
        (
            PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
            - atom.GetFormalCharge()
            + atom.GetNumRadicalElectrons()
            + int(order)
        )
        / 2
    )


def has_empty_orbitals(atom: Chem.Atom) -> bool:
    """
    Determine whether an atom has empty orbitals.

    Args:
        atom (Atom): The atom to be checked.

    Returns:
        bool: ``True`` if the atom has empty orbitals, ``False`` otherwise.
    """
    atomic_num = atom.GetAtomicNum()
    num_occupied_orbitals = get_num_occupied_orbitals(atom)
    if atomic_num == 1:
        # s
        return num_occupied_orbitals < 1
    elif atomic_num <= 10:
        # sp3
        return num_occupied_orbitals < 4
    elif atomic_num < 36:
        # sp3d2
        return num_occupied_orbitals < 6
    else:
        # sp3d3. IF7. But let's not worry about it for now
        return num_occupied_orbitals < 7


def increment_radical(atom: Chem.Atom):
    """
    Increment the number of radical electrons on an atom by one. It will increase the number of radical electrons by 1 despite
    whether valid. The cleaning step should be done later to ensure the validity of the molecule.

    Args:
        atom (Atom): The atom whose radical count is to be incremented.
    """
    atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
