"""A collection of helper functions for the viewer."""

from rdkit import Chem


def clean_ts(
    mol: Chem.Mol,
    broken_bonds: list[tuple[int, int]],
    formed_bonds: list[tuple[int, int]],
) -> Chem.Mol:
    """Remove changing bonds in the TS.

    Args:
        mol (Chem.Mol): The molecule to be cleaned.
        broken_bonds (list[tuple[int, int]]): The bonds that are broken.
        formed_bonds (list[tuple[int, int]]): The bonds that are formed.

    Returns:
        Chem.Mol: The cleaned molecule.
    """
    if isinstance(mol, Chem.RWMol):
        mol = mol.__copy__()
    else:  # Assume it is a Mol, we need to make it editable
        mol = Chem.RWMol(mol)

    for bond in broken_bonds + formed_bonds:
        if mol.GetBondBetweenAtoms(*bond) is None:
            continue
        mol.RemoveBond(*bond)

    return mol


# The following two functions have duplicates in rdtools
# These are copied over in preparation of making view completely standalone
def get_broken_formed_bonds(
    r_mol: Chem.Mol, p_mol: Chem.Mol
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """A helper function to get the broken and formed bonds.

    Args:
        r_mol (Chem.Mol): The reactant molecule.
        p_mol (Chem.Mol): The product molecule.

    Returns:
        tuple[list[tuple[int, int]], list[tuple[int, int]]]: The broken and formed bonds.
    """
    r_bonds = set(get_bonds_as_tuples(r_mol))
    p_bonds = set(get_bonds_as_tuples(p_mol))

    return list(r_bonds - p_bonds), list(p_bonds - r_bonds)


def get_bonds_as_tuples(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Get the bonds of a molecule as a list of tuples.

    Args:
        mol (Chem.Mol): The molecule whose bonds are to be returned.

    Returns:
        list[tuple[int, int]]: The bonds of the molecule as a list of tuples.
    """
    return [
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
    ]


def merge_xyz_dxdydz(
    xyz: str,
    dxdydz: list[list[float]],
) -> str:
    """A helper function to create input for freq_viewer.

    It merges the xyz string with the dxdydz information.

    Args:
        xyz (str): The xyz string.
        dxdydz (list[list[float]]): The dx dy dz in a 3 x N matrix-like list or array.

    Returns:
        str: The xyz string with dxdydz information.
    """
    lines = xyz.strip().splitlines()

    assert len(lines) - 2 == len(dxdydz), (
        f"The number of atoms doesn't match xyz ({len(lines) - 2}) "
        f"and dxdydz ({len(dxdydz)})"
    )

    new_lines = lines[0:2] + [
        line.strip() + "\t\t" + "".join([f"{dx:<12}" for dx in dxdydz_onerow])
        for line, dxdydz_onerow in zip(lines[2:], dxdydz)
    ]

    return "\n".join(new_lines)
