
def clean_ts(mol, broken_bonds, formed_bonds):
    """
    A helper function remove changing bonds in the TS.
    """
    mol = mol.__copy__()
    for bond in broken_bonds + formed_bonds:
        mol.RemoveBond(*bond)

    return mol



def merge_xyz_dxdydz(
    xyz: str,
    dxdydz: list,
) -> str:
    """
    A helper function to create input for freq_viewer.
    Merge the xyz string with the dxdydz information.

    Args:
        xyz (str): The xyz string.
        dxdydz (list): The dx dy dz in a 3 x N matrix like list or array.

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
