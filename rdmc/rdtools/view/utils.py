
def clean_ts(mol, broken_bonds, formed_bonds):
    """
    A helper function remove changing bonds in the TS.
    """
    mol = mol.__copy__()
    for bond in broken_bonds + formed_bonds:
        mol.RemoveBond(*bond)

    return mol


