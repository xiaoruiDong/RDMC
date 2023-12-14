from rdkit import Chem


def get_spin_multiplicity(mol: Chem.Mol) -> int:
    """
    Get spin multiplicity of a molecule. The spin multiplicity is calculated
    using Hund's rule of maximum multiplicity defined as 2S + 1.

    Args:
        mol (Chem.Mol): The molecule to get spin multiplicity.

    Returns:
        int : Spin multiplicity.
    """
    return 1 + sum(
        [
            mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
            for i in range(mol.GetNumAtoms())
        ]
    )


def get_formal_charge(mol: Chem.Mol) -> int:
    """
    Get formal charge of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get formal charge.

    Returns:
        int : Formal charge.
    """
    return Chem.GetFormalCharge(mol)


def force_no_implicit(mol: Chem.Mol):
    """
    Set no implicit hydrogen for atoms without implicit/explicit hydrogens. When
    manipulating molecules by changing number of radical electrons / charges and then updating the cached properties,
    additional hydrogens may be added to the molecule. This function helps avoid this problem.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1 and not atom.GetTotalNumHs():
            atom.SetNoImplicit(True)


def reflect(mol: Chem.Mol, conf_id: int = 0):
    """
    Reflect the coordinates of the conformer of the molecule.

    Args:
        mol (Chem.Mol): The molecule to reflect.
        conf_id (int, optional): The conformer ID to reflect.
    """
    conf = mol.GetConformer(conf_id)
    _reflect(conf)
