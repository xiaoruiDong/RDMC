from rdkit import Chem
from typing import Union


def prepare_output_mol(
    mol: Chem.Mol,
    remove_hs: bool = False,
    sanitize: Union[bool, Chem.SanitizeFlags] = True,
):
    """
    Generate a RDKit Mol instance for output purpose, to ensure that the original molecule is not modified.

    Args:
        remove_hs (bool, optional): Remove less useful explicit H atoms. E.g., When output SMILES, H atoms,
            if explicitly added, will be included, which reduces the readability. Defaults to ``False``.
            Note, following the Hs are not removed:

                1. H which aren't connected to a heavy atom. E.g.,[H][H].
                2. Labelled H. E.g., atoms with atomic number=1, but isotope > 1.
                3. Two coordinate Hs. E.g., central H in C[H-]C.
                4. Hs connected to dummy atoms
                5. Hs that are part of the definition of double bond Stereochemistry.
                6. Hs that are not connected to anything else.

        sanitize (bool, optional): Whether to sanitize the molecule. Defaults to ``True``. Using Chem.SanitizeFlags
            is also acceptable.

    Returns:
        Mol: A Mol instance used for output purpose.
    """
    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=False)
    else:
        mol = Chem.Mol(mol)  # make sure it is copied
    if isinstance(sanitize, bool) and sanitize:
        Chem.SanitizeMol(mol)
    elif sanitize:
        Chem.SanitizeMol(mol, sanitizeOps=sanitize)
    return mol
