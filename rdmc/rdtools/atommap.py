from rdkit import Chem


# TODO: Add a function to set atom map numbers to atoms
def set_atom_map_numbers(mol: Chem.Mol):
    """
    Set atom map numbers to atoms. This is used to generate atom map numbers for molecules.
    """
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(idx + 1)


def update_product_atom_map_after_reaction(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    clean_rxn_props: bool = True,
):
    """
    Update the atom map number of the product molecule after reaction according to the reference molecule (usually the reactant).
    The operation is in-place.

    Args:
        mol (RDKitMol): The product molecule after reaction.
        ref_mol (RDKitMol): The reference molecule (usually the reactant).
        clean_rxn_props (bool, optional): Whether to clean the reaction properties (`"old_mapno"` and `"react_atom_idx"`).
                                          Defaults to ``True``.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("old_mapno"):
            # atom map number of the product will zeroed out during the reaction
            react_atom_idx = atom.GetProp("react_atom_idx")
            atom.SetAtomMapNum(ref_mol.GetAtomWithIdx(react_atom_idx).GetAtomMapNum())
        if clean_rxn_props:
            atom.ClearProp("react_atom_idx")
            atom.ClearProp("old_mapno")
