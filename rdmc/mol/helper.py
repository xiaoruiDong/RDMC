from rdmc.mol import Mol
from rdtools.fix import saturate_mol
from rdmc.utils import filter_kwargs


def parse_xyz_or_smiles_list(mol_list, **kwargs):
    """
    A helper function to parse xyz and smiles and list if the
    conformational information is provided.

    Args:
        mol_list (list): a list of dict. These dicts should have a key of
                        "smi" or "xyz", with values to SMILES or XYZ string,
                        respectively. you can also pass in "mult" to correct
                        the multiplicity if applicable.
        **kwargs: additional arguments to pass to the Mol.FromXYZ or Mol.FromSmiles.

    Returns:
        list: The same list passed in but with mol attached
    """
    for val in mol_list:
        if "smi" in val:
            mol = Mol.FromSmiles(
                val['smi'],
                **filter_kwargs(
                    Mol.FromSmiles,
                    kwargs
                ),
            )
            mol.EmbedConformer()
            val['is3D'] = False
        elif "xyz" in val:
            mol = Mol.FromXYZ(
                val['xyz'],
                **filter_kwargs(
                    Mol.FromXYZ,
                    kwargs
                ),
            )
            val['is3D'] = True
        if 'mult' in val and mol.GetSpinMultiplicity() != val['mult']:
            saturate_mol(mol, val['mult'])
            # todo raise a warning if the saturate fails.
        val['mol'] = mol
    return mol_list
