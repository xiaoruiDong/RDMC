from rdkit import Chem

from rdtools.atommap import (
    clear_atom_map_numbers,
    has_atom_map_numbers,
    needs_renumber,
    renumber_atoms,
    reset_atom_map_numbers,
)
from rdtools.conversion.utils import prepare_output_mol


def get_smiles_parser_params(
    remove_hs: bool = True,
    sanitize: bool = True,
    allow_cxsmiles: bool = False
) -> Chem.SmilesParserParams:
    """
    Get the parameters for the RDKit SMILES parser.

    Args:
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule. Defaults to ``True``.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to ``True``.
        allow_cxsmiles (bool, optional): Whether to recognize and parse CXSMILES. Defaults to ``False``.

    Returns:
        Chem.SmilesParserParams: The parameters for the RDKit SMILES parser.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = remove_hs
    params.sanitize = sanitize
    params.allowCXSMILES = allow_cxsmiles
    return params


def process_mol_from_smiles(
    mol: Chem.Mol,
    remove_hs: bool = True,
    add_hs: bool = True,
) -> Chem.Mol:
    """
    A helper function processing molecules generated from MolFromSmiles.

    Args:
        mol (Chem.Mol): The RDKit Mol/RWMol object to be processed.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule. Defaults to ``True``.
        add_hs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. Defaults to ``True``.
            Only functioning when ``removeHs`` is False.
    Returns:
        Chem.Mol: The processed RDKit Mol/RWMol object.
    """
    if mol is None:
        raise ValueError("The provided SMILES is not valid. Please double check.")

    # By default, for a normal SMILES (e.g.,[CH2]CCCO) other than H indexed SMILES
    # (e.g., [C+:1]#[C:2][C:3]1=[C:7]([H:10])[N-:6][O:5][C:4]1([H:8])[H:9]),
    # no hydrogens are automatically added. So, we need to add H atoms.
    if not remove_hs and add_hs:
        try:
            mol.UpdatePropertyCache(strict=False)
            mol = Chem.rdmolops.AddHs(mol)
        except AttributeError:
            raise ValueError("The provided SMILES is not valid. Please double check.")
    elif remove_hs and add_hs:
        raise ValueError("Cannot add hydrogen atoms when removing hydrogen atoms.")
    return mol


def mol_from_smiles(
    smiles: str,
    remove_hs: bool = False,
    add_hs: bool = True,
    sanitize: bool = True,
    allow_cxsmiles: bool = True,
    keep_atom_map: bool = True,
    assign_atom_map: bool = True,
    atom_order: str = "atom_map",
) -> Chem.Mol:
    """
    Convert a SMILES string to an Chem.Mol molecule object.

    Args:
        smiles (str): A SMILES representation of the molecule.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
        add_hs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. ``True`` to add.
                                Only functioning when removeHs is False.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.
        allow_cxsmiles (bool, optional): Whether to recognize and parse CXSMILES. Defaults to ``True``.
        keep_atom_map (bool, optional): Whether to keep the atom mapping contained in the SMILES. Defaults
                                        Defaults to ``True``.
        assign_atom_map (bool, optional): Whether to assign the atom mapping according to the atom index
                                          if no atom mapping available in the SMILES. Defaults to ``True``.
        atom_order (str, optional): Whether the atom order in the returned molecule (indexes)
                                    is according to atom map ("atom_map") or FIFO (RDKit's default, "fifo").
                                    Defaults to "atom_map".

    Returns:
        Chem.Mol: An RDKit molecule object corresponding to the SMILES.
    """
    params = get_smiles_parser_params(remove_hs, sanitize, allow_cxsmiles)
    mol = Chem.MolFromSmiles(smiles, params)
    mol = process_mol_from_smiles(mol, remove_hs, add_hs)
    if not keep_atom_map:
        reset_atom_map_numbers(mol)
    elif atom_order.lower() == "atom_map" and needs_renumber(mol):
        if has_atom_map_numbers(mol):
            mol = renumber_atoms(mol, update_atom_map=False)
        elif assign_atom_map:
            reset_atom_map_numbers(mol)
    return mol


def mol_to_smiles(
    mol: Chem.Mol,
    stereo: bool = True,
    kekule: bool = False,
    canonical: bool = True,
    remove_atom_map: bool = True,
    remove_hs: bool = True,
) -> str:
    """
    Convert an RDKit molecule object to a SMILES string.

    Args:
        mol (Chem.Mol): An RDKit molecule object.
        stereo (bool, optional): Whether to include stereochemistry information in the SMILES. Defaults to ``True``.
        kekule (bool, optional): Whether to use Kekule encoding. Defaults to ``False``.
        canonical (bool, optional): Whether to use canonical SMILES. Defaults to ``True``.
        remove_atom_map (bool, optional): Whether to keep the Atom mapping contained in the SMILES. Defaults
            Defaults to ``True``.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule. Defaults to ``True``.

    Returns:
        str: A SMILES string corresponding to the RDKit molecule object.
    """
    sanitize = Chem.SanitizeFlags.SANITIZE_ALL
    if kekule:
        sanitize ^= Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    mol = prepare_output_mol(mol, remove_hs, sanitize)

    if remove_atom_map:
        clear_atom_map_numbers(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=stereo, kekuleSmiles=kekule, canonical=canonical)
