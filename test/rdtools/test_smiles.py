import pytest

from rdmc.rdtools.conversion import mol_from_smiles, mol_to_smiles
from rdmc.rdtools.atommap import get_atom_map_numbers


@pytest.mark.parametrize(
    "smi",
    [
        "[C-]#[O+]",
        "[C]",
        "[CH]",
        "OO",
        "[H][H]",
        "[H]",
        "[He]",
        "[O]",
        "O",
        "[CH3]",
        "C",
        "[OH]",
        "CCC",
        "CC",
        "N#N",
        "[O]O",
        "[CH2]C",
        "[Ar]",
        "CCCC",
        "O=C=O",
        "[C]#N",
    ]
)
def test_smiles_without_atom_mapping_and_hs(smi):
    """
    Test exporting a molecule as a SMILES string without atom mapping and explicit H atoms.
    """
    mol = mol_from_smiles(smi)
    assert mol_to_smiles(mol) == smi


def test_smiles_with_atom_mapping_and_hs():
    """
    Test exporting a molecule as a SMILES string with atom mapping and explicit H atoms.
    """
    # Normal SMILES without atom mapping, atommap and H atoms will be
    # assigned during initiation
    mol1 = mol_from_smiles("[CH2]C")
    # Export SMILES with H atoms
    assert (
        mol_to_smiles(
            mol1,
            remove_hs=False,
        )
        == "[H][C]([H])C([H])([H])[H]"
    )
    # Export SMILES with H atoms and indexes
    assert (
        mol_to_smiles(mol1, remove_hs=False, remove_atom_map=False)
        == "[C:1]([C:2]([H:5])([H:6])[H:7])([H:3])[H:4]"
    )

    # SMILES with atom mapping
    mol2 = mol_from_smiles("[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]")
    # Test the atom indexes and atom map numbers share the same order
    assert get_atom_map_numbers(mol2) == [1, 2, 3, 4, 5, 6, 7]
    # Test the 2nd and 4th atoms are carbons
    assert mol2.GetAtomWithIdx(1).GetAtomicNum() == 6
    assert mol2.GetAtomWithIdx(3).GetAtomicNum() == 6
    # Export SMILES without H atoms and atom map
    assert mol_to_smiles(mol2) == "[CH2]C"
    # Export SMILES with H atoms and without atom map
    assert (
        mol_to_smiles(
            mol2,
            remove_hs=False,
        )
        == "[H][C]([H])C([H])([H])[H]"
    )
    # Export SMILES without H atoms and with atom map
    # Atom map numbers for heavy atoms are perserved
    assert (
        mol_to_smiles(
            mol2,
            remove_atom_map=False,
        )
        == "[CH3:2][CH2:4]"
    )
    # Export SMILES with H atoms and with atom map
    assert (
        mol_to_smiles(
            mol2,
            remove_hs=False,
            remove_atom_map=False,
        )
        == "[H:1][C:4]([C:2]([H:5])([H:6])[H:7])[H:3]"
    )

    # SMILES with atom mapping but neglect the atom mapping
    mol3 = mol_from_smiles(
        "[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]",
        keep_atom_map=False
    )
    # Test the atom indexes and atom map numbers share the same order
    assert get_atom_map_numbers(mol3) == [1, 2, 3, 4, 5, 6, 7]
    # However, now the 4th atom is not carbon (3rd instead), and atom map numbers
    # are determined by the sequence of atom appear in the SMILES.
    assert mol3.GetAtomWithIdx(1).GetAtomicNum() == 6
    assert mol3.GetAtomWithIdx(2).GetAtomicNum() == 6
    # Export SMILES with H atoms and with atom map
    assert (
        mol_to_smiles(
            mol3,
            remove_hs=False,
            remove_atom_map=False,
        )
        == "[H:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]"
    )

    # SMILES with uncommon atom mapping starting from 0 and being discontinue
    mol4 = mol_from_smiles("[H:9][C:2]([C:5]([H:0])[H:3])([H:4])[H:8]")
    # Test the atom indexes and atom map numbers share the same order
    assert get_atom_map_numbers(mol4) == [0, 2, 3, 4, 5, 8, 9]
    # Check Heavy atoms' index
    assert mol4.GetAtomWithIdx(1).GetAtomicNum() == 6
    assert mol4.GetAtomWithIdx(4).GetAtomicNum() == 6
    # Export SMILES without H atoms and with atom map
    # Atom map numbers for heavy atoms are perserved
    assert (
        mol_to_smiles(
            mol4,
            remove_atom_map=False,
        )
        == "[CH3:2][CH2:5]"
    )
    # Export SMILES with H atoms and with atom map
    assert (
        mol_to_smiles(
            mol4,
            remove_hs=False,
            remove_atom_map=False,
        )
        == "[H:0][C:5]([C:2]([H:4])([H:8])[H:9])[H:3]"
    )
