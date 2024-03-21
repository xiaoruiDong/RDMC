import pytest

import numpy as np
from rdkit import Chem

from rdmc.rdtools.atommap import (
    reverse_map,
    get_atom_map_numbers,
    has_atom_map_numbers,
    needs_renumber,
    renumber_atoms,
    renumber_atoms_by_substruct_match_result,
    move_atommaps_to_notes,
    move_notes_to_atommaps,
)

# In this module, RDKit's native smiles to mol is used,
# as RDMC's version includes atom mapping change / reordering,
# making it not the best way to test pure atom mapping functions
smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


@pytest.mark.parametrize(
    "smiles, numbers",
    [
        ("C1CCCCC1", [0] * 6),
        ("[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1", list(range(1, 7))),
    ],
)
def test_get_atom_map_numbers(smiles, numbers):
    mol = Chem.MolFromSmiles(smiles, smi_params)
    atom_map_nums = get_atom_map_numbers(mol)
    assert atom_map_nums == numbers


@pytest.mark.parametrize(
    "smiles",
    [
        (
            "[C:1]1([H:7])([H:8])[C:2]([H:9])([H:10])[C:3]([H:11])([H:12])"
            "[C:4]([H:13])([H:14])[C:5]([H:15])([H:16])[C:6]1([H:17])[H:18]"
        )
    ],
)
def test_renumber_atoms_by_atom_map_numbers(smiles):
    mol = Chem.MolFromSmiles(smiles, smi_params)
    atom_map_nums = get_atom_map_numbers(mol)

    # atom has atom map numbers but not consistent the atom indices
    assert atom_map_nums != [0] * len(atom_map_nums)
    assert atom_map_nums != list(range(1, len(atom_map_nums) + 1))

    new_mol = renumber_atoms(mol)
    atom_map_nums = get_atom_map_numbers(new_mol)
    assert atom_map_nums == list(range(1, len(atom_map_nums) + 1))


def test_renumber_atoms_by_map_dict():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    atom_map_numbers_before = get_atom_map_numbers(mol)

    mapping = {0: 2}
    new_mol = renumber_atoms(mol, mapping, update_atom_map=False)
    atom_map_numbers_after = get_atom_map_numbers(new_mol)

    assert atom_map_numbers_before != atom_map_numbers_after
    assert atom_map_numbers_after == [3, 2, 1]

    new_mol = renumber_atoms(mol, mapping, update_atom_map=True)
    atom_map_numbers_after = get_atom_map_numbers(new_mol)
    # atom map number now is consistent with the atom indices
    # But the first atom is oxygen now
    assert atom_map_numbers_before == atom_map_numbers_after
    assert new_mol.GetAtomWithIdx(0).GetAtomicNum() == 8
    assert new_mol.GetAtomWithIdx(2).GetAtomicNum() == 6


@pytest.mark.parametrize(
    "smi1, smi2",
    [
        (
            "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
            "[H:1][C:4]([C:2]([O:3][H:9])([H:7])[H:8])([H:5])[H:6]",
        )
    ],
)
def test_renumber_atoms_by_substruct_match_result(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1, smi_params)
    mol2 = Chem.MolFromSmiles(smi2, smi_params)

    # Renumber to make atom index and atom map consistent
    # so that comparison can be easier
    mol1 = renumber_atoms(mol1)
    mol2 = renumber_atoms(mol2)

    mapping = mol2.GetSubstructMatch(mol1)
    new_mol1 = renumber_atoms_by_substruct_match_result(mol1, mapping, as_ref=True)
    new_mol2 = renumber_atoms_by_substruct_match_result(mol2, mapping, as_ref=False)

    assert Chem.MolToSmiles(new_mol1) == Chem.MolToSmiles(mol2)
    assert Chem.MolToSmiles(new_mol2) == Chem.MolToSmiles(mol1)


def test_renumber_atoms_for_fragments():
    mol = Chem.MolFromSmiles("[Cl:2][CH2:4][OH:1]", smi_params)
    atom_map_numbers_before = get_atom_map_numbers(mol)

    # Renumber to make atom index and atom map consistent
    # so that comparison can be easier
    new_mol = renumber_atoms(mol, update_atom_map=False)
    atom_map_numbers_after = get_atom_map_numbers(new_mol)
    assert atom_map_numbers_after == sorted(atom_map_numbers_before)
    for atom, atomic_num in zip(new_mol.GetAtoms(), [8, 17, 6]):
        assert atom.GetAtomicNum() == atomic_num


@pytest.mark.parametrize(
    "smiles, if_needs_renumber",
    [
        ("[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1", False),
        ("[CH2:1]1[CH2:3][CH2:2][CH2:4][CH2:5][CH2:6]1", True),
    ],
)
def test_needs_renumber(smiles, if_needs_renumber):
    mol = Chem.MolFromSmiles(smiles, smi_params)
    assert needs_renumber(mol) == if_needs_renumber


@pytest.mark.parametrize(
    "smiles, if_has_numbers",
    [
        ("C1CCCCC1", False),
        ("[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1", True),
    ],
)
def test_has_atom_map_numbers(smiles, if_has_numbers):
    mol = Chem.MolFromSmiles(smiles, smi_params)
    assert has_atom_map_numbers(mol) == if_has_numbers


def test_reverse_match():
    """
    Test the functionality to reverse a mapping.
    """
    map = [
        1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22, 23, 24, 25, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39
    ]
    r_map = [
        0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 5,
        6, 7, 8, 9, 10, 11, 12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38
    ]

    np.testing.assert_equal(np.array(r_map), np.array(reverse_map(map)))


def test_move_atommaps_to_notes():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    atom_map_numbers_before = get_atom_map_numbers(mol)
    move_atommaps_to_notes(mol)
    atom_map_numbers_after = get_atom_map_numbers(mol)
    assert atom_map_numbers_after == [0] * len(atom_map_numbers_before)
    assert [int(atom.GetProp('atomNote')) for atom in mol.GetAtoms()] == atom_map_numbers_before


def test_move_notes_to_atommaps():
    mol = Chem.MolFromSmiles("[CH3][CH2][OH]")
    atom_notes = []
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx() + 1))
        atom_notes.append(atom.GetIdx() + 1)
    atom_map_numbers_before = get_atom_map_numbers(mol)
    move_notes_to_atommaps(mol)
    atom_map_numbers_after = get_atom_map_numbers(mol)
    assert atom_map_numbers_before == [0] * len(atom_map_numbers_before)
    assert atom_notes == atom_map_numbers_after
