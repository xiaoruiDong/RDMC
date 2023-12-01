import pytest

from rdkit import Chem

from rdmc.rdtools.atom import (
    decrement_radical,
    electronegativity,
    get_electronegativity,
    get_lone_pair,
    get_num_occupied_orbitals,
    get_total_bond_order,
    has_empty_orbitals,
    increment_radical,
)


smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


@pytest.mark.parametrize(
    "smi",
    [
        "[CH3]",
        "[CH2]",
        "[CH]",
        "[C]",
    ],
)
def test_decrement_radical(smi):
    mol = Chem.MolFromSmiles(smi, smi_params)
    atom = mol.GetAtomWithIdx(0)
    radical_count_before = atom.GetNumRadicalElectrons()
    decrement_radical(atom)
    radical_count_after = atom.GetNumRadicalElectrons()
    assert radical_count_after == radical_count_before - 1


def test_decrement_radical_raises_ValueError():
    mol = Chem.MolFromSmiles("[CH4]", smi_params)
    atom = mol.GetAtomWithIdx(0)
    atom.SetNumRadicalElectrons(0)
    with pytest.raises(ValueError):
        decrement_radical(atom)


def test_get_electronegativity():
    for i in range(119):
        X = get_electronegativity(Chem.Atom(i))
        if i in electronegativity:
            assert X == electronegativity[i]
        else:
            assert X == 1.0


@pytest.mark.parametrize(
    "smi, bo",
    [
        ("C", 4),
        ("[C]([H])([H])([H])[H]", 4),
        ("[CH3]", 3),
        ("[C]([H])([H])[H]", 3),
        ("[CH]([H])[H]", 3),
        ("c1ccccc1", 4),
        ("c1([H])ccccc1", 4),
        ("[C]", 0),
    ],
)
def test_get_total_bond_order(smi, bo):
    mol = Chem.MolFromSmiles(smi, smi_params)
    atom = mol.GetAtomWithIdx(0)
    assert get_total_bond_order(atom) == bo


@pytest.mark.parametrize(
    "smi, expect",
    [
        ("C", 0),
        ("[CH3]", 0),
        ("N", 1),
        ("O", 2),
        ("[NH4+]", 0),
        ("[OH-]", 3),
        ("[F]", 3),
    ],
)
def test_get_lone_pair(smi, expect):
    mol = Chem.MolFromSmiles(smi, smi_params)
    atom = mol.GetAtomWithIdx(0)
    assert get_lone_pair(atom) == expect


@pytest.mark.parametrize(
    "smi, expect",
    [
        ("[H]", 1),
        ("[H+]", 0),
        ("[H][H]", 1),
        ("[BH3]", 3),
        ("C", 4),
        ("[CH3]", 4),
        ("N", 4),
        ("O", 4),
        ("[NH4+]", 4),
        ("[OH-]", 4),
        ("[F]", 4),
        ("[S](=O)(C)C", 5),
        ("[S+](F)(F)(F)(F)F", 5),
        ("S(F)(F)(F)(F)(F)F", 6),
        ("[I+](F)(F)(F)(F)(F)F", 6)
        # ("I(F)(F)(F)(F)(F)(F)F", 7),  # RDKit does not support generating IF7 from SMILES
    ],
)
def test_get_num_occupided_orbitals(smi, expect):
    mol = Chem.MolFromSmiles(smi, smi_params)
    atom = mol.GetAtomWithIdx(0)
    assert get_num_occupied_orbitals(atom) == expect


def test_get_num_occupided_orbitals_carbenes():
    mol = Chem.MolFromSmiles("[CH2]", smi_params)
    atom = mol.GetAtomWithIdx(0)
    # Triplet carbene
    atom.SetNumRadicalElectrons(2)
    assert get_num_occupied_orbitals(atom) == 4
    # Singlet carbene
    atom.SetNumRadicalElectrons(0)
    assert get_num_occupied_orbitals(atom) == 3


@pytest.mark.parametrize(
    "smi, expect",
    [
        ("[H]", False),
        ("[H+]", True),
        ("[H][H]", False),
        ("[BH3]", True),
        ("C", False),
        ("[CH3]", False),
        ("N", False),
        ("O", False),
        ("[NH4+]", False),
        ("[OH-]", False),
        ("[F]", False),
        ("[S](=O)(C)C", True),
        ("[S+](F)(F)(F)(F)F", True),
        ("S(F)(F)(F)(F)(F)F", False),
        ("[I+](F)(F)(F)(F)(F)F", True)
        # ("I(F)(F)(F)(F)(F)(F)F", 7),  # RDKit does not support generating IF7 from SMILES
    ],
)
def test_has_empty_orbitals(smi, expect):
    mol = Chem.MolFromSmiles(smi, smi_params)
    atom = mol.GetAtomWithIdx(0)
    assert has_empty_orbitals(atom) == expect


def test_increment_radical():
    mol = Chem.MolFromSmiles("[CH4]", smi_params)
    atom = mol.GetAtomWithIdx(0)
    radical_count_before = atom.GetNumRadicalElectrons()
    increment_radical(atom)
    radical_count_after = atom.GetNumRadicalElectrons()
    assert radical_count_after == radical_count_before + 1
