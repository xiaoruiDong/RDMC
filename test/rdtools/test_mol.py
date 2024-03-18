import pytest

import numpy as np

from rdmc.rdtools.conf import (
    add_null_conformer,
)
from rdmc.rdtools.conversion import (
    mol_from_smiles,
)
from rdmc.rdtools.mol import (
    combine_mols,
    get_element_symbols,
    get_element_counts,
    get_atomic_nums,
)
from rdmc.rdtools.conf import (
    embed_multiple_null_confs,
)

from rdkit import Chem


def test_combine_mol_without_conformer():
    mol1 = Chem.MolFromSmiles("C")
    mol2 = Chem.MolFromSmiles("CC")

    combined_mol = combine_mols(mol1, mol2)
    assert combined_mol.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()


def test_combine_mols_1():
    mol1 = Chem.MolFromSmiles("CC")
    mol2 = Chem.MolFromSmiles("C")
    add_null_conformer(mol1, random=True)
    add_null_conformer(mol2, random=True)

    pos1 = mol1.GetConformer().GetPositions()
    pos2 = mol2.GetConformer().GetPositions()

    offset = np.array([1, 0, 0], dtype=float)
    combined_mol = combine_mols(mol1, mol2, offset=offset, c_product=False)

    assert combined_mol.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()
    assert combined_mol.GetNumConformers() == 1
    np.testing.assert_allclose(
        combined_mol.GetConformer().GetPositions(),
        np.concatenate([pos1, pos2 + offset]),
    )


def test_combine_mols_2():
    mol1 = Chem.MolFromSmiles("CC")
    mol2 = Chem.MolFromSmiles("C")
    n_conf1, n_conf2 = 2, 3
    embed_multiple_null_confs(mol1, n_conf1, random=True)
    embed_multiple_null_confs(mol2, n_conf2, random=True)

    offset = np.array([1, 0, 0], dtype=float)
    combined_mol = combine_mols(mol1, mol2, offset=offset, c_product=True)

    assert combined_mol.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()
    assert combined_mol.GetNumConformers() == n_conf1 * n_conf2
    for i in range(n_conf1):
        for j in range(n_conf2):
            assert np.allclose(
                combined_mol.GetConformer(i * n_conf2 + j).GetPositions(),
                np.concatenate(
                    [
                        mol1.GetConformer(i).GetPositions(),
                        mol2.GetConformer(j).GetPositions() + offset,
                    ]
                ),
            )


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [
                "H",
                "C",
                "H",
                "H",
                "H",
                "H",
            ],
        ),
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            ["O", "C", "C", "H", "H", "H", "H"],
        ),
    ],
)
def test_get_element_symbols(smi, expected):
    mol = mol_from_smiles(smi)
    assert get_element_symbols(mol) == expected


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            {"C": 1, "H": 5},
        ),
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            {"C": 2, "H": 4, "O": 1},
        ),
    ],
)
def test_get_element_counts(smi, expected):
    mol = mol_from_smiles(smi)
    assert get_element_counts(mol) == expected


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [1, 6, 1, 1, 1, 1],
        ),
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            [8, 6, 6, 1, 1, 1, 1]
        ),
    ],
)
def test_get_atomic_nums(smi, expected):
    mol = mol_from_smiles(smi)
    assert get_atomic_nums(mol) == expected
