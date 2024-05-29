import pytest

import numpy as np

from rdtools.conf import (
    add_null_conformer,
)
from rdtools.conversion import (
    mol_from_smiles,
    mol_to_smiles,
)
from rdtools.mol import (
    combine_mols,
    get_dehydrogenated_mol,
    get_element_symbols,
    get_element_counts,
    get_atomic_nums,
    get_atom_masses,
    get_closed_shell_mol,
    get_formal_charge,
)
from rdtools.conf import (
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


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [1.008, 12.011, 1.008, 1.008, 1.008, 1.008],
        ),
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            [15.999, 12.011, 12.011, 1.008, 1.008, 1.008, 1.008]),
    ],
)
def test_get_atom_masses(smi, expected):
    mol = mol_from_smiles(smi)
    np.testing.assert_allclose(
        np.array(get_atom_masses(mol)),
        np.array(expected)
    )


@pytest.mark.parametrize(
    "rad_smi, expect_smi",
    [
        ("[CH2]", "C"),
        ("[CH3]", "C"),
        ("c1[c]cccc1", "c1ccccc1"),
        ("C[NH]", "CN"),
        ("[CH2]C[CH2]", "CCC"),
        ("C", "C"),
    ],
)
@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("atommap", [True, False])
def test_get_closed_shell_mol(rad_smi, expect_smi, explicit, atommap):

    rad_mol = mol_from_smiles(rad_smi, assign_atom_map=atommap)
    cs_mol = get_closed_shell_mol(rad_mol, explicit=explicit)
    assert mol_to_smiles(cs_mol) == expect_smi


def test_get_closed_shell_mol_one_hs():

    rad_mol = mol_from_smiles("[CH2]")
    cs_mol = get_closed_shell_mol(rad_mol)
    assert mol_to_smiles(cs_mol) == "C"

    rad_mol = mol_from_smiles("[C]")
    cs_mol = get_closed_shell_mol(rad_mol, max_num_hs=1)
    assert mol_to_smiles(cs_mol) == "[CH]"

    rad_mol = mol_from_smiles("[C]")
    cs_mol = get_closed_shell_mol(rad_mol, max_num_hs=2)
    assert mol_to_smiles(cs_mol) == "[CH2]"

    rad_mol = mol_from_smiles("[C]")
    cs_mol = get_closed_shell_mol(rad_mol, max_num_hs=3)
    assert mol_to_smiles(cs_mol) == "[CH3]"

    rad_mol = mol_from_smiles("[C]")
    cs_mol = get_closed_shell_mol(rad_mol, max_num_hs=4)
    assert mol_to_smiles(cs_mol) == "C"

    rad_mol = mol_from_smiles("[C]")
    cs_mol = get_closed_shell_mol(rad_mol, max_num_hs=5)
    assert mol_to_smiles(cs_mol) == "C"

    rad_mol = mol_from_smiles("[C]", add_hs=False)
    cs_mol = get_closed_shell_mol(rad_mol, max_num_hs=1, explicit=False)
    assert mol_to_smiles(cs_mol) == "[CH]"


@pytest.mark.parametrize(
    "smi, n_result",
    [
        ("C", 1),
        ("c1ccccc1", 6),
        ("CN", 2),
        ("CO", 2),
    ],
)
@pytest.mark.parametrize(
    "kind, charge",
    [
        ("cation", 1),
        ("anion", -1),
        ("radical", 0),
    ],
)
def test_dehydrogenated_mol(smi, n_result, kind, charge):
    mol = mol_from_smiles(smi)
    children = get_dehydrogenated_mol(mol, kind=kind)
    assert len(children) == n_result
    for child in children:
        assert get_formal_charge(child) == charge
