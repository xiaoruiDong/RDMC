import pytest

import numpy as np

from rdtools.mol import (
    combine_mols,
    get_element_symbols,
    get_element_counts,
    get_atomic_nums,
    get_atom_masses,
    get_closed_shell_mol,
)

from rdmc.mol import RDKitMol


def test_combine_mol_without_conformer():
    mol1 = RDKitMol.FromSmiles("C")
    mol2 = RDKitMol.FromSmiles("CC")

    combined_mol = combine_mols(mol1, mol2)
    assert combined_mol.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()


def test_combine_mols_1():
    mol1 = RDKitMol.FromSmiles("CC")
    mol2 = RDKitMol.FromSmiles("C")
    mol1.EmbedNullConformer()
    mol2.EmbedNullConformer()

    pos1 = mol1.GetPositions()
    pos2 = mol2.GetPositions()

    offset = np.array([1, 0, 0], dtype=float)
    combined_mol = combine_mols(mol1, mol2, offset=offset, c_product=False)  # Return Chem.Mol

    assert combined_mol.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()
    assert combined_mol.GetNumConformers() == 1
    np.testing.assert_allclose(
        combined_mol.GetConformer().GetPositions(),
        np.concatenate([pos1, pos2 + offset]),
    )


def test_combine_mols_2():
    mol1 = RDKitMol.FromSmiles("CC")
    mol2 = RDKitMol.FromSmiles("C")
    n_conf1, n_conf2 = 2, 3
    mol1.EmbedMultipleNullConfs(n_conf1, random=True)
    mol2.EmbedMultipleNullConfs(n_conf2, random=True)

    offset = np.array([1, 0, 0], dtype=float)
    combined_mol = combine_mols(mol1, mol2, offset=offset, c_product=True)

    assert combined_mol.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()
    assert combined_mol.GetNumConformers() == n_conf1 * n_conf2
    for i in range(n_conf1):
        for j in range(n_conf2):
            np.testing.assert_allclose(
                combined_mol.GetConformer(i * n_conf2 + j).GetPositions(),
                np.concatenate(
                    [
                        mol1.GetConformer(i).GetPositions(),
                        mol2.GetConformer(j).GetPositions() + offset,
                    ]
                ),
                verbose=True,
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
    mol = RDKitMol.FromSmiles(smi)
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
    mol = RDKitMol.FromSmiles(smi)
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
    mol = RDKitMol.FromSmiles(smi)
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
    mol = RDKitMol.FromSmiles(smi)
    np.testing.assert_allclose(
        np.array(get_atom_masses(mol)),
        np.array(expected)
    )


@pytest.mark.parametrize(
    "rad_smi, expect_smi",
    [
        ("[CH3]", "C"),
        ("c1[c]cccc1", "c1ccccc1"),
        ("C[NH2]", "CN"),
        ("[CH2]C[CH2]", "CCC"),
        ("C", "C"),
    ],
)
@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("atommap", [True, False])
def test_get_closed_shell_mol(rad_smi, expect_smi, explicit, atommap):

    rad_mol = RDKitMol.FromSmiles(rad_smi, assignAtomMap=atommap)
    cs_mol = get_closed_shell_mol(rad_mol, explicit=explicit)
    assert cs_mol.ToSmiles() == expect_smi
