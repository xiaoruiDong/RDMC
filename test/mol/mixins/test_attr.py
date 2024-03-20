import pytest
import numpy as np
from rdkit import Chem

from rdmc import RDKitMol


def test_get_finger_print():
    """
    Test the function that generates molecular finger prints.
    """
    # We only test one case here to check the functionality of the function
    # other cases are covered by test_fingerprints
    smi = "O=C(Nc1cc2c(cn1)CCCC2)N1CCCC1c1ccc(O)cc1"
    fp = RDKitMol.FromSmiles(smi, addHs=False).GetFingerprint(
        fpType="morgan", numBits=2048, count=True, radius=3
    )
    fp_expect = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=2048
    ).GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smi))
    assert np.isclose(fp, fp_expect).all()


@pytest.mark.parametrize(
    "smi, exp_bonds",
    [
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            [(0, 1), (1, 2), (1, 5), (1, 6), (2, 3), (2, 4)],
        ),
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [(1, 2), (1, 3), (1, 4), (1, 5)],
        ),
    ],
)
def test_get_bond_as_tuples(smi, exp_bonds):
    """
    Test getBondsAsTuples returns a list of atom pairs corresponding to each bond.
    """
    # Single molecule
    mol = RDKitMol.FromSmiles(smi)
    bonds = mol.GetBondsAsTuples()
    assert set(bonds) == set(exp_bonds)


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [1, 6, 1, 1, 1, 1],
        ),
        ("[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]", [8, 6, 6, 1, 1, 1, 1]),
    ],
)
def test_get_atomic_nums(smi, expected):
    assert RDKitMol.FromSmiles(smi).GetAtomicNumbers() == expected


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [1.008, 12.011, 1.008, 1.008, 1.008, 1.008],
        ),
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            [15.999, 12.011, 12.011, 1.008, 1.008, 1.008, 1.008],
        ),
    ],
)
def test_get_atom_masses(smi, expected):
    mol = RDKitMol.FromSmiles(smi)
    np.testing.assert_allclose(np.array(mol.GetAtomMasses()), np.array(expected))


@pytest.mark.parametrize(
    "smi, expected",
    [
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            ["H", "C", "H", "H", "H", "H",],
        ),
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            ["O", "C", "C", "H", "H", "H", "H"],
        ),
    ],
)
def test_get_element_symbols(smi, expected):
    assert RDKitMol.FromSmiles(smi).GetElementSymbols() == expected


@pytest.mark.parametrize(
    "smi",
    [
        "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
        "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
    ]
)
def test_get_atoms(smi):
    """
    Test the rewrite version of GetAtoms returns the same results as Mol.GetAtoms.
    """
    mol = RDKitMol.FromSmiles(smi)
    assert np.all(
        [
            atom1.GetIdx() == atom2.GetIdx()
            for atom1, atom2 in zip(mol.GetAtoms(), super(Chem.RWMol, mol).GetAtoms())
        ]
    )
