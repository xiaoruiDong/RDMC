import pytest

from rdkit import Chem

from rdtools.bond import (
    add_bond,
    add_bonds,
    get_bonds_as_tuples,
    _get_bonds_as_sets,
    get_formed_bonds,
    get_broken_bonds,
    get_formed_and_broken_bonds,
    get_all_changing_bonds,
    get_atoms_in_bonds,
)
from rdtools.conversion import mol_from_smiles


def test_add_bond():
    """
    Test the function ``add_bond``.
    """
    mol0 = Chem.RWMol()
    mol0.AddAtom(Chem.Atom(6))
    mol0.AddAtom(Chem.Atom(6))
    assert len(mol0.GetBonds()) == 0
    mol1 = add_bond(mol0, (0, 1), inplace=False, update_properties=False)
    assert len(mol0.GetBonds()) == 0
    assert len(mol1.GetBonds()) == 1
    assert mol1.NeedsUpdatePropertyCache()
    assert mol1.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE

    mol1.AddAtom(Chem.Atom(6))
    mol2 = add_bond(mol1, (1, 2), Chem.BondType.DOUBLE)
    assert len(mol2.GetBonds()) == len(mol1.GetBonds()) == 2


def test_add_bonds():
    """
    Test the function ``add_bonds``.
    """
    mol0 = Chem.RWMol()
    for _ in range(3):
        mol0.AddAtom(Chem.Atom(6))
    add_bonds(mol0, [(0, 1), (1, 2), (0, 2)], update_properties=False)
    assert len(mol0.GetBonds()) == 3


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
    mol = mol_from_smiles(smi)
    bonds = get_bonds_as_tuples(mol)
    assert set(bonds) == set(exp_bonds)


def test_get_bonds_as_sets():
    """
    Test get bonds for multiple mols.
    """
    mol1 = mol_from_smiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
    mol2 = mol_from_smiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")
    mol3 = mol_from_smiles(
        "[C:1]2([C:2]1[N:3]2[O:4][N:5][N:6]1)([H:7])([H:8])[H:9]", sanitize=False
    )

    assert _get_bonds_as_sets(mol1) == (
        {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
    )
    assert _get_bonds_as_sets(mol1, mol2) == (
        {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
        {(0, 2), (1, 2), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
    )
    assert _get_bonds_as_sets(mol1, mol2, mol3) == (
        {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
        {(0, 2), (1, 2), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
        {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 5),
            (0, 2),
        },
    )


def test_get_formed_bonds():
    """
    Test get formed bonds between the reactant complex and the product complex.
    """
    mol1 = mol_from_smiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
    mol2 = mol_from_smiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

    # The backend molecule should be the same as the input RWMol object
    assert set(get_formed_bonds(mol1, mol2)) == {(0, 2)}
    assert set(get_formed_bonds(mol2, mol1)) == {(0, 1), (2, 3)}


def test_get_broken_bonds():
    """
    Test get broken bonds between the reactant complex and the product complex.
    """
    mol1 = mol_from_smiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
    mol2 = mol_from_smiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

    # The backend molecule should be the same as the input RWMol object
    assert set(get_broken_bonds(mol1, mol2)) == {(0, 1), (2, 3)}
    assert set(get_broken_bonds(mol2, mol1)) == {(0, 2)}


def test_get_formed_and_broken_bonds() -> None:
    """
    Test get formed and broken bonds between the reactant complex and the product complex.
    """
    mol1 = mol_from_smiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
    mol2 = mol_from_smiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

    # The backend molecule should be the same as the input RWMol object
    assert [set(blist) for blist in get_formed_and_broken_bonds(mol1, mol2)] == [
        {(0, 2)},
        {(0, 1), (2, 3)},
    ]
    assert [set(blist) for blist in get_formed_and_broken_bonds(mol2, mol1)] == [
        {(0, 1), (2, 3)},
        {(0, 2)},
    ]


def test_get_all_changing_bonds() -> None:
    """
    Test get formed and broken bonds between the reactant complex and the product complex.
    """
    mol1 = mol_from_smiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
    mol2 = mol_from_smiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

    # The backend molecule should be the same as the input RWMol object
    assert [set(blist) for blist in get_all_changing_bonds(mol1, mol2)] == [
        {(0, 2)},
        {(0, 1), (2, 3)},
        {(1, 2), (4, 5), (1, 5), (3, 4)},
    ]
    assert [set(blist) for blist in get_all_changing_bonds(mol2, mol1)] == [
        {(0, 1), (2, 3)},
        {(0, 2)},
        {(1, 2), (4, 5), (1, 5), (3, 4)},
    ]


@pytest.mark.parametrize(
    "bonds, atoms",
    [
        ([(1, 2)], [1, 2]),
        ([(0, 1), (1, 2)], [0, 1, 2]),
        (
            [
                (0, 1),
                (2, 3),
                (4, 5),
            ],
            [0, 1, 2, 3, 4, 5],
        ),
    ],
)
@pytest.mark.parametrize("sorted", [True, False])
def test_get_atoms_in_bonds(bonds, atoms, sorted):
    result_atoms = get_atoms_in_bonds(bonds, sorted=sorted)

    if sorted:
        assert result_atoms == atoms
    else:
        assert set(result_atoms) == set(atoms)
