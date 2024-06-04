import pytest

from rdtools.compare import get_match_and_recover_recipe, get_unique_mols, has_matched_mol, is_same_connectivity_mol, is_symmetric_to_substructure
from rdtools.conversion import mol_from_smiles, mol_to_smiles

from rdkit.Chem import MolFromSmarts

@pytest.mark.parametrize(
    "smi1, smi2, expected",
    [
        (
            "[H:20].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
            "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:18])[H:19])([H:9])[H:10])[H:8]",
            "[H:18].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
            "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:19])[H:20])([H:9])[H:10])[H:8]",
            (
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 18, 17),
                {17: 19, 19: 17},
            ),
        ),
        (
            "[H:20].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
            "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:18])[H:19])([H:9])[H:10])[H:8]",
            "[H:20].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
            "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:18])[H:19])([H:9])[H:10])[H:8]",
            (
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
                {},
            ),
        ),
        (
            "[H:20].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
            "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:18])[H:19])([H:9])[H:10])[H:8]",
            "[H:20].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
            "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:18])[H:19])([H:9])[H:10])[H:8]",
            (
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
                {},
            ),
        ),
        (
            "[O:1][O:2][H:3]",
            "[H:1][O:2][H:3]",
            ((), {}),
        ),
    ],
)
def test_get_match_and_recover_recipe(smi1, smi2, expected):

    mol1 = mol_from_smiles(smi1)
    mol2 = mol_from_smiles(smi2)
    assert expected == get_match_and_recover_recipe(mol1, mol2)


def test_has_matched_mol():
    """
    Test the function that indicates if there is a matched molecule to the query molecule from a list.
    """
    query = '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]'
    query_mol = mol_from_smiles(query)

    list1 = [mol_from_smiles(smi) for smi in ['C', 'O']]
    list2 = [mol_from_smiles(smi) for smi in ['C', 'O', '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]']]
    list3 = [mol_from_smiles(smi) for smi in ['C', 'O', '[C:1]([O:6][H:2])([H:3])([H:4])[H:5]']]

    assert not has_matched_mol(
        query_mol,
        list1,
    )
    assert has_matched_mol(
        query_mol,
        list2,
    )
    assert has_matched_mol(
        query_mol,
        list2,
        consider_atommap=True,
    )
    assert has_matched_mol(
        query_mol,
        list3,
    )
    assert not has_matched_mol(
        query_mol,
        list3,
        consider_atommap=True,
    )


def test_get_unique_mols():
    """
    Test the function that extract unique molecules from a list of molecules.
    """
    list1 = ['C', 'O']
    list2 = [
        'C', 'O',
        '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]',
        '[C:1]([H:3])([H:4])([H:5])[O:6][H:2]',
    ]

    list1 = [mol_from_smiles(smi) for smi in ["C", "O"]]
    list2 = [
        mol_from_smiles(smi)
        for smi in [
            "C",
            "O",
            "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
            "[C:1]([H:3])([H:4])([H:5])[O:6][H:2]",
        ]
    ]

    assert len(get_unique_mols(list1)) == 2
    assert set(
        [mol_to_smiles(mol) for mol in get_unique_mols(list1)]
    ) == {'C', 'O'}
    assert len(get_unique_mols(
        list2,
        consider_atommap=True
    )) == 4
    assert set(
        [
            mol_to_smiles(mol, remove_hs=False, remove_atom_map=False)
            for mol in get_unique_mols(
                list2,
                consider_atommap=True,
            )
        ]
    ) == {
        '[O:1]([H:2])[H:3]',
        '[C:1]([H:2])([H:3])([H:4])[H:5]',
        '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]',
        '[C:1]([H:3])([H:4])([H:5])[O:6][H:2]',
    }
    assert len(get_unique_mols(
        list2, consider_atommap=False
    )) == 3


@pytest.mark.parametrize(
    'smi1, smi2, expect_match',
    [
        ('[CH3]', 'C', False),
        ('[O:1]([H:2])[H:3]', '[O:1]([H:2])[H:3]', True),
        ('[O:1]([H:2])[H:3]', '[O:2]([H:1])[H:3]', False),
        ('[H:1][C:2](=[O:3])[O:4]', '[H:1][C:2]([O:3])=[O:4]', True)
    ])
def test_has_same_connectivity(smi1, smi2, expect_match):
    mol1 = mol_from_smiles(smi1)
    mol2 = mol_from_smiles(smi2)
    assert is_same_connectivity_mol(mol1, mol2) == expect_match


@pytest.mark.parametrize(
    'smi, sma, expect_match',
    [
        ('CC(=O)C', '[CX3]=[OX1]', True),
        ('CC(=O)C(=O)C', '[CX3]=[OX1]', True),
        ('CC(=O)CC(=O)', '[CX3]=[OX1]', False),
        ('C', '[CX3]=[OX1]', False),
        ('OCC(CO)(CO)CO', '[CX4]-[OX2]', True),
    ])
def test_is_symmetric_to_substructure(smi, sma, expect_match):
    mol = mol_from_smiles(smi)
    substructure = MolFromSmarts(sma)
    assert is_symmetric_to_substructure(mol, substructure) == expect_match
