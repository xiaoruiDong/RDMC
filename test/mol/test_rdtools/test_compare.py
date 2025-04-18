import pytest

from rdtools.compare import get_match_and_recover_recipe, get_unique_mols, has_matched_mol

from rdmc.mol import RDKitMol


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

    mol1 = RDKitMol.FromSmiles(smi1)
    mol2 = RDKitMol.FromSmiles(smi2)
    assert expected == get_match_and_recover_recipe(mol1, mol2)


def test_has_matched_mol():
    """
    Test the function that indicates if there is a matched molecule to the query molecule from a list.
    """
    query = '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]'
    query_mol = RDKitMol.FromSmiles(query)

    list1 = [RDKitMol.FromSmiles(smi) for smi in ['C', 'O']]
    list2 = [RDKitMol.FromSmiles(smi) for smi in ['C', 'O', '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]']]
    list3 = [RDKitMol.FromSmiles(smi) for smi in ['C', 'O', '[C:1]([O:6][H:2])([H:3])([H:4])[H:5]']]

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
        '[C:1]([H:3])([H:4])([H:5])[O:6][H:2]',]

    list1 = [RDKitMol.FromSmiles(smi) for smi in ["C", "O"]]
    list2 = [
        RDKitMol.FromSmiles(smi)
        for smi in [
            "C",
            "O",
            "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
            "[C:1]([H:3])([H:4])([H:5])[O:6][H:2]",
        ]
    ]

    assert len(get_unique_mols(list1)) == 2
    assert set(
        [mol.ToSmiles() for mol in get_unique_mols(list1)]
    ) == {'C', 'O'}
    assert len(get_unique_mols(
        list2,
        consider_atommap=True
    )) == 4
    assert set(
        [
            mol.ToSmiles(removeHs=False, removeAtomMap=False)
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
