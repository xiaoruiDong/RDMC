import pytest

from rdmc import RDKitMol

@pytest.mark.parametrize(
    'smi1, smi2, expect_match',
    [
        ('[CH3]', 'C', False),
        ('[O:1]([H:2])[H:3]', '[O:1]([H:2])[H:3]', True),
        ('[O:1]([H:2])[H:3]', '[O:2]([H:1])[H:3]', False),
        ('[H:1][C:2](=[O:3])[O:4]', '[H:1][C:2]([O:3])=[O:4]', True)
    ])
def test_is_same_connectivity(smi1, smi2, expect_match):
    mo1 = RDKitMol.FromSmiles(smi1)
    mol2 = RDKitMol.FromSmiles(smi2)
    assert mo1.IsSameConnectivity(mol2) == expect_match


@pytest.mark.parametrize(
    "smi1, smi2, expect_match",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])[H:5]",
            "[H:1][C:2]([H:3])([H:4])[H:5]",
            ((1, 0, 2, 3, 4), {0: 1, 1: 0})
        ),
        (
            "[C:1]([H:2])([H:3])[H:4].[H:5]",
            "[H:1][C:2]([H:3])[H:5].[H:4]",
            ((1, 0, 2, 4, 3), {0: 1, 1: 0, 3: 4, 4: 3})
        )
    ]
)
def get_match_and_recover_recipe(smi1, smi2, expect_match):
    """
    Test GetMatchAndRecoverRecipe
    """
    mol1 = RDKitMol.FromSmiles(smi1)
    mol2 = RDKitMol.FromSmiles(smi2)

    match, recipe = mol1.GetMatchAndRecoverRecipe(mol2)
    assert match == expect_match[0]
    assert recipe == expect_match[1]
