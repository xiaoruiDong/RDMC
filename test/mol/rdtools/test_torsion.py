import pytest

from rdmc.new_mol import RDKitMol
from rdmc.rdtools.torsion import get_torsional_modes, get_torsion_tops


@pytest.mark.parametrize(
    "smi, methyl_torsions, other_torsions, ring_torsions",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])[H:5]",
            [],
            [],
            [],
        ),
        (
            "[C:1]([O:2][H:3])([H:4])([H:5])[H:6]",
            [[3, 0, 1, 2]],
            [],
            [],
        ),
        (
            "[c:1]1([H:2])[c:3]([H:4])[c:5]([H:6])[c:7]([H:8])[c:9]([H:10])[c:11]1[H:12]",
            [],
            [],
            [
                [10, 0, 2, 4],
                [0, 2, 4, 6],
                [2, 4, 6, 8],
                [4, 6, 8, 10],
                [6, 8, 10, 0],
                [2, 0, 10, 8],
            ],
        ),
        (
            "[C:1]1([H:2])([H:3])[C:4]([H:5])([H:6])[C:7]1([C:8]([C:9]1([H:10])[C:11]([H:12])([H:13])[C:14]([H:15])"
            "([H:16])[C:17]([H:18])([H:19])[C:20]1([H:21])[H:22])([H:23])[H:24])[H:25]",
            [],
            [[0, 6, 7, 8], [6, 7, 8, 10]],
            [
                [0, 3, 6, 0],
                [6, 0, 3, 6],
                [3, 0, 6, 3],
                [10, 8, 19, 16],
                [13, 16, 19, 8],
                [10, 13, 16, 19],
                [8, 10, 13, 16],
                [19, 8, 10, 13],
            ],
        ),
    ],
)
@pytest.mark.parametrize("exclude_methyl", [True, False])
@pytest.mark.parametrize("include_ring", [True, False])
def test_get_torsional_modes(
    smi, methyl_torsions, other_torsions, ring_torsions, exclude_methyl, include_ring
):
    mol = RDKitMol.FromSmiles(smi)

    torsions = [
        tuple(tor)
        for tor in get_torsional_modes(
            mol, exclude_methyl=exclude_methyl, include_ring=include_ring
        )
    ]

    expect_tors = other_torsions[:]
    if not exclude_methyl:
        expect_tors += methyl_torsions
    if include_ring:
        expect_tors += ring_torsions

    expect_tors = [tuple(tor) for tor in expect_tors]

    assert set(torsions) == set(expect_tors)


def test_get_torsion_tops():
    """
    Test get torsion tops of a given molecule.
    """
    smi1 = "[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]"
    mol = RDKitMol.FromSmiles(smi1)
    tops = get_torsion_tops(mol, [2, 0, 1, 5])
    assert len(tops) == 2
    assert set(tops) == {(0, 2, 3, 4), (1, 5, 6, 7)}
    tops = get_torsion_tops(mol, [0, 1], allow_non_bonding_pivots=True)
    assert len(tops) == 2
    assert set(tops) == {(0, 2, 3, 4), (1, 5, 6, 7)}

    smi2 = "[C:1]([C:2]#[C:3][C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]"
    mol = RDKitMol.FromSmiles(smi2)
    with pytest.raises(ValueError):
        get_torsion_tops(mol, [4, 0, 3, 7])
    tops = get_torsion_tops(mol, [4, 0, 3, 7], allow_non_bonding_pivots=True)
    assert len(tops) == 2
    assert set(tops) == {(0, 4, 5, 6), (3, 7, 8, 9)}

    smi3 = "[C:1]([H:3])([H:4])([H:5])[H:6].[O:2][H:7]"
    mol = RDKitMol.FromSmiles(smi3).AddBonds([(1,2)])
    with pytest.raises(ValueError):
        get_torsion_tops(mol, [3, 0, 1, 6])
    tops = get_torsion_tops(mol, [3, 0, 1, 6], allow_non_bonding_pivots=True)
    assert len(tops) == 2
    assert set(tops) == {(0, 3, 4, 5), (1, 6)}
