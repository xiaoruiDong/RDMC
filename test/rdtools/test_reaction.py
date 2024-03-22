import pytest

from rdkit.Chem import rdChemReactions

from rdtools.reaction.draw import draw_reaction


@pytest.mark.parametrize(
    "smi",
    [
        "[CH3:1][CH3:2]>>[CH2:1]=[CH2:2].[H:3][H:4]",
        "[C:1]([C:2]([H:4])([H:7])[H:8])([H:3])([H:5])[H:6]>>"
        "[C:1](=[C:2]([H:7])[H:8])([H:5])[H:6].[H:3][H:4]",
    ]
)
@pytest.mark.parametrize(
    "figsize", [(800, 300), (400, 400)]
)
@pytest.mark.parametrize(
    "font_scale", [0.5, 1.0, 2.0]
)
@pytest.mark.parametrize(
    "highlight_atoms", [True, False]
)
def test_draw_reaction(smi, figsize, font_scale, highlight_atoms):
    """
    Test issue-free in the workflow drawing a reaction
    """
    rxn = rdChemReactions.ReactionFromSmarts(
        smi,
        useSmiles=True,
    )
    draw_reaction(rxn, figsize, font_scale, highlight_atoms)
