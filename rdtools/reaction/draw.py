"""Draw a reaction using RDKit."""

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdChemReactions import ChemicalReaction

from rdtools.atommap import move_atommaps_to_notes


def draw_reaction(
    rxn: ChemicalReaction,
    figsize: tuple[int, int] = (800, 300),
    font_scale: float = 1.0,
    highlight_by_reactant: bool = True,
) -> str:
    """Generate SVG str for a RDKit reaction.

    Args:
        rxn (ChemicalReaction): The reaction to be drawn.
        figsize (tuple[int, int], optional): The size of the figure. Defaults to ``(800, 300)``.
        font_scale (float, optional): The font scale. Defaults to ``1.0``.
        highlight_by_reactant (bool, optional): Whether to highlight by reactant. Defaults to ``True``.

    Returns:
        str: The SVG string of the reaction.
    """
    # move atom maps to be annotations:
    for mol in rxn.GetReactants():
        move_atommaps_to_notes(mol, clear_atommap=False)
    for mol in rxn.GetProducts():
        move_atommaps_to_notes(mol, clear_atommap=False)

    d2d = rdMolDraw2D.MolDraw2DSVG(*figsize)
    draw_options = d2d.drawOptions()
    draw_options.annotationFontScale = font_scale
    draw_options.includeRadicals = True
    d2d.DrawReaction(rxn, highlightByReactant=highlight_by_reactant)

    d2d.FinishDrawing()

    return d2d.GetDrawingText()
