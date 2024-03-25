from rdkit.Chem.Draw import rdMolDraw2D

from rdtools.atommap import move_atommaps_to_notes


def draw_reaction(
    rxn: 'ChemicalReaction',
    figsize: tuple = (800, 300),
    font_scale: float = 1.0,
    highlight_by_reactant: bool = True,
):
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
