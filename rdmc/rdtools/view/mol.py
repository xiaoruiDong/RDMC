import logging
from rdkit import Chem
from rdmc.rdtools.view.base import base_viewer

logger = logging.getLogger(__name__)


def mol_viewer(
    mol: 'Mol',
    conf_id: int = 0,
    **kwargs,
) -> 'py3Dmol.view':
    """
    Create a viewer for viewing the RDKit Molecule. This viewer
    accepts additional keyword arguments for viewer specs,
    following the same way base_viewer is implemented.

    Args:
        mol (Mol): The RDKit Molecule.
        conf_id (int): The ID of the conformer to view.

    Returns:
        py3Dmol.view: The viewer.
    """
    try:
        obj, model = Chem.MolToMolBlock(mol, confId=conf_id), 'sdf'
    except BaseException as e:
        logger.debug(f"Failed to convert mol to MolBlock when using mol_viewer. Got: {e}")
        obj, model = Chem.MolToXYZBlock(mol, confId=conf_id), 'xyz'

    return base_viewer(obj, model, **kwargs)
