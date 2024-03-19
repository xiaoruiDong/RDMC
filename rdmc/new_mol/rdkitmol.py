from rdkit import Chem

from rdmc.new_mol.conf import MolConfMixin
from rdmc.new_mol.compare import MolCompareMixin
from rdmc.new_mol.transform import MolTransformMixin
from rdmc.new_mol.attribute import MolAttrMixin
from rdmc.new_mol.ops import MolOpsMixin


class RDKitMol(
    MolConfMixin,
    MolCompareMixin,
    MolTransformMixin,
    MolAttrMixin,
    MolOpsMixin,
    Chem.RWMol
):
    """
    A helpful wrapper for ``Chem.RWMol``.
    The method nomenclature follows the Camel style to be consistent with RDKit.
    It keeps almost all of the original methods of ``Chem.RWMol`` but has a few useful
    shortcuts so that users don't need to refer to other RDKit modules.
    """

Mol = RDKitMol
