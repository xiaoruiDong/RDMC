from rdkit import Chem

from rdmc.mol.conf import MolConfMixin
from rdmc.mol.compare import MolCompareMixin
from rdmc.mol.transform import MolTransformMixin
from rdmc.mol.attribute import MolAttrMixin
from rdmc.mol.ops import MolOpsMixin


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

    def __copy__(self):
        return RDKitMol(self)


Mol = RDKitMol
