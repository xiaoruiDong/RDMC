import logging

from rdkit import Chem
import numpy as np

from rdtools.obabel import (
    parse_xyz_by_openbabel as xyz_from_openbabel,
    openbabel_mol_to_rdkit_mol,
)

from rdtools.element import PERIODIC_TABLE
from rdtools.conversion.xyz2mol import (
    parse_xyz_by_jensen as parse_xyz_by_xyz2mol_rdmc,
)

logger = logging.getLogger(__name__)

# Since 2022.09.1, RDKit added built-in XYZ parser using xyz2mol approach
try:
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    rdDetermineBonds = None
    logger.warn(
        "The current version of RDKit does not contain built-in xyz2mol."
        "Using the original python implementation instead."
    )


def parse_xyz_by_xyz2mol_rdkit_native(
    xyz: str,
    charge: int = 0,
    allow_charged_fragments: bool = False,
    use_huckel: bool = False,
    embed_chiral: bool = True,
    use_atom_maps: bool = False,
):
    """
    Parse xyz with RDKit's native implementation of xyz2mol.

    Args:
        xyz (str): The xyz string.
        charge (int, optional): The charge of the species. Defaults to ``0``.
        allow_charged_fragments (bool, optional): ``True`` for charged fragment, ``False`` for radical. Defaults to ``False``.
        use_huckel (bool, optional): ``True`` for Huckel method, ``False`` for Gasteiger method. Defaults to False.
        embed_chiral (bool, optional): ``True`` for chiral molecule, ``False`` for non-chiral molecule. Defaults to ``True``.
        use_atom_maps (bool, optional): ``True`` for atom map, ``False`` for non-atom map. Defaults to ``False``.

    Returns:
        Chem.Mol: The RDKit Mol instance.
    """
    try:
        mol = Chem.Mol(Chem.MolFromXYZBlock(xyz))
    except BaseException:
        raise ValueError("Unable to parse the provided xyz.")
    else:
        if mol is None:
            raise ValueError("Unable to parse the provided xyz.")

    # Uni-atom molecule
    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        # No implicit Hs for single atom molecule
        atom.SetNoImplicit(True)
        valence = PERIODIC_TABLE.GetDefaultValence(atom.GetAtomicNum())
        atom.SetFormalCharge(charge)
        atom.SetNumRadicalElectrons(valence - atom.GetFormalCharge())
        return mol

    # Multi-atom molecule
    rdDetermineBonds.DetermineConnectivity(
        mol,
        useHueckel=use_huckel,
        charge=charge,
    )
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=charge,
        allowChargedFragments=allow_charged_fragments,
        embedChiral=embed_chiral,
        useAtomMap=use_atom_maps,
    )
    return mol


def parse_xyz_by_xyz2mol(
    xyz: str,
    charge: int = 0,
    allow_charged_fragments: bool = False,
    use_huckel: bool = False,
    embed_chiral: bool = True,
    use_atom_maps: bool = False,
    force_rdmc: bool = False,
    **kwargs,
) -> "Mol":
    """
    Perceive a xyz str using `xyz2mol` by Jensen et al. and generate the corresponding RDKit Mol.
    The implementation refers the following blog: https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html

    Args:
        charge: The charge of the species. Defaults to ``0``.
        allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to False.
        use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to False.
        embed_chiral: ``True`` to embed chiral information. Defaults to True.
        use_atom_maps(bool, optional): ``True`` to set atom map numbers to the molecule. Defaults to ``False``.
        force_rdmc (bool, optional): Defaults to ``False``. In rare case, we may hope to use a tailored
                                     version of the Jensen XYZ parser, other than the one available in RDKit.
                                     Set this argument to ``True`` to force use RDMC's implementation,
                                     which user's may have some flexibility to modify.

    Returns:
        Mol: A RDKit Mol corresponding to the xyz.
    """
    if rdDetermineBonds is None or force_rdmc:
        return parse_xyz_by_xyz2mol_rdmc(
            xyz=xyz,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
            use_graph=True,
            use_huckel=use_huckel,
            embed_chiral=embed_chiral,
            use_atom_maps=use_atom_maps,
        )

    return parse_xyz_by_xyz2mol_rdkit_native(
        xyz=xyz,
        charge=charge,
        allow_charged_fragments=allow_charged_fragments,
        use_huckel=use_huckel,
        embed_chiral=embed_chiral,
        use_atom_maps=use_atom_maps,
    )


def parse_xyz_by_openbabel(
    xyz: str,
    embed_chiral: bool = True,
    sanitize: bool = True,
):
    """
    Parse xyz with openbabel.

    Args:
        xyz (str): The xyz string.
        embed_chiral (bool, optional): ``True`` to embed chiral information. Defaults to ``True``.
        sanitize (bool, optional): ``True`` to sanitize the molecule. Defaults to ``True``.

    Returns:
        Chem.Mol: The RDKit Mol instance.
    """
    try:
        obmol = xyz_from_openbabel(xyz)
    except TypeError:   # xyz_from_openbabel is None if openbabel is not available
        raise ImportError(
            "Unable to parse XYZ with openbabel as openbabel is not installed. Please install openbabel first."
        )
    rdmol = openbabel_mol_to_rdkit_mol(obmol, remove_hs=False, sanitize=sanitize)
    if embed_chiral:
        Chem.AssignStereochemistryFrom3D(rdmol)
    return rdmol


def add_header_to_xyz(xyz: str, title: str = "") -> str:
    """
    Add header to xyz string.

    Args:
        xyz (str): The xyz string to be added header.
        title (str, optional): The title to be added. Defaults to ``''``.

    Returns:
        str: The xyz string with header.
    """
    return f"{len(xyz.strip().splitlines())}\n{title}\n{xyz}"


def mol_from_xyz(
    xyz: str,
    backend: str = "openbabel",
    header: bool = True,
    sanitize: bool = True,
    embed_chiral: bool = False,
    **kwargs,
) -> Chem.RWMol:
    """
    Convert xyz string to RDKit Chem.RWMol.

    Args:
        xyz (str): A XYZ String.
        backend (str): The backend used to perceive molecule. Defaults to ``'openbabel'``.
                       Currently, we only support ``'openbabel'`` and ``'xyz2mol'``.
        header (bool, optional): If lines of the number of atoms and title are included.
                                 Defaults to ``True.``
        sanitize (bool): Sanitize the RDKit molecule during conversion. Helpful to set it to ``False``
                         when reading in TSs. Defaults to ``True``.
        embed_chiral: ``True`` to embed chiral information. Defaults to ``True``.
        supported kwargs:
            xyz2mol:
                - charge: The charge of the species. Defaults to ``0``.
                - allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to ``False``.
                - use_graph: ``True`` to use networkx module for accelerate. Defaults to ``True``.
                - use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to ``False``.
                - forced_rdmc: Defaults to ``False``. In rare case, we may hope to use a tailored
                               version of the Jensen XYZ parser, other than the one available in RDKit.
                               Set this argument to ``True`` to force use RDMC's implementation,
                               which user's may have some flexibility to modify.

    Returns:
        Chem.RWMol: An RDKit molecule object corresponding to the xyz.
    """
    if not header:
        xyz = add_header_to_xyz(xyz, title="")

    if backend.lower() == "openbabel":
        return parse_xyz_by_openbabel(xyz, embed_chiral=embed_chiral, sanitize=sanitize)

    elif backend.lower() == "xyz2mol":
        # Sanitization is done inside the function
        return parse_xyz_by_xyz2mol(xyz, embed_chiral=embed_chiral, **kwargs)

    else:
        raise NotImplementedError(
            f"Backend ({backend}) is not supported. Only `openbabel` and `xyz2mol`"
            f" are supported."
        )


def mol_to_xyz(
    mol: Chem.Mol,
    conf_id: int = -1,
    header: bool = True,
    comment: str = "",
) -> str:
    """
    Convert Chem.Mol to a XYZ string.

    Args:
        mol (RDKitMol): A RDKitMol object.
        conf_id (int, optional): The index of the conformer to be converted. Defaults to ``-1``, exporting the XYZ of the first conformer.
        header (bool, optional): If lines of the number of atoms and title are included. Defaults to ``True``.
        comment (str, optional): The comment to be added. Defaults to ``''``.

    Returns:
        str: A XYZ string.
    """
    xyz = Chem.MolToXYZBlock(mol, confId=conf_id)

    if not header:
        xyz = "\n".join(xyz.splitlines()[2:]) + "\n"
    elif comment:
        xyz = (
            f"{mol.GetNumAtoms()}\n{comment}\n" + "\n".join(xyz.splitlines()[2:]) + "\n"
        )
    return xyz


def xyz_to_coords(
    xyz: str,
    header: bool = False,
) -> np.ndarray:
    """
    Convert xyz string to coordinates in numpy.

    Args:
        xyz (str): A XYZ String.
        header (bool, optional): If lines of the number of atoms and title are included.
                                 Defaults to ``False.``
    Returns:
        np.ndarray: the coordinates
    """
    xyz_lines = xyz.splitlines()[2:] if header else coords.splitlines()
    coords = np.array(
        [[float(atom) for atom in line.strip().split()[1:4]] for line in xyz_lines]
    )
