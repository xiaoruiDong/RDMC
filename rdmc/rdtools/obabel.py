import numpy as np

from rdkit import Chem

from rdmc.rdtools.utils import get_fake_module
from rdmc.rdtools.conf import add_conformer
from rdmc.rdtools.bond import BOND_ORDERS

try:
    # Openbabel 3
    from openbabel import openbabel as ob
except ImportError:
    # Openbabel 2
    try:
        import openbabel as ob
    except ImportError:
        ob = get_fake_module("openbabel")


RDKIT_TO_OB_BOND_ORDER = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.QUADRUPLE: 4,
    Chem.BondType.AROMATIC: 5,
}


def get_obmol_coords(obmol: "ob.OBMol"):
    """
    Get the atom coordinates from an openbabel molecule. If all coordinates are zero,
    None will be returned.

    Args:
        obmol (OBMol): The openbabel OBMol to get coordinates from.

    Returns:
        np.array: The coordinates.
    """
    coords = []
    for obatom in ob.OBMolAtomIter(obmol):
        coords.append([obatom.GetX(), obatom.GetY(), obatom.GetZ()])
    return np.array(coords)


def openbabel_mol_to_rdkit_mol(
    obmol: "openbabel.OBMol",
    remove_hs: bool = False,
    sanitize: bool = True,
    embed: bool = True,
) -> Chem.RWMol:
    """
    Convert a OpenBabel molecular structure to a Chem.rdchem.RWMol object.
    Args:
        obmol (Molecule): An OpenBabel Molecule object for the conversion.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, Defaults to False.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to True.
        embed (bool, optional): Whether to embeb 3D conformer from OBMol. Defaults to True.

    Returns:
        RWMol: A writable RDKit RWMol instance.
    """
    # For efficiency consideration, split keep_hs and remove_hs
    if remove_hs:
        return _obmol_to_rdkitmol_removehs(obmol, sanitize, embed)
    else:
        return _obmol_to_rdkitmol_keephs(obmol, sanitize, embed)


def _obmol_to_rdkitmol_keephs(
    obmol: "ob.OBMol",
    sanitize: bool,
    embed: bool,
) -> Chem.RWMol:
    rw_mol = Chem.RWMol()
    for obatom in ob.OBMolAtomIter(obmol):
        atomic_num = obatom.GetAtomicNum()
        atom = Chem.Atom(atomic_num)
        isotope = obatom.GetIsotope()
        if isotope != 0:
            atom.SetIsotope(isotope)
        spin = obatom.GetSpinMultiplicity()
        atom.SetNoImplicit(True)
        if spin == 2:  # radical
            atom.SetNumRadicalElectrons(1)
        elif spin in [1, 3]:  # carbene
            # TODO: Not sure if singlet and triplet are distinguished
            atom.SetNumRadicalElectrons(2)
        atom.SetFormalCharge(obatom.GetFormalCharge())
        rw_mol.AddAtom(atom)

    for bond in ob.OBMolBondIter(obmol):
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        # Get the bond order. For aromatic molecules, the bond order is not
        # 1.5 but 1 or 2. Manually set them to 1.5
        bond_order = bond.GetBondOrder()
        if bond_order not in [1, 2, 3, 4] or bond.IsAromatic():
            bond_order = 12
        # Atom indexes in Openbabel is 1-indexed, so we need to convert them to 0-indexed
        rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, BOND_ORDERS[bond_order])

    if sanitize:
        Chem.SanitizeMol(rw_mol)

    # If OBMol has 3D information, it can be embed to the RDKit Mol
    if embed and (obmol.HasNonZeroCoords() or obmol.NumAtoms() == 1):
        coords = get_obmol_coords(obmol)
        add_conformer(rw_mol, coords=coords)

    return rw_mol


def _obmol_to_rdkitmol_removehs(
    obmol: "ob.OBMol",
    sanitize: bool,
    embed: bool,
) -> Chem.RWMol:
    rw_mol = Chem.RWMol()
    ob_rdkit_atoms = {}
    total_num_atoms = 0
    for i, obatom in enumerate(ob.OBMolAtomIter(obmol)):
        atomic_num = obatom.GetAtomicNum()
        if atomic_num == 1:
            continue
        atom = Chem.Atom(atomic_num)
        isotope = obatom.GetIsotope()
        if isotope != 0:
            atom.SetIsotope(isotope)
        spin = obatom.GetSpinMultiplicity()
        if spin == 2:  # radical
            atom.SetNumRadicalElectrons(1)
        elif spin in [1, 3]:  # carbene
            # TODO: Not sure if singlet and triplet are distinguished
            atom.SetNumRadicalElectrons(2)
        atom.SetFormalCharge(obatom.GetFormalCharge())
        rw_mol.AddAtom(atom)
        ob_rdkit_atoms[i + 1] = total_num_atoms
        total_num_atoms += 1

    for bond in ob.OBMolBondIter(obmol):
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        if (
            ob_rdkit_atoms.get(atom1_idx, -1) < 0
            or ob_rdkit_atoms.get(atom2_idx, -1) < 0
        ):
            continue
        # Get the bond order. For aromatic molecules, the bond order is not
        # 1.5 but 1 or 2. Manually set them to 1.5
        bond_order = bond.GetBondOrder()
        if bond_order not in [1, 2, 3, 4] or bond.IsAromatic():
            bond_order = 12
        # Atom indexes in Openbabel is 1-indexed, so we need to convert them to 0-indexed
        rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, BOND_ORDERS[bond_order])

    # Rectify the molecule
    Chem.RemoveHs(rw_mol, sanitize=sanitize)

    # If OBMol has 3D information, it can be embed to the RDKit Mol
    if embed and (obmol.HasNonZeroCoords() or obmol.NumAtoms() == 1):
        coords = get_obmol_coords(obmol)
        unmask_indices = np.array(sorted(ob_rdkit_atoms.keys())) - 1
        add_conformer(rw_mol, coords=coords[unmask_indices])

    return rw_mol


def rdkit_mol_to_openbabel_mol(
    rdmol: Chem.Mol,
    embed: bool = True,
) -> "ob.OBMol":
    """
    Convert a Mol to an Openbabel mol. This a temporary replace of
    ``rdkit_mol_to_openbabel_mol_manual``. When the molecule has multiple conformers,
    this function will only use the first conformer.

    Args:
        rdmol (Mol): The RDKit Mol object to be converted.
        embed (bool, optional): Whether to embed conformer into the OBMol. Defaults to ``True``.

    Returns:
        OBMol: An openbabel OBMol instance.
    """
    sdf_str = Chem.MolToMolBlock(rdmol)
    obconv, obmol = ob.OBConversion(), ob.OBMol()
    obconv.SetInFormat("sdf")
    obconv.ReadString(obmol, sdf_str)

    # Temporary Fix for spin multiplicity
    _correct_atom_spin_mult(obmol)

    if not embed:
        for atom in ob.OBMolAtomIter(obmol):
            atom.SetVector(ob.vector3(0, 0, 0))

    return obmol


def rdkit_mol_to_openbabel_mol_manual(
    rdmol: Chem.Mol,
    embed: bool = True,
) -> "ob.OBMol":
    """
    Convert a Mol/RWMol to a Openbabel mol. This function has a problem converting
    aromatic molecules. Example: 'c1nc[nH]n1'. Currently use a workaround, converting an
    RDKit Mol to sdf string and read by openbabel.

    Args:
        rdmol (Mol): The RDKit Mol/RWMol object to be converted.
        embed (bool, optional): Whether to embed conformer into the OBMol. Defaults to True.

    Returns:
        OBMol: An openbabel OBMol instance.
    """
    obmol = ob.OBMol()
    for rdatom in rdmol.GetAtoms():
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(rdatom.GetAtomicNum())
        isotope = rdatom.GetIsotope()
        if isotope != 0:
            obatom.SetIsotope(isotope)
        obatom.SetFormalCharge(rdatom.GetFormalCharge())
        obatom.SetImplicitHCount(rdatom.GetTotalNumHs())

    for bond in rdmol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx() + 1
        atom2_idx = bond.GetEndAtomIdx() + 1
        order = RDKIT_TO_OB_BOND_ORDER[bond.GetBondType()]
        obmol.AddBond(atom1_idx, atom2_idx, order)

    # Note: aromatic is not correctly handled for
    # heteroatom involved rings in the current molecule buildup.
    # May need to update in the future

    _correct_atom_spin_mult(obmol)

    if embed:
        try:
            conf = rdmol.GetConformer()
        except ValueError:
            # No conformer
            pass
        else:
            coords = conf.GetPositions()
            set_obmol_coords(obmol, coords)

    return obmol


def set_obmol_coords(obmol: "ob.OBMol", coords: np.array):
    """
    Set the atom coordinates of an openbabel molecule.

    Args:
        obmol (OBMol): The openbabel OBMol to get coordinates from.
        coords (np.array): The coordinates to set.
    """
    for atom_idx, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atom.SetVector(ob.vector3(*coords[atom_idx].tolist()))


def parse_xyz_by_openbabel(xyz: str) -> "ob.OBMol":
    """
    Perceive a xyz str using openbabel and generate the corresponding OBMol.

    Args:
        xyz (str): A str in xyz format containing atom positions.

    Returns:
        ob.OBMol: An openbabel molecule from the xyz
    """
    obconversion = ob.OBConversion()
    obconversion.SetInFormat("xyz")
    obmol = ob.OBMol()
    success = obconversion.ReadString(obmol, xyz)
    if not success:
        raise ValueError("Unable to parse the provided xyz.")

    # Temporary Fix for spin multiplicity
    _correct_atom_spin_mult(obmol)

    return obmol


def _correct_atom_spin_mult(obmol: "ob.OBMol"):
    """
    This is a temporary Fix for Issue # 1. Openbabel has a native function called AssignSpinMultiplicity
    however,  it becomes a dummy function in openbabel 3,
    where it only returns True. See https://github.com/openbabel/
    openbabel/blob/bcb790049bf43415a0b99adf725249c3f4da02bf/src/mol.cpp#L2389-L2395

    The function implements a naive fix for the atom spin multiplicity in openbabel.
    Currently, it cannot deal with any charged species properly.

    Args:
        obmol (OBMol): The openbabel OBMol to be fixed.
    """
    for obatom in ob.OBMolAtomIter(obmol):
        atomic_num = obatom.GetAtomicNum()
        total_valence = obatom.GetTotalValence()
        # Find the unsaturated carbons
        if atomic_num == 6 and total_valence < 4:
            obatom.SetSpinMultiplicity(5 - total_valence)
        # Find the unsaturated nitrogen
        elif atomic_num == 7 and total_valence < 3:
            obatom.SetSpinMultiplicity(4 - total_valence)
        # Find the unsaturated oxygen
        elif atomic_num == 8 and total_valence < 2:
            obatom.SetSpinMultiplicity(3 - total_valence)
        # Find the unsaturated nitrogen and halogen
        elif atomic_num in [1, 9, 17, 35, 53] and total_valence == 0:
            obatom.SetSpinMultiplicity(2)
