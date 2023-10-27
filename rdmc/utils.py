#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides methods that can directly apply to RDKit Mol/RWMol.
"""

from typing import Iterable, Union

import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, Mol, RWMol

# Since 2022.09.1, RDKit added built-in XYZ parser using xyz2mol approach
try:
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    rdDetermineBonds = None
    # TODO: Could raise an warning says RDKit built-in xyz2mol not supported due to lower version
    # uses a rdmc-implemented version of xyz2mol
from rdmc.external.xyz2mol import parse_xyz_by_jensen as parse_xyz_by_jensen_rdmc
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists

# Mute RDKit's error logs
# They can be confusing at places where try ... except ... are implemented.
RDLogger.DisableLog("rdApp.*")

try:
    # Openbabel 3
    from openbabel import openbabel as ob
except ImportError:
    # Openbabel 2
    import openbabel as ob

# Bond order dictionary for RDKit, numbers are the bond order.
ORDERS = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    1.5: BondType.AROMATIC,
    4: BondType.QUADRUPLE,
    "S": BondType.SINGLE,
    "D": BondType.DOUBLE,
    "T": BondType.TRIPLE,
    "B": BondType.AROMATIC,
    "Q": BondType.QUADRUPLE,
}

# The rotational bond definition in RDkit
# It is the same as rdkit.Chem.Lipinski import RotatableBondSmarts
ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
ROTATABLE_BOND_SMARTS_WO_METHYL = Chem.MolFromSmarts(
    "[!$(*#*)&!D1!H3]-&!@[!$(*#*)&!D1&!H3]"
)

# When perceiving molecules, openbabel will always perceive carbon monoxide as [C]=O
# Needs to correct it by [C-]#[O+]
CO_OPENBABEL_PATTERN = ob.OBSmartsPattern()
CO_OPENBABEL_PATTERN.Init("[Cv2X1]=[OX1]")

# Carbene, nitrene, and atomic oxygen templates. RDKit and Openbabel have difficulty
# distinguish their multiplicity when input as SMILES or XYZ
CARBENE_PATTERN = Chem.MolFromSmarts("[Cv0,Cv1,Cv2,Nv0,Nv1,Ov0]")

PERIODIC_TABLE = Chem.GetPeriodicTable()
VDW_RADII = {i: PERIODIC_TABLE.GetRvdw(i) for i in range(1, 36)}


def determine_smallest_atom_index_in_torsion(
    atom1: "rdkit.Chem.rdchem.Atom",
    atom2: "rdkit.Chem.rdchem.Atom",
) -> int:
    """
    Determine the smallest atom index in mol connected to ``atom1`` which is not ``atom2``.
    Returns a heavy atom if available, otherwise a hydrogen atom.
    Useful for deterministically determining the indices of four atom in a torsion.
    This function assumes there ARE additional atoms connected to ``atom1``, and that ``atom2`` is not a hydrogen atom.

    Args:
        atom1 (Atom): The atom who's neighbors will be searched.
        atom2 (Atom): An atom connected to ``atom1`` to exclude (a pivotal atom).

    Returns:
        int: The smallest atom index (1-indexed) connected to ``atom1`` which is not ``atom2``.
    """
    neighbor = [a for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()]
    atomic_num_list = sorted([nb.GetAtomicNum() for nb in neighbor])
    min_atomic, max_atomic = atomic_num_list[0], atomic_num_list[-1]
    if min_atomic == max_atomic or min_atomic > 1:
        return min([nb.GetIdx() for nb in neighbor])
    else:
        return min([nb.GetIdx() for nb in neighbor if nb.GetAtomicNum() != 1])


def find_internal_torsions(
    mol: Union["Mol", "RWMol"],
    exclude_methyl: bool = False,
) -> list:
    """
    Find the internal torsions from RDkit molecule.

    Args:
        mol (Union[Mol, RWMol]): RDKit molecule.
        exclude_methyl (bool): Whether exclude the torsions with methyl groups.

    Returns:
        list: A list of internal torsions.
    """
    torsions = list()
    smarts = (
        ROTATABLE_BOND_SMARTS if not exclude_methyl else ROTATABLE_BOND_SMARTS_WO_METHYL
    )
    rot_atom_pairs = mol.GetSubstructMatches(smarts)

    for atoms_ind in rot_atom_pairs:
        pivots = [mol.GetAtomWithIdx(i) for i in atoms_ind]
        first_atom_ind = determine_smallest_atom_index_in_torsion(*pivots)
        pivots.reverse()
        last_atom_ind = determine_smallest_atom_index_in_torsion(*pivots)
        torsions.append([first_atom_ind, *atoms_ind, last_atom_ind])
    return torsions


def find_ring_torsions(mol: Union["Mol", "RWMol"]) -> list:
    """
    Find the ring from RDkit molecule.

    Args:
        mol (Union[Mol, RWMol]): RDKit molecule.

    Returns:
        list: A list of ring torsions.
    """
    try:
        _, ring_torsions = CalculateTorsionLists(mol.ToRWMol())
    except AttributeError:
        _, ring_torsions = CalculateTorsionLists(mol)
    if ring_torsions:
        ring_torsions = [list(t) for t in ring_torsions[0][0]]
    return ring_torsions


def openbabel_mol_to_rdkit_mol(
    obmol: "openbabel.OBMol",
    remove_hs: bool = False,
    sanitize: bool = True,
    embed: bool = True,
) -> "RWMol":
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
    rw_mol = Chem.rdchem.RWMol()
    for obatom in ob.OBMolAtomIter(obmol):
        atom = Chem.rdchem.Atom(obatom.GetAtomicNum())
        isotope = obatom.GetIsotope()
        if isotope != 0:
            atom.SetIsotope(isotope)
        spin = obatom.GetSpinMultiplicity()
        if not remove_hs:
            atom.SetNoImplicit(True)
        if spin == 2:  # radical
            atom.SetNumRadicalElectrons(1)
        elif spin in [1, 3]:  # carbene
            # TODO: Not sure if singlet and triplet are distinguished
            atom.SetNumRadicalElectrons(2)
        atom.SetFormalCharge(obatom.GetFormalCharge())
        if not (remove_hs and obatom.GetAtomicNum == 1):
            rw_mol.AddAtom(atom)

    for bond in ob.OBMolBondIter(obmol):
        # Atom indexes in Openbabel is 1-indexed, so we need to convert them to 0-indexed
        atom1_idx = bond.GetBeginAtomIdx() - 1
        atom2_idx = bond.GetEndAtomIdx() - 1
        # Get the bond order. For aromatic molecules, the bond order is not
        # 1.5 but 1 or 2. Manually set them to 1.5
        bond_order = bond.GetBondOrder()
        if bond_order not in [1, 2, 3, 4] or bond.IsAromatic():
            bond_order = 1.5

        rw_mol.AddBond(atom1_idx, atom2_idx, ORDERS[bond_order])

    # Rectify the molecule
    if remove_hs:
        rw_mol = Chem.RemoveHs(rw_mol, sanitize=sanitize)
    elif sanitize:
        Chem.SanitizeMol(rw_mol)

    # If OBMol has 3D information, it can be embed to the RDKit Mol
    if embed and (obmol.HasNonZeroCoords() or obmol.NumAtoms() == 1):
        coords = get_obmol_coords(obmol)
        conf = Chem.rdchem.Conformer(
            coords.shape[0]
        )  # Create a conformer that has number of atoms specified
        set_rdconf_coordinates(conf, coords)
        rw_mol.AddConformer(conf, assignId=True)
    return rw_mol


def rdkit_mol_to_openbabel_mol(
    rdmol: Union["Mol", "RWMol"],
    embed: bool = True,
) -> "openbabel.OBMol":
    """
    Convert a Mol/RWMol to a Openbabel mol. This a temporary replace of
    ``rdkit_mol_to_openbabel_mol_manual``.

    Args:
        rdmol (Mol): The RDKit Mol/RWMol object to be converted.
        embed (bool, optional): Whether to embed conformer into the OBMol. Defaults to True.

    Returns:
        OBMol: An openbabel OBMol instance.
    """
    try:
        # RDKitMol
        sdf_str = rdmol.ToMolBlock()
    except AttributeError:
        # RDKit Mol or RWMol
        sdf_str = Chem.MolToMolBlock(rdmol)
    obconv = ob.OBConversion()
    obconv.SetInFormat("sdf")
    obmol = ob.OBMol()
    obconv.ReadString(obmol, sdf_str)

    # Temporary Fix for Issue # 1
    # This function works okay with openbabel 2.4.1 but not 3.1.1
    # The atom spin multiplicity looks not right in the obmol
    # A naive fix for carbons and oxygens
    # This fix cannot deal with any charged species!!!
    for obatom in ob.OBMolAtomIter(obmol):
        # Find the unsaturated carbons
        if obatom.GetAtomicNum() == 6 and obatom.GetTotalValence() < 4:
            obatom.SetSpinMultiplicity(5 - obatom.GetTotalValence())
        elif obatom.GetAtomicNum() == 7 and obatom.GetTotalValence() < 3:
            obatom.SetSpinMultiplicity(4 - obatom.GetTotalValence())
        elif obatom.GetAtomicNum() == 8 and obatom.GetTotalValence() < 2:
            obatom.SetSpinMultiplicity(3 - obatom.GetTotalValence())
        elif obatom.GetAtomicNum() == 1 and obatom.GetTotalValence() == 0:
            obatom.SetSpinMultiplicity(2)

    if not embed:
        for atom in ob.OBMolAtomIter(obmol):
            atom.SetVector(ob.vector3(0, 0, 0))

    return obmol


def rdkit_mol_to_openbabel_mol_manual(
    rdmol: Union["Mol", "RWMol"],
    embed: bool = True,
) -> "openbabel.OBMol":
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
    bond_type_dict = {
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.QUADRUPLE: 4,
        BondType.AROMATIC: 5,
    }
    for bond in rdmol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx() + 1
        atom2_idx = bond.GetEndAtomIdx() + 1
        order = bond_type_dict[bond.GetBondType()]
        obmol.AddBond(atom1_idx, atom2_idx, order)

    # Note: aromatic is not correctly handeled for
    # heteroatom involved rings in the current molecule buildup.
    # May need to update in the future

    obmol.AssignSpinMultiplicity(True)

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


def rmg_mol_to_rdkit_mol(
    rmgmol: "rmgpy.molecule.Molecule",
    remove_hs: bool = False,
    sanitize: bool = True,
) -> "RWMol":
    """
    Convert a RMG molecular structure to an RDKit Mol object. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    Perceives aromaticity.
    Adopted from rmgpy/molecule/converter.py

    Args:
        rmgmol (Molecule): An RMG Molecule object for the conversion.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

    Returns:
        RWMol: An RWMol molecule object corresponding to the input RMG Molecule object.
    """
    atom_id_map = dict()

    # only manipulate a copy of ``mol``
    mol_copy = rmgmol.copy(deep=True)
    if not mol_copy.atom_ids_valid():
        mol_copy.assign_atom_ids()
    for i, atom in enumerate(mol_copy.atoms):
        # keeps the original atom order before sorting
        atom_id_map[atom.id] = i
    atoms_copy = mol_copy.vertices

    rwmol = Chem.rdchem.RWMol()
    reset_num_electron = {}
    for i, rmg_atom in enumerate(atoms_copy):
        rd_atom = Chem.rdchem.Atom(rmg_atom.element.symbol)
        if rmg_atom.element.isotope != -1:
            rd_atom.SetIsotope(rmg_atom.element.isotope)
        if not remove_hs:
            # Avoid `SanitizeMol` adding undesired hydrogens
            rd_atom.SetNoImplicit(True)
        else:
            explicit_Hs = [
                True
                for a, b in rmg_atom.edges.items()
                if a.is_hydrogen() and b.is_single()
            ]
            rd_atom.SetNumExplicitHs(sum(explicit_Hs))
            rd_atom.SetNoImplicit(True)
        rd_atom.SetNumRadicalElectrons(rmg_atom.radical_electrons)
        rd_atom.SetFormalCharge(rmg_atom.charge)

        # There are cases requiring to reset electrons after sanitization
        # for carbene, nitrene and atomic oxygen
        # For other atoms, to be added once encountered
        if rmg_atom.is_carbon() and rmg_atom.lone_pairs >= 1 and not rmg_atom.charge:
            reset_num_electron[i] = rmg_atom.radical_electrons
        elif (
            rmg_atom.is_nitrogen() and rmg_atom.lone_pairs >= 2 and not rmg_atom.charge
        ):
            reset_num_electron[i] = rmg_atom.radical_electrons
        elif rmg_atom.is_oxygen and rmg_atom.lone_pairs >= 3 and not rmg_atom.charge:
            reset_num_electron[i] = rmg_atom.radical_electrons
        if not (remove_hs and rmg_atom.symbol == "H"):
            rwmol.AddAtom(rd_atom)

    # Add the bonds
    for atom1 in atoms_copy:
        if remove_hs and atom1.is_hydrogen():
            continue
        for atom2, bond12 in atom1.edges.items():
            if remove_hs and atom2.is_hydrogen():
                continue
            if bond12.is_hydrogen_bond():
                continue
            if atoms_copy.index(atom1) < atoms_copy.index(atom2):
                rwmol.AddBond(
                    atom_id_map[atom1.id],
                    atom_id_map[atom2.id],
                    ORDERS[bond12.get_order_str()],
                )

    # Rectify the molecule
    if remove_hs:
        rwmol = Chem.RemoveHs(rwmol, sanitize=sanitize)
    elif sanitize:
        Chem.SanitizeMol(rwmol)

    for key, val in reset_num_electron.items():
        rwmol.GetAtomWithIdx(key).SetNumRadicalElectrons(val)

    return rwmol


def set_rdconf_coordinates(
    conf: Union["Conformer", "RDKitConf"], coords: Union[tuple, list, np.ndarray]
):
    """
    Set the Positions of atoms of the conformer.

    Args:
        conf (Union[Conformer, 'RDKitConf']): The conformer to be set.
        coords (Union[tuple, list, np.ndarray]): The coordinates to be set.

    Raises:
        ValueError: Not a valid ``coords`` input, when giving something else.
    """
    try:
        num_atoms = coords.shape[0]
    except AttributeError:
        coords = np.array(coords)
        num_atoms = coords.shape[0]
    finally:
        for i in range(num_atoms):
            conf.SetAtomPosition(i, coords[i, :])


def get_obmol_coords(obmol: ob.OBMol):
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


def set_obmol_coords(obmol: ob.OBMol, coords: np.array):
    """
    Get the atom coordinates from an openbabel molecule. If all coordinates are zero,
    It will return None

    Args:
        obmol (OBMol): The openbabel OBMol to get coordinates from.
        coords (np.array): The coordinates to set.
    """
    for atom_idx, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atom.SetVector(ob.vector3(*coords[atom_idx].tolist()))


def fix_CO_openbabel(obmol: "Openbabel.OBMol", correct_CO: bool = True):
    """
    Fix the CO perception issue for openbabel molecule.

    Args:
        obmol (Openbabel.OBMol): The Openbabel molecule instance.
        correct_CO (bool, optional): Whether to fix this issue. Defaults to True.
    """
    if not correct_CO:
        return
    CO_OPENBABEL_PATTERN.Match(obmol)
    for pair in CO_OPENBABEL_PATTERN.GetUMapList():
        obmol.GetBond(*pair).SetBondOrder(3)
        for idx in pair:
            atom = obmol.GetAtom(idx)
            if atom.GetAtomicNum() == 6:
                atom.SetSpinMultiplicity(0)
                atom.SetFormalCharge(-1)
            elif atom.GetAtomicNum() == 8:
                atom.SetSpinMultiplicity(0)
                atom.SetFormalCharge(+1)


def parse_xyz_by_openbabel(xyz: str, correct_CO: bool = True):
    """
    Perceive a xyz str using openbabel and generate the corresponding OBMol.

    Args:
        xyz (str): A str in xyz format containing atom positions.
        correctCO (bool, optional): It is known that openbabel will parse carbon monoxide
                                    as [C]=O instead of [C-]#[O+]. This function contains
                                    a patch to correct that. Defaults to ``True``.

    Returns:
        ob.OBMol: An openbabel molecule from the xyz
    """
    obconversion = ob.OBConversion()
    obconversion.SetInFormat("xyz")
    obmol = ob.OBMol()
    success = obconversion.ReadString(obmol, xyz)
    if not success:
        raise ValueError("Unable to parse the provided xyz.")

    # Temporary Fix for Issue # 1
    # This function works okay with openbabel 2.4.1 but not 3.1.1
    # The atom spin multiplicity looks not right in the obmol
    # A naive fix for carbons and oxygens
    # This fix cannot deal with any charged species!!!
    for obatom in ob.OBMolAtomIter(obmol):
        # Find the unsaturated carbons
        if obatom.GetAtomicNum() == 6 and obatom.GetTotalValence() < 4:
            obatom.SetSpinMultiplicity(5 - obatom.GetTotalValence())
        # Find the unsaturated nitrogen
        elif obatom.GetAtomicNum() == 7 and obatom.GetTotalValence() < 3:
            obatom.SetSpinMultiplicity(4 - obatom.GetTotalValence())
        # Find the unsaturated oxygen
        elif obatom.GetAtomicNum() == 8 and obatom.GetTotalValence() < 2:
            obatom.SetSpinMultiplicity(3 - obatom.GetTotalValence())
        # Find the unsaturated nitrogen and halogen
        elif (
            obatom.GetAtomicNum() in [1, 9, 17, 35, 53]
            and obatom.GetTotalValence() == 0
        ):
            obatom.SetSpinMultiplicity(2)

    # Correct [C]=O to [C-]#[O+]
    fix_CO_openbabel(obmol, correct_CO=correct_CO)

    return obmol


def get_element_symbols(atom_nums: Iterable):
    """
    Get the element symbols for a given atom index list.

    Args:
        atom_nums (Iterable): A list of elemental numbers.
    Returns:
        list: A list of element symbols.
    """
    return [PERIODIC_TABLE.GetElementSymbol(int(atom_num)) for atom_num in atom_nums]


def get_atom_masses(atom_nums: Iterable):
    """
    Get the atom masses for a given atom index list.

    Args:
        atom_nums (Iterable): A list of elemental numbers.
    Returns:
        list: A list of element symbols.
    """
    return [PERIODIC_TABLE.GetAtomicWeight(int(atom_num)) for atom_num in atom_nums]


def get_internal_coords(
    obmol,
    nonredundant: bool = True,
) -> list:
    """
    Generate a non_redundant_internal coordinate.

    Args:
        obmol (OBMol): Openbabel mol.
        nonredundant (bool): whether non-redundant. Defaults to ``True``.
    """
    obconv = ob.OBConversion()
    obconv.SetOutFormat("gzmat")
    gzmat_str = obconv.WriteString(obmol)
    lines = gzmat_str.split("Variables:")[0].splitlines()[6:]
    bonds = []
    angles = []
    torsions = []
    for idx, line in enumerate(lines):
        items = line.split()
        try:
            bonds.append((idx + 1, int(items[1])))
            angles.append([idx + 1, int(items[1]), int(items[3])])
            torsions.append([idx + 1, int(items[1]), int(items[3]), int(items[5])])
        except IndexError:
            # First, second, and third lines are special
            pass
    if nonredundant:
        non_red_torsions = []
        pivots = []
        for tor in torsions:
            if tor[1:3] not in pivots and tor[-2:-4:-1] not in pivots:
                pivots.append(tor[1:3])
                non_red_torsions.append(tor)
        pass
        torsions = non_red_torsions
    return bonds, angles, torsions


def reverse_map(map: Iterable, as_list: bool = True):
    """
    Inverse-transform the index and value relationship in a mapping.
    E.g., when doing a subgraph match, RDKit will returns a list
    that the indexes correspond to the reference molecule and the values
    correspond to the probing molecule. One by renumber the atoms in the
    probing molecule according to the reverse_map, the atom numbering between
    the two molecules should be consistent

    Args:
        map (Iterable): An atom mapping.
        as_list (bool, optional): Output result as a `list` object. Otherwise,
                                  the output is a np.ndarray.

    Returns:
        An inverted atom map from the given ``match`` atom map
    """
    if as_list:
        return np.argsort(map).tolist()
    else:
        return np.argsort(map)


def parse_xyz_by_jensen(
    xyz: str,
    charge: int = 0,
    allow_charged_fragments: bool = False,
    use_huckel: bool = False,
    embed_chiral: bool = True,
    correct_CO: bool = True,
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
        correctCO (bool, optional): Defaults to ``True``.
                                    In order to get correct RDKit molecule for carbon monoxide
                                    ([C-]#[O+]), allow_charged_fragments should be forced to ``True``.
                                    This function contains a patch to correct that.
        use_atom_maps(bool, optional): ``True`` to set atom map numbers to the molecule. Defaults to ``False``.
        force_rdmc (bool, optional): Defaults to ``False``. In rare case, we may hope to use a tailored
                                     version of the Jensen XYZ parser, other than the one available in RDKit.
                                     Set this argument to ``True`` to force use RDMC's implementation,
                                     which user's may have some flexibility to modify.

    Returns:
        Mol: A RDKit Mol corresponding to the xyz.
    """
    # Version < 2022.09.1
    if rdDetermineBonds is None or force_rdmc:
        return parse_xyz_by_jensen_rdmc(
            xyz=xyz,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
            use_graph=True,
            use_huckel=use_huckel,
            embed_chiral=embed_chiral,
            use_atom_maps=use_atom_maps,
            correct_CO=correct_CO,
        )

    # Version >= 2022.09.1
    try:
        mol = Chem.Mol(Chem.MolFromXYZBlock(xyz))
    except BaseException:
        raise ValueError("Unable to parse the provided xyz.")
    else:
        if mol is None:
            raise ValueError("Unable to parse the provided xyz.")
    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        # No implicit Hs for single atom molecule
        atom.SetNoImplicit(True)
        # Get the valence of the atom
        valence = PERIODIC_TABLE.GetDefaultValence(atom.GetAtomicNum())
        # Set the charge of the atom
        atom.SetFormalCharge(charge)
        # Set the num radical electrons
        atom.SetNumRadicalElectrons(valence - atom.GetFormalCharge())
        return mol
    rdDetermineBonds.DetermineConnectivity(
        mol,
        useHueckel=use_huckel,
        charge=charge,
    )
    # A force correction for CO
    if (
        correct_CO
        and mol.GetNumAtoms() == 2
        and {atom.GetAtomicNum() for atom in mol.GetAtoms()} == {6, 8}
    ):
        allow_charged_fragments = True
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=charge,
        allowChargedFragments=allow_charged_fragments,
        embedChiral=embed_chiral,
        useAtomMap=use_atom_maps,
    )
    return mol


def get_closed_shell_cheap(mol: "RWMol") -> "RWMol":
    """
    Get the closed shell molecule of a radical molecule. This is a cheap version
    where no new atom is actually added to the molecule and all operation is inplace.

    Args:
        mol (RWMol): The radical molecule.

    Returns:
        RWMol: The closed shell molecule.
    """
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(False)
    return mol


def get_closed_shell_by_add_hs(
    mol: "RWMol",
) -> "RWMol":
    """
    Get the closed shell molecule of a radical molecule by explicitly adding
    hydrogen atoms to the molecule.
    """
    atom_idx = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        num_rad_elecs = atom.GetNumRadicalElectrons()
        if num_rad_elecs:
            for _ in range(num_rad_elecs):
                mol.AddAtom(Chem.rdchem.Atom(1))
                mol.AddBond(atom_idx, atom.GetIdx(), Chem.rdchem.BondType.SINGLE)
                atom_idx += 1
            atom.SetNumRadicalElectrons(0)
    return mol


# CPK (Corey-Pauling-Koltun) color scheme, Generated using ChatGPT
CPK_COLOR_PALETTE = {
    "H": (1.00, 1.00, 1.00),
    "He": (0.85, 1.00, 1.00),
    "Li": (0.80, 0.50, 1.00),
    "Be": (0.76, 1.00, 0.00),
    "B": (1.00, 0.71, 0.71),
    "C": (0.56, 0.56, 0.56),
    "N": (0.19, 0.31, 0.97),
    "O": (1.00, 0.05, 0.05),
    "F": (0.56, 0.88, 0.31),
    "Ne": (0.70, 0.89, 0.96),
    "Na": (0.67, 0.36, 0.95),
    "Mg": (0.54, 1.00, 0.00),
    "Al": (0.75, 0.65, 0.65),
    "Si": (0.94, 0.78, 0.63),
    "P": (1.00, 0.50, 0.00),
    "S": (1.00, 1.00, 0.19),
    "Cl": (0.12, 0.94, 0.12),
    "Ar": (0.50, 0.82, 0.89),
    "K": (0.56, 0.25, 0.83),
    "Ca": (0.24, 1.00, 0.00),
    "Sc": (0.90, 0.90, 0.90),
    "Ti": (0.75, 0.76, 0.78),
    "V": (0.65, 0.65, 0.67),
    "Cr": (0.54, 0.60, 0.78),
    "Mn": (0.61, 0.47, 0.78),
    "Fe": (0.87, 0.39, 0.29),
    "Co": (0.94, 0.56, 0.63),
    "Ni": (0.31, 0.82, 0.31),
    "Cu": (0.78, 0.50, 0.20),
    "Zn": (0.49, 0.50, 0.69),
    "Ga": (0.76, 0.56, 0.56),
    "Ge": (0.40, 0.56, 0.56),
    "As": (0.74, 0.50, 0.89),
    "Se": (1.00, 0.63, 0.00),
    "Br": (0.65, 0.16, 0.16),
    "Kr": (0.36, 0.72, 0.82),
    "Rb": (0.44, 0.18, 0.69),
    "Sr": (0.00, 1.00, 0.00),
    "Y": (0.58, 1.00, 1.00),
    "Zr": (0.58, 0.88, 0.88),
    "Nb": (0.45, 0.76, 0.79),
    "Mo": (0.33, 0.71, 0.71),
    "Tc": (0.23, 0.62, 0.62),
    "Ru": (0.14, 0.56, 0.56),
    "Rh": (0.04, 0.49, 0.55),
    "Pd": (0.00, 0.41, 0.52),
    "Ag": (0.75, 0.75, 0.75),
    "Cd": (1.00, 0.85, 0.56),
    "In": (0.65, 0.46, 0.45),
    "Sn": (0.40, 0.50, 0.50),
    "Sb": (0.62, 0.39, 0.71),
    "Te": (0.83, 0.48, 0.00),
    "I": (0.58, 0.00, 0.58),
    "Xe": (0.26, 0.62, 0.69),
    "Cs": (0.34, 0.09, 0.56),
    "Ba": (0.00, 0.79, 0.00),
    "La": (0.44, 0.83, 1.00),
    "Ce": (1.00, 1.00, 0.78),
    "Pr": (0.85, 1.00, 0.78),
    "Nd": (0.78, 1.00, 0.78),
    "Pm": (0.64, 1.00, 0.78),
    "Sm": (0.56, 1.00, 0.78),
    "Eu": (0.38, 1.00, 0.78),
    "Gd": (0.27, 1.00, 0.78),
    "Tb": (0.19, 1.00, 0.78),
    "Dy": (0.12, 1.00, 0.78),
    "Ho": (0.00, 1.00, 0.61),
    "Er": (0.00, 0.90, 0.46),
    "Tm": (0.00, 0.83, 0.32),
    "Yb": (0.00, 0.75, 0.22),
    "Lu": (0.00, 0.67, 0.14),
    "Hf": (0.30, 0.76, 1.00),
    "Ta": (0.30, 0.65, 1.00),
    "W": (0.13, 0.58, 0.84),
    "Re": (0.15, 0.49, 0.55),
    "Os": (0.15, 0.40, 0.49),
    "Ir": (0.09, 0.33, 0.34),
    "Pt": (0.82, 0.82, 0.88),
    "Au": (1.00, 0.82, 0.14),
    "Hg": (0.72, 0.72, 0.82),
    "Tl": (0.65, 0.33, 0.30),
    "Pb": (0.34, 0.35, 0.38),
    "Bi": (0.62, 0.31, 0.71),
    "Th": (0.00, 0.73, 1.00),
    "Pa": (0.00, 0.63, 1.00),
    "U": (0.00, 0.56, 1.00),
    "Np": (0.00, 0.50, 1.00),
    "Pu": (0.00, 0.42, 1.00),
    "Am": (0.33, 0.36, 0.95),
    "Cm": (0.47, 0.36, 0.89),
    "Bk": (0.54, 0.31, 0.89),
    "Cf": (0.63, 0.21, 0.83),
    "Es": (0.70, 0.12, 0.83),
    "Fm": (0.70, 0.12, 0.73),
    "Md": (0.70, 0.05, 0.65),
    "No": (0.74, 0.05, 0.53),
    "Lr": (0.78, 0.00, 0.40),
    "Rf": (0.80, 0.00, 0.35),
    "Db": (0.82, 0.00, 0.31),
    "Sg": (0.85, 0.00, 0.27),
    "Bh": (0.88, 0.00, 0.22),
    "Hs": (0.90, 0.00, 0.18),
    "Mt": (0.92, 0.00, 0.15),
    "Ds": (0.94, 0.00, 0.12),
    "Rg": (0.96, 0.00, 0.09),
    "Cn": (0.98, 0.00, 0.06),
    "Nh": (1.00, 0.00, 0.02),
    "Fl": (1.00, 0.00, 0.00),
    "Mc": (1.00, 0.02, 0.00),
    "Lv": (1.00, 0.06, 0.00),
    "Ts": (1.00, 0.10, 0.00),
    "Og": (1.00, 0.16, 0.00),
}
