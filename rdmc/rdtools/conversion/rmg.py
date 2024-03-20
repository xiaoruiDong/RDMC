from rdkit import Chem

from rdmc.rdtools.utils import get_fake_module

try:
    import rmgpy.molecule.element as elements
    import rmgpy.molecule.molecule as mm
except ImportError:
    elements = get_fake_module("elements", "rmgpy")
    mm = get_fake_module("molecule", "rmgpy")


ORDERS = {
    "S": Chem.BondType.SINGLE,
    "D": Chem.BondType.DOUBLE,
    "T": Chem.BondType.TRIPLE,
    "B": Chem.BondType.AROMATIC,
    "Q": Chem.BondType.QUADRUPLE,
}


def mol_from_rmg_mol(
    rmgmol: "rmgpy.molecule.Molecule",
    remove_hs: bool = False,
    sanitize: bool = True,
) -> Chem.RWMol:
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
    atom_id_map = {}

    # only manipulate a copy of ``mol``
    mol_copy = rmgmol.copy(deep=True)
    if not mol_copy.atom_ids_valid():
        mol_copy.assign_atom_ids()
    for i, atom in enumerate(mol_copy.atoms):
        # keeps the original atom order before sorting
        atom_id_map[atom.id] = i
    atoms_copy = mol_copy.vertices

    rwmol = Chem.RWMol()
    reset_num_electron = {}
    for i, rmg_atom in enumerate(atoms_copy):
        rd_atom = Chem.Atom(rmg_atom.element.symbol)
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


def mol_to_rmg_mol(
    rdkitmol,
    sort: bool = False,
    raise_atomtype_exception: bool = True,
) -> "Molecule":
    """
    Convert a RDKit Mol object `rdkitmol` to a molecular structure. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    This Kekulizes everything, removing all aromatic atom types.
    """
    mol = mm.Molecule()
    mol.vertices = []

    # Add hydrogen atoms to complete molecule if needed
    rdkitmol.UpdatePropertyCache(strict=False)
    try:
        Chem.rdmolops.Kekulize(rdkitmol, clearAromaticFlags=True)
    except (
        Exception
    ):  # hope to only catch Boost.Python.ArgumentError. Haven't find a easy way
        Chem.rdmolops.Kekulize(rdkitmol._mol, clearAromaticFlags=True)
    # iterate through atoms in rdkitmol
    for i in range(rdkitmol.GetNumAtoms()):
        rdkitatom = rdkitmol.GetAtomWithIdx(i)

        # Use atomic number as key for element
        number = rdkitatom.GetAtomicNum()
        isotope = rdkitatom.GetIsotope()
        element = elements.get_element(number, isotope or -1)

        # Process charge
        charge = rdkitatom.GetFormalCharge()
        radical_electrons = rdkitatom.GetNumRadicalElectrons()

        atom = mm.Atom(element, radical_electrons, charge, "", 0)
        mol.vertices.append(atom)

        # Add bonds by iterating again through atoms
        for j in range(0, i):
            rdkitbond = rdkitmol.GetBondBetweenAtoms(i, j)
            if rdkitbond is not None:
                order = 0

                # Process bond type
                rdbondtype = rdkitbond.GetBondType()
                if rdbondtype.name == "SINGLE":
                    order = 1
                elif rdbondtype.name == "DOUBLE":
                    order = 2
                elif rdbondtype.name == "TRIPLE":
                    order = 3
                elif rdbondtype.name == "QUADRUPLE":
                    order = 4
                elif rdbondtype.name == "AROMATIC":
                    order = 1.5

                bond = mm.Bond(mol.vertices[i], mol.vertices[j], order)
                mol.add_bond(bond)

    # We need to update lone pairs first because the charge was set by RDKit
    mol.update_lone_pairs()
    # Set atom types and connectivity values
    mol.update(raise_atomtype_exception=raise_atomtype_exception, sort_atoms=sort)

    # Assume this is always true
    # There are cases where 2 radical_electrons is a singlet, but
    # the triplet is often more stable,
    mol.multiplicity = mol.get_radical_count() + 1
    # mol.update_atomtypes()

    return mol
