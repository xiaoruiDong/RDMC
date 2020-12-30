#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains RMG related functions. Needs to run with rmg_env.
"""
from typing import List, Optional
from rdkit import Chem

try:
    from rmgpy import settings
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.exceptions import ForbiddenStructureException
    import rmgpy.molecule.element as elements
    import rmgpy.molecule.molecule as mm
except ModuleNotFoundError:
    print('You need to install RMG-Py first!')
    raise


def load_rmg_database():
    """
    A helper function to load RMG database.

    Returns:
        RMGDatabase: A instance of RMG database
    """
    database_path = settings['database.directory']
    database = RMGDatabase()
    database.load(
        path=database_path,
        thermo_libraries=['primaryThermoLibrary'],
        reaction_libraries=[],
        seed_mechanisms=[],
        kinetics_families='default',
    )
    return database


def from_rdkit_mol(rdkitmol,
                   sort: bool = False,
                   raise_atomtype_exception: bool = True,
                   ) -> 'Molecule':
    """
    Convert a RDKit Mol object `rdkitmol` to a molecular structure. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    This Kekulizes everything, removing all aromatic atom types.
    """
    mol = mm.Molecule()
    mol.vertices = []

    # Add hydrogen atoms to complete molecule if needed
    rdkitmol.UpdatePropertyCache(strict=False)
    Chem.rdmolops.Kekulize(rdkitmol, clearAromaticFlags=True)

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

        atom = mm.Atom(element, radical_electrons, charge, '', 0)
        mol.vertices.append(atom)

        # Add bonds by iterating again through atoms
        for j in range(0, i):
            rdkitbond = rdkitmol.GetBondBetweenAtoms(i, j)
            if rdkitbond is not None:
                order = 0

                # Process bond type
                rdbondtype = rdkitbond.GetBondType()
                if rdbondtype.name == 'SINGLE':
                    order = 1
                elif rdbondtype.name == 'DOUBLE':
                    order = 2
                elif rdbondtype.name == 'TRIPLE':
                    order = 3
                elif rdbondtype.name == 'QUADRUPLE':
                    order = 4
                elif rdbondtype.name == 'AROMATIC':
                    order = 1.5

                bond = mm.Bond(mol.vertices[i], mol.vertices[j], order)
                mol.add_bond(bond)

    # We need to update lone pairs first because the charge was set by RDKit
    mol.update_lone_pairs()
    # Set atom types and connectivity values
    mol.update(raise_atomtype_exception=raise_atomtype_exception,
               sort_atoms=sort)

    # Assume this is always true
    # There are cases where 2 radical_electrons is a singlet, but
    # the triplet is often more stable,
    mol.multiplicity = mol.get_radical_count() + 1
    # mol.update_atomtypes()

    return mol

def find_reaction_family(database: 'RMGDatabase',
                         reactants: list,
                         products: list,
                         verbose: bool = True,
                         ) -> Optional[tuple]:
    """
    A helper function to find reaction families for given reactants and products.txt

    Args:
        database (RMGDatabase): A RMG database instance.
        reactants (list): A list of reactant molecules.
        products (list): A list of product molecules.
        verbose (bool, optional): Whether to print results. Defaults to print.

    Returns:
        tuple: (family_label, is_forward). None if no match.
    """
    # Check if the RMG can find this reaction. Use ``copy``` to avoid changing reactants and products.
    for family in database.kinetics.families.values():
        family.save_order = False
    reaction_list = database.kinetics.generate_reactions(reactants=[mol.copy() for mol in reactants],
                                                         products=[mol.copy() for mol in products])
    # Get reaction information
    for rxn in reaction_list:
        family = rxn.family
        forward = rxn.is_forward
        if verbose:
            print(f'{rxn}\nRMG family: {family}\nIs forward reaction: {forward}')
        return family, forward
    else:
        if verbose:
            print('Doesn\'t match any RMG reaction family!')


def generate_product_complex(database: 'RMGDatabase',
                             reactants: list,
                             products: list,
                             verbose: bool = True,
                             ) -> List['Molecule']:
    """
    Generate a product complex according to RMG reaction family. Currently,
    this function is only valid for reaction with a single reactant.

    Args:
        database (RMGDatabase): An RMG database instance.
        reactants (list): a list of reactant molecules.
        products (list): a list of product molecules.
        verbose (bool, optional): Whether to print results. Defaults to print.

    Returns:
        Molecule: a product complex with consistent atom indexing as in the reactant.
    """
    # Find the reaction in the RMG database
    try:
        family_label, forward = find_reaction_family(database,
                                                    reactants,
                                                    products)
    except TypeError:
        # Cannot find any matches
        return

    # Make the reaction family preserver atom orders
    family = database.kinetics.families[family_label]
    family.save_order = True

    # Get reaction template and species number
    template = family.forward_template if forward else family.reverse_template
    reactant_num = family.reactant_num if forward else family.product_num
    product_num = family.product_num if forward else family.reactant_num

    # The following block copied from rmgpy.data.kinetics.family
    # Its actual role is not clear
    if family.auto_generated and reactant_num != len(reactants):
        raise NotImplementedError

    # A <=> B or A <=> B + C reactions
    if len(reactants) == 1:
        # Find all possible mappings
        mappings = family._match_reactant_to_template(reactants[0],
                                                      template.reactants[0].item)
        # Apply each mapping to the reactant
        for mapping in mappings:
            # Clear * labels
            reactants[0].clear_labeled_atoms()
            # Reassign * labels
            for reactant_atom, template_atom in mapping.items():
                reactant_atom.label = template_atom.label
            # Apply reaction recipe
            reactant_structure = reactants[0].copy(deep=True)
            if forward:
                family.forward_recipe.apply_forward(reactant_structure, unique=True)
            else:
                family.reverse_recipe.apply_forward(reactant_structure, unique=True)

            # The product complex is stored in `product_structure`
            # `reactant_structure` will be used to check isomorphism
            product_structure = reactant_structure.copy(deep=True)
            reactant_structures = reactant_structure.split()
            for struct in reactant_structures:
                struct.update()
            match = check_isomorphic_molecules(reactant_structures,
                                               products)
            if match:
                return product_structure
    else:
        raise NotImplementedError


def check_isomorphic_molecules(mols_1: List['Molecule'],
                               mols_2: List['Molecule'],
                               ) -> bool:
    """
    Check if two lists share the same set of molecules.

    Args:
        mols_1 (list): The first set of molecules.
        mols_2 (list): The second set of molecules.

    Returns:
        bool: If both sets share exactly same molecules.
    """
    if len(mols_1) != len(mols_2):
        return False

    match = False
    for mol1 in mols_1:
        for mol2 in mols_2:
            if mol1.is_isomorphic(mol2):
                match = True
                break
        # an element in mols_1 does not match any element in mols_2
        else:
            return False
    return True
