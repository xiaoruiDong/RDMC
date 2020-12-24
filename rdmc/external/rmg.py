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


def renumber_product_atom_by_reaction(database: 'RMGDatabase',
                                      reactants: list,
                                      products: list,
                                      verbose: bool = True,
                                      ) -> List['Molecule']:
    """
    Renumber the product atom indexes according to RMG reaction family.

    Args:
        database (RMGDatabase): An RMG database instance.
        reactants (list): a list of reactant molecules.
        products (list): a list of product molecules.
        verbose (bool, optional): Whether to print results. Defaults to print.

    Returns:
        list of Molecules: Product molecules with correct atom indexes.
    """
    try:
        family_label, forward = find_reaction_family(database,
                                                    reactants,
                                                    products)
    except TypeError:
        return

    # Make the reaction family preserver atom orders
    family = database.kinetics.families[family_label]
    family.save_order = True

    # Get reaction template
    template = family.forward_template if forward else family.reverse_template
    reactant_num = family.reactant_num if forward else family.product_num
    if family.auto_generated and reactant_num != len(reactants):
        raise NotImplementedError

    # 1 <=> 1 reactions
    if len(reactants) == len(products) == 1:
        mappings = family._match_reactant_to_template(reactants[0],
                                                      template.reactants[0].item)
        for mapping in mappings:
            try:
                product_structures = family._generate_product_structures(
                    reactants, [mapping], forward)
            except ForbiddenStructureException:
                pass
            else:
                # isomorphic changes the atom index
                if product_structures[0].copy().is_isomorphic(products[0]):
                    return (product_structures)
    else:
        raise NotImplementedError
