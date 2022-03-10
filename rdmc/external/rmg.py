#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains RMG related functions. Needs to run with rmg_env.
"""

from itertools import product as set_product
from typing import List, Optional

from rdkit import Chem

try:
    from rmgpy import settings
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.exceptions import ForbiddenStructureException
    import rmgpy.molecule.element as elements
    import rmgpy.molecule.molecule as mm
    from rmgpy.species import Species
except ModuleNotFoundError:
    print('You need to install RMG-Py first!')
    raise


def load_rmg_database(families: list = [],
                      all_families: bool = False):
    """
    A helper function to load RMG database.

    Returns:
        RMGDatabase: A instance of RMG database
    """
    if all_families:
        kinetics_families='all'
    elif families:
        kinetics_families=families
    else:
        kinetics_families='default'
    database_path = settings['database.directory']
    database = RMGDatabase()
    database.load(
        path=database_path,
        thermo_libraries=['primaryThermoLibrary'],
        reaction_libraries=[],
        seed_mechanisms=[],
        kinetics_families=kinetics_families,
    )
    return database


def from_rdkit_mol(rdkitmol,
                   sort: bool = False,
                   raise_atomtype_exception: bool = True,
                   smiles: str = None,
                   ) -> 'Molecule':
    """
    Convert a RDKit Mol object `rdkitmol` to a molecular structure. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    This Kekulizes everything, removing all aromatic atom types.
    """
    is_carbene_or_nitrene = False

    mol = mm.Molecule()
    mol.vertices = []

    # Add hydrogen atoms to complete molecule if needed
    rdkitmol.UpdatePropertyCache(strict=False)
    try:
        Chem.rdmolops.Kekulize(rdkitmol, clearAromaticFlags=True)
    except Exception:  # hope to only catch Boost.Python.ArgumentError. Haven't find a easy way
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

        if (radical_electrons == 2 and element.symbol == "C") or smiles == "[CH2]":
            is_carbene_or_nitrene = True
            atom = mm.Atom(element, 0, charge, '', 1)
        elif (radical_electrons == 2 and element.symbol == "N") or smiles == "[NH]":
            is_carbene_or_nitrene = True
            atom = mm.Atom(element, 0, charge, '', 1)
        else:
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

    if is_carbene_or_nitrene:
        mol.multiplicity = 1
    else:
        mol.multiplicity = mol.get_radical_count() + 1
    # mol.update_atomtypes()

    return mol


def find_reaction_family(database: 'RMGDatabase',
                         reactants: list,
                         products: list,
                         only_families: list = None,
                         unique: bool = True,
                         verbose: bool = True,
                         resonance: bool = True,
                         ) -> Optional[tuple]:
    """
    A helper function to find reaction families for given reactants and products.txt

    Args:
        database (RMGDatabase): A RMG database instance.
        reactants (list): A list of reactant molecules.
        products (list): A list of product molecules.
        only_families (list, optional): A list of family to search from. Defaults to ``None``
                                        for unlimited.
        unique (bool): Whether to only return a single results. Defaults to ``True``.
        verbose (bool, optional): Whether to print results. Defaults to print.
        resonance (bool): Whether to generate resonance strucuture when searching for reaction 
                          family. Defauls to ```True``.

    Returns:
        - tuple: (family_label, is_forward) if ``unique == True``.
        - None: if no match is found.
        - list: [(family_label1, is_foward_1), (family_label2, is_forward2)...] if ``unique == False``.
    """
    # Check if the RMG can find this reaction. Use ``copy``` to avoid changing reactants and products.
    for family in database.kinetics.families.values():
        family.save_order = False

    if resonance:
        reactants = [Species().from_smiles(mol.smiles) for mol in reactants]
        products = [Species().from_smiles(mol.smiles) for mol in products]
    else:
        reactants = [mol.copy() for mol in reactants]
        products = [mol.copy() for mol in products]

    reaction_list = database.kinetics.generate_reactions_from_families(
                                                        reactants=reactants,
                                                        products=products,
                                                        only_families=only_families)
    # Get reaction information
    all_matches = []
    for rxn in reaction_list:
        if verbose:
            print(f'{rxn}\nRMG family: {rxn.family}\nIs forward reaction: {rxn.is_forward}')
        if unique:
            return rxn.family, rxn.is_forward
        else:
            all_matches.append((rxn.family, rxn.is_forward))
    if all_matches:
        return all_matches
    else:
        if verbose:
            print('Doesn\'t match any RMG reaction family!')
        return None, None


def generate_reaction_complex(database: 'RMGDatabase',
                              reactants: list,
                              products: list,
                              only_families: list = None,
                              verbose: bool = True,
                              resonance: bool = True,
                              ) -> List['Molecule']:
    """
    Generate a product complex according to RMG reaction family. Please note,
    that this will only return one template.
    # TODO: provide an option if multiple channel if available.

    Args:
        database (RMGDatabase): An RMG database instance.
        reactants (list): A list of reactant molecules.
        products (list): A list of product molecules.
        only_families (list): A list of families that constrains the search.
        resonance (bool): generate resonance structures when identifying template matching.
                          Can be potentially expensive for some complicated structures.
                          Defaults to ``True``.
        verbose (bool, optional): Whether to print results. Defaults to ``True`` as to print.

    Returns:
        Molecule: a product complex with consistent atom indexing as in the reactant.
    """
    # Find the reaction in the RMG database
    try:
        family_label, forward = find_reaction_family(database,
                                                     reactants,
                                                     products,
                                                     only_families=only_families,
                                                     verbose=verbose,
                                                     resonance=resonance)
    except TypeError:
        # Cannot find any matches
        return None, None
    else:
        if family_label == None:
            return None, None

    # Make the reaction family preserver atom orders
    family = database.kinetics.families[family_label]
    family.save_order = True

    # Get reaction template and species number
    template = family.forward_template if forward else family.reverse_template
    reactant_num = family.reactant_num if forward else family.product_num
    if not reactant_num:
        # Some families doesn't contain reactant num...
        # As a workaround, the following may not always works
        reactant_num = 0
        for reactant in template.reactants:
            try:
                reactant_num += len(reactant.item.split())
            except AttributeError:
                reactant_num += 1

    # The following block copied from rmgpy.data.kinetics.family
    # Its actual role is not clear
    if family.auto_generated and reactant_num != len(reactants):
        raise NotImplementedError

    # Note, for some bimolecular families, it may just have one templates
    if len(template.reactants) < reactant_num:
        try:
            grps = template.reactants[0].item.split()
            template_reactants = [grp for grp in grps]
        except AttributeError:
            template_reactants = [x.item for x in template.reactants]
    else:
        template_reactants = [x.item for x in template.reactants]

    # Resonance structure is generated for both reactants and product
    if resonance:
        r_to_gen = [r.copy(deep=True).generate_resonance_structures(keep_isomorphic=False,
                                                                    filter_structures=True,
                                                                    save_order=True) for r in reactants]
        rs_to_gen = list(set_product(*r_to_gen))
        p_to_match = [p.copy(deep=True).generate_resonance_structures() for p in products]
        ps_to_match = list(set_product(*p_to_match))
    else:
        rs_to_gen = [reactants]
        ps_to_match = [products]

    for reactants in rs_to_gen:
    # A = B or A = B + C
        if len(reactants) == 1:
            # Find all possible mappings
            mappings = family._match_reactant_to_template(reactants[0], template_reactants[0])
            mappings = [[map0] for map0 in mappings]

        # A + B = C + D
        elif len(reactants) == 2:
            # Get A + B mappings
            mappings_a = family._match_reactant_to_template(reactants[0], template_reactants[0])
            mappings_b = family._match_reactant_to_template(reactants[1], template_reactants[1])
            mappings = list(set_product(mappings_a, mappings_b))
            # Get B + A mappings
            mappings_a = family._match_reactant_to_template(reactants[0], template_reactants[1])
            mappings_b = family._match_reactant_to_template(reactants[1], template_reactants[0])
            mappings.extend(list(set_product(mappings_a, mappings_b)))
        else:
            raise NotImplementedError

        # Iterate each found mapping
        for mapping in mappings:
            # Delete old reaction labels
            for struct in reactants:
                struct.clear_labeled_atoms()

            # Apply new reaction labels
            for m in mapping:
                for reactant_atom, template_atom in m.items():
                    reactant_atom.label = template_atom.label

            # Create a reactant complex
            reactant_structure = mm.Molecule()
            if len(reactants) == 1:
                reactant_structure = reactants[0].copy(deep=True)
            else:
                for struct in reactants:
                    reactant_structure = reactant_structure.merge(
                                                struct.copy(deep=True)
                                                )

            # Make a copy for the reactant complex for output
            reactant_complex = reactant_structure.copy(deep=True)
            # Apply reaction recipe
            if forward:
                family.forward_recipe.apply_forward(reactant_structure, unique=True)
            else:
                family.reverse_recipe.apply_forward(reactant_structure, unique=True)
            # Now reactant_structure is tranformed to a product complex
            # First, clean it up
            for atom in reactant_structure.atoms:
                atom.update_charge()
            # Second, Store the product complex in `product_complex` for output
            product_complex = reactant_structure.copy(deep=True)
            # Then, `reactant_structure` will be used to check isomorphism
            # Although it is called reactant structure, but its connectivity
            # has been modified according to the template
            product_structures = reactant_structure.split()

            for struct in product_structures:
                struct.update()  # Clean up each structure
            # Isomorphic check considering resonance structures
            for products in ps_to_match:
                match = check_isomorphic_molecules(product_structures, products)
                if match:
                    return reactant_complex, product_complex
    else:
        raise RuntimeError('Cannot find the correct reaction mapping.')


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

    for mol1 in mols_1:
        for mol2 in mols_2:
            if mol1.is_isomorphic(mol2):
                break
        # an element in mols_1 does not match any element in mols_2
        else:
            return False
    return True
