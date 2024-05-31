from itertools import chain

# import the bond analysis for simplified imports
from rdtools.bond import (
    get_broken_bonds,
    get_formed_bonds,
    get_bonds_with_BO_changed,
    get_formed_and_broken_bonds,
    get_all_changing_bonds,
    get_atoms_in_bonds,
)
from rdtools.mol import (
    get_element_counts,
    get_formal_charge,
    get_spin_multiplicity,
)


def is_num_atoms_balanced(rmol: "Mol", pmol: "Mol") -> bool:
    """
    Check if the number of atoms in reactant and product molecule are the same.

    Args:
        rmol (Mol): The reactant molecule.
        pmol (Mol): The product molecule.

    Returns:
        bool: True if the number of atoms in reactant and product molecule are the same.
    """
    return rmol.GetNumAtoms() == pmol.GetNumAtoms()


def is_element_balanced(rmol: "Mol", pmol: "Mol") -> bool:
    """
    Check if the element counts in reactant and product molecule are the same.

    Args:
        rmol (Mol): The reactant molecule.
        pmol (Mol): The product molecule.

    Returns:
        bool: True if the element counts in reactant and product molecule are the same.
    """
    if is_num_atoms_balanced(rmol, pmol):
        return get_element_counts(rmol) == get_element_counts(pmol)
    return False


def is_charge_balanced(rmol: "Mol", pmol: "Mol") -> bool:
    """
    Check if the charge in reactant and product molecule are the same.

    Args:
        rmol (Mol): The reactant molecule.
        pmol (Mol): The product molecule.

    Returns:
        bool: True if the charge in reactant and product molecule are the same.
    """
    return get_formal_charge(rmol) == get_formal_charge(pmol)


def is_mult_equal(rmol: "Mol", pmol: "Mol") -> bool:
    """
    Check if the spin multiplicity in reactant and product molecule are the same.

    Args:
        rmol (Mol): The reactant molecule.
        pmol (Mol): The product molecule.

    Returns:
        bool: True if the spin multiplicity in reactant and product molecule are the same.
    """
    return get_spin_multiplicity(rmol) == get_spin_multiplicity(pmol)


def get_active_atoms(rmol: "Mol", pmol: "Mol") -> list:
    """
    Get the list of active atoms in the reactant and product molecule. Active atoms are atoms having at least one
    bond breaking or forming.

    Args:
        rmol (Mol): The reactant molecule.
        pmol (Mol): The product molecule.

    Returns:
        list: The list of active atoms in the reactant and product molecule.
    """
    formed, broken = get_formed_and_broken_bonds(rmol, pmol)
    return get_atoms_in_bonds(formed + broken)


def get_involved_atoms(rmol: "Mol", pmol: "Mol") -> list:
    """
    Get the list of involved atoms in the reactant and product molecule. Involved atoms are atoms having at least
    one bond breaking, forming, or with bond order changed.

    Args:
        rmol (Mol): The reactant molecule.
        pmol (Mol): The product molecule.

    Returns:
        list: The list of involved atoms in the reactant and product molecule.
    """
    formed, broken, changed = get_all_changing_bonds(rmol, pmol)
    return get_atoms_in_bonds(formed + broken + changed)