from functools import lru_cache
from itertools import chain
import logging

logger = logging.getLogger(__name__)


from rdkit import Chem

from rdmc.rdtools.atom import decrement_radical
from rdmc.rdtools.bond import increment_order, BOND_ORDERS
from rdmc.rdtools.mol import get_spin_multiplicity


def _check_viability(
    cur_mult: int,
    target_mult: int,
) -> bool:
    """
    Check whether the molecule is viable to be fixed by saturating methods.
    Saturating methods works by removing 2 radicals to form a bond / lone pair.
    Therefore, the difference between the current multiplicity and the target multiplicity
    should be even.

    Args:
        cur_mult (int): The current multiplicity of the molecule.
        target_mult (int): The target multiplicity of the molecule.

    Returns:
        bool: ``True`` if the molecule is viable, ``False`` otherwise.
    """
    if cur_mult <= target_mult:
        return False
    return (cur_mult - target_mult) % 2 == 0


@lru_cache(maxsize=1)
def get_biradical_12_query() -> Chem.Mol:
    """
    Get a template molecule for 1,2 biradical. The template is two atoms each with >= 1 radical electrons.
    """
    query = Chem.MolFromSmarts("*~*")
    radical_query = Chem.rdqueries.NumRadicalElectronsGreaterQueryAtom(0)
    for atom in query.GetAtoms():
        atom.ExpandQuery(radical_query)
    return query


def saturate_biradical_12(
    mol,
    multiplicity: int,
):
    """
    A method help to saturate 1,2 biradicals to match the given
    molecule spin multiplicity. E.g.::

        *C - C* => C = C

    In the current implementation, no error will be raised,
    if the function doesn't achieve the goal. This function has not been
    been tested on nitrogenate.

    Args:
        multiplicity (int): The target multiplicity.
        verbose (bool): Whether to print additional information. Defaults to ``True``.
    """
    cur_mult = get_spin_multiplicity(mol)
    if not _check_viability(cur_mult, multiplicity):
        return

    # Find all radical sites and save in `rad_atoms` and `rad_atom_elec_nums`
    target_bonds = list(mol.GetSubstructMatches(get_biradical_12_query()))
    if not target_bonds:
        return

    # bookkeep the number of double bonds to be added and the radical sites
    num_dbs = (cur_mult - multiplicity) // 2
    rad_atom_pool = set(chain(*target_bonds))

    while num_dbs and target_bonds:
        for i, j in target_bonds:
            bond = mol.GetBondBetweenAtoms(i, j)
            try:
                increment_order(bond)
            except KeyError:  # Cannot get the bond with bond order + 1
                continue
            for atom in [mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)]:
                decrement_radical(atom)
                if atom.GetNumRadicalElectrons() == 0:
                    rad_atom_pool.remove(atom.GetIdx())

            num_dbs -= 1
            break
        else:  # no update is made at all, stop the outer loop
            break
        # Whenever a bond is added, the available radical sites may change
        # Therefore, we need to update the target bonds
        target_bonds = [
            pair
            for pair in target_bonds
            if pair[0] in rad_atom_pool and pair[1] in rad_atom_pool
        ]

    mol.UpdatePropertyCache(strict=False)

    if num_dbs:
        logger.debug(
            f"Target spin multiplicity {multiplicity} cannot be fulfilled by"
            f" saturating 1,2 biradicals for {mol}"
        )


@lru_cache(maxsize=1)
def get_carbene_query() -> Chem.Mol:
    """
    Get a template molecule for carbene. The template is an atom with >= 2 radical electrons.
    """
    query = Chem.MolFromSmarts("*")
    radical_query = Chem.rdqueries.NumRadicalElectronsGreaterQueryAtom(1)
    query.GetAtomWithIdx(0).ExpandQuery(radical_query)
    return query


def saturate_carbene(
    mol: Chem.Mol,
    multiplicity: int,
):
    """
    A method help to saturate carbenes and nitrenes to match the given
    molecule spin multiplicity::

        *C* (triplet) => C(**) (singlet)

    In the current implementation, no error will be raised,
    if the function doesn't achieve the goal. This function has not been
    been tested on nitrogenate.

    Args:
        multiplicity (int): The target multiplicity.
        verbose (int): Whether to print additional information. Defaults to ``True``.
    """
    cur_mult = get_spin_multiplicity(mol)
    if not _check_viability(cur_mult, multiplicity):
        return

    num_e_to_pair = cur_mult - multiplicity

    carbene_atoms = mol.GetSubstructMatches(get_carbene_query())
    if not carbene_atoms:
        return

    for aidx in carbene_atoms:
        atom = mol.GetAtomWithIdx(aidx[0])  # aidx here is a tuple with single element
        while atom.GetNumRadicalElectrons() >= 2 and num_e_to_pair:
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 2)
            num_e_to_pair -= 2
        if not num_e_to_pair:
            break

    mol.UpdatePropertyCache(strict=False)

    if num_e_to_pair:
        logger.debug(
            f"Target spin multiplicity {multiplicity} cannot be fulfilled by"
            f" saturating carbene-like sites for {mol}"
        )


@lru_cache(maxsize=1)
def get_radical_site_query() -> Chem.Mol:
    """
    Get a template molecule for carbene. The template is an atom with >= 1 radical electrons.
    """
    query = Chem.MolFromSmarts("*")
    radical_query = Chem.rdqueries.NumRadicalElectronsGreaterQueryAtom(0)
    query.GetAtomWithIdx(0).ExpandQuery(radical_query)
    return query


@lru_cache(maxsize=10)
def get_biradical_cdb_query(num_segment: int) -> Chem.Mol:
    """
    Get a template molecule for 1,4 to 1,N biradical. The template has the two end atoms each with >= 1 radical electrons.
    The template used is something like '*-,=*=,#*-,=*=,#*-,=*'

    Args:
        num_segment (int): The number of segments in the template. E.g., 1,4 biradical has 1 segments.

    Returns:
        Chem.Mol: the structure query.
    """
    segment = "-,=*=,#*"
    query_str = "*" + segment * num_segment + "-,=*"
    query = Chem.MolFromSmarts(query_str)
    radical_query = Chem.rdqueries.NumRadicalElectronsGreaterQueryAtom(0)
    query.GetAtomWithIdx(0).ExpandQuery(radical_query)
    query.GetAtomWithIdx(1 + 2 * num_segment).ExpandQuery(radical_query)
    return query


def saturate_biradical_cdb(mol: Chem.Mol, multiplicity: int, chain_length: int = 8):
    """
    A method help to saturate biradicals that have conjugated double bond in between
    to match the given molecule spin multiplicity. E.g, 1,4 biradicals can be saturated
    if there is a unsaturated bond between them::

        *C - C = C - C* => C = C - C = C

    In the current implementation, no error will be raised,
    if the function doesn't achieve the goal. This function has not been
    been tested on nitrogenate.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        multiplicity (int): The target multiplicity.
        chain_length (int): How long the conjugated double bond chain is.
                            A larger value will result in longer computational time.
                            Defaults to ``8``.
    """
    cur_multiplicity = get_spin_multiplicity(mol)
    if not _check_viability(cur_multiplicity, multiplicity):
        return

    num_dbs = (cur_multiplicity - multiplicity) // 2

    rad_atom_pool = set(chain(*mol.GetSubstructMatches(get_radical_site_query())))
    if len(rad_atom_pool) < 2:  # Need at least 2 atoms
        return

    chain_length = min(mol.GetNumAtoms(), chain_length)

    # Find all paths satisfy *C -[- C = C -]n- C*
    # n = 0 is skipped because it is already handled by saturate_biradical_12
    # n = 1, 2, 3 is corresponding to chain length = 4, 6, 8
    for path_length in range(4, chain_length + 1, 2):
        if num_dbs == 0:
            # problem solved in the previous run
            break

        seg_num = (path_length - 2) // 2
        paths = mol.GetSubstructMatches(get_biradical_cdb_query(seg_num))
        logger.debug(f"Found paths for {path_length} chain length:\n{paths}")

        while num_dbs and paths:
            for path in paths:
                # Switch over the bond types
                # E.g., C-C=C-C => C=C-C=C
                bonds = [
                    mol.GetBondBetweenAtoms(*path[i : i + 2])
                    for i in range(path_length - 1)
                ]

                new_bond_types = [
                    BOND_ORDERS.get(bond.GetBondType() + (-1) ** i)
                    for i, bond in enumerate(bonds)
                ]

                if any([bond_type is None for bond_type in new_bond_types]):
                    # Although a match is found, cannot decide what bond to make, skip
                    logger.debug(f"Cannot determine how to saturate {path} for {mol}")
                    continue

                for bond, bond_type in zip(bonds, new_bond_types):
                    bond.SetBondType(bond_type)

                # Modify radical site properties
                for atom in [
                    mol.GetAtomWithIdx(path[0]),
                    mol.GetAtomWithIdx(path[-1]),
                ]:
                    decrement_radical(atom)
                    # remove it from the list if it is no longer a radical
                    if atom.GetNumRadicalElectrons() == 0:
                        rad_atom_pool.remove(atom.GetIdx())

                num_dbs -= 1
                break
            else:  # Tried all paths, none of them works, break
                break
            # Whenever a path is updated, the available radical sites may change
            # Therefore, we need to update the paths
            paths = [
                path
                for path in paths
                if path[0] in rad_atom_pool and path[-1] in rad_atom_pool
            ]

        mol.UpdatePropertyCache(strict=False)

    if num_dbs:
        logger.debug(
            f"Target spin multiplicity {multiplicity} cannot be fulfilled by"
            f" saturating conjugated double bond biradicals for {mol}"
        )


def saturate_mol(
    mol: Chem.Mol,
    multiplicity: int = 0,
    chain_length: int = 8,
    verbose: bool = False,
):
    """
    A method help to saturate the molecule to match the given
    molecule spin multiplicity. This is just a wrapper to call
    :func:`SaturateBiradicalSites12`, :func:`SaturateBiradicalSitesCDB`, and
    :func:`SaturateCarbene`::

        *C - C* => C = C
        *C - C = C - C* => C = C - C = C
        *-C-* (triplet) => C-(**) (singlet)

    In the current implementation, no error will be raised,
    if the function doesn't achieve the goal. This function has not been
    been tested on nitrogenate.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        multiplicity (int): The target multiplicity. If ``0``, the target multiplicity will be inferred from the number of unpaired electrons.
                            Defaults to ``0``.
        chain_length (int): How long the conjugated double bond chain is. A larger value will result in longer computational time.
                            Defaults to ``8``. It should be an even number >= 4.
    """
    # Infer the possible lowest spin multiplicity from the number of unpaired electrons
    multiplicity = multiplicity or (1 if get_spin_multiplicity(mol) % 2 else 2)

    saturate_biradical_12(mol, multiplicity=multiplicity)
    saturate_biradical_cdb(mol, multiplicity=multiplicity, chain_length=chain_length)
    saturate_carbene(mol, multiplicity=multiplicity)

    if get_spin_multiplicity(mol) != multiplicity and verbose:
        logger.warn(
            f"Target spin multiplicity {multiplicity} cannot be fulfilled by"
            f" saturating methods for {mol}"
        )
