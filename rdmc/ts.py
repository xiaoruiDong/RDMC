#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with Transition states.
"""

from typing import List, Set, Tuple, Union

import numpy as np
from scipy import optimize
from scipy.spatial import distance

from rdmc import RDKitMol
from rdmc.mathlib.geom import get_centroid, get_max_distance_from_center, rotate, translate, translate_centroid
from rdmc.utils import PERIODIC_TABLE as PT


def _get_bonds_as_sets(*mols: Union['RDKitMol', 'Mol'],
                       ) -> Tuple[Set]:
    """
    Get the set of bonds for the provided list of mols.

    Args:
        mols ('RDKitMol' or 'Mol'): a RDKit Mol object

    Returns
        Tuple[Set]: (bond set in the reactant, bond set in the product)
    """
    return tuple(set(mol.GetBondsAsTuples()) for mol in mols)


def get_formed_bonds(r_mol: Union['RDKitMol', 'Mol'],
                     p_mol:  Union['RDKitMol', 'Mol'],
                     ) -> List:
    """
    Get all bonds formed in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns
        list: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    return list(p_bonds - r_bonds)


def get_broken_bonds(r_mol: Union['RDKitMol', 'Mol'],
                     p_mol:  Union['RDKitMol', 'Mol'],
                     ) -> List:
    """
    Get all bonds broken in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    return list(r_bonds - p_bonds)


def get_formed_and_broken_bonds(r_mol: Union['RDKitMol', 'Mol'],
                                p_mol:  Union['RDKitMol', 'Mol'],
                                ) -> List:
    """
    Get all bonds broken in the reaction. Both reactant and product complexes
    need to be atom-mapped. This function doesn't count bonds whose bond order
    is lowered but not equal to zero.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: - formed bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
              - broken bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    return (list(p_bonds - r_bonds),
            list(r_bonds - p_bonds))


def get_all_changing_bonds(r_mol: Union['RDKitMol', 'Mol'],
                           p_mol: Union['RDKitMol', 'Mol'],
                           ) -> List:
    """
    Get all bonds changed in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: - formed bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
              - broken bonds: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
              - bonds with BO changed: A list of length-2 tuples that contain the atom indexes of the bonded atoms.
    """
    r_bonds, p_bonds = _get_bonds_as_sets(r_mol, p_mol)
    formed_bonds, broken_bonds = p_bonds - r_bonds, r_bonds - p_bonds
    changed_bonds = [bond for bond in (r_bonds & p_bonds)
                     if r_mol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble() != \
                        p_mol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble()]
    return list(formed_bonds), list(broken_bonds), changed_bonds


def clean_ts(r_mol: 'RDKitMol',
             p_mol: 'RDKitMol',
             ts_mol: 'RDKitMol'):
    """
    Cleans transition state `ts_mol` by removing all bonds that correspond to broken or formed bonds.
    `r_mol`, `p_mol`, and `ts_mol` need to be atom mapped. Bond order changes are not considered.

    Args:
        r_mol (RDKitMol): the reactant complex.
        p_mol (RDKitMol): the product complex.
        ts_mol (RDKitMol): the transition state corresponding to `r_mol` and `p_mol`.

    Returns:
        RDKitMol: an edited version of ts_mol, which is the original `ts_mol` with cleaned bonding.
        list: broken bonds: A list of length-2 tuples that contains the atom indexes of the bonds broken in the rxn.
              formed bonds: A list of length-2 tuples that contains the atom indexes of the bonds formed in the rxn.
    """
    r_bonds, p_bonds, ts_bonds = _get_bonds_as_sets(r_mol, p_mol, ts_mol)
    formed_bonds, broken_bonds = p_bonds - r_bonds, r_bonds - p_bonds

    edited_ts_mol = ts_mol.Copy()
    edit_bonds = ts_bonds - (r_bonds & p_bonds)  # bonds in ts but not in r_mol and p_mol simultaneously
    for bond_idxs in edit_bonds:
        edited_ts_mol.RemoveBond(*bond_idxs)
        edited_ts_mol.AddBond(*bond_idxs)

    return edited_ts_mol, list(broken_bonds), list(formed_bonds)


def is_DA_rxn_endo(r_mol: 'RDKitMol',
                   p_mol: 'RDKitMol',
                   embed: bool = False):
    """
    Determine the Diels Alder reaction stereo type (endo or exo),
    based on the provided reactants and products.

    Args:
        r_mol (RDKitMol): the reactant complex.
        p_mol (RDKitMol): the product complex.
        embed (bool): bool. If the DA product has no conformer embedded.
                            Whether to embed a conformer. Defaults to ``False``.
    """
    frags = r_mol.GetMolFrags()

    if len(frags) == 1:
        # This reaction is defined in the reverse direction:
        # DA_product <=> diene + dienophile
        r_mol, p_mol = p_mol, r_mol
        frags = r_mol.GetMolFrags()

    assert len(frags) == 2

    if p_mol.GetNumConformers() == 0 and embed:
        p_mol.EmbedConformer()
    elif p_mol.GetNumConformers() == 0:
        raise ValueError('The provided DA product has no geometry available'
                         'Cannot determine the stereotype of the DA reaction')

    # Analyze the reaction center
    formed, _, changing = get_all_changing_bonds(r_mol, p_mol)
    # `fbond_atoms` are atoms in the formed bonds
    fbond_atoms = set([atom for bond in formed for atom in bond])
    for bond in changing:
        bond = set(bond)
        if len(bond & fbond_atoms) == 0:
            # Find the single bond in the diene
            dien_sb = list(bond)
        elif len(bond & fbond_atoms) == 2:
            # Find the double bond of the dienophile
            dinp_db = list(bond)
    # Make `fbond_atoms` for convenience in slicing
    fbond_atoms = list(fbond_atoms)

    # Find the atom indexes in diene and dienophile
    _, dienophile = frags if dien_sb[0] in frags[0] else frags[::-1]

    # Get the 3D geometry of the DA product
    # Create a reference plane from atoms in formed bonds
    # The reference point is chosen to be the center of the plane
    xyz = p_mol.GetPositions()
    ref_plane = xyz[fbond_atoms]
    ref_pt = ref_plane.mean(axis=0, keepdims=True)

    # Use the total least square algorithm to find
    # the normal vector of the reference plane
    A = ref_plane - ref_pt
    norm_vec = np.linalg.svd(A.T @ A)[0][:, -1].reshape(1, -1)

    # Use the vector between middle point of the diene single point
    # and the reference point as one direction vector
    dien_vec = xyz[dien_sb, :].mean(axis=0, keepdims=True) - ref_pt

    # Use the vector between mass center of the dienophile
    # and the reference point as one direction vector
    # excluding atom in dienophile's double bond
    atom_scope = [atom for atom in dienophile if atom not in dinp_db]
    mass_vec = [r_mol.GetAtomWithIdx(i).GetMass() for i in atom_scope]
    wt_xyz = (xyz[atom_scope, :] * np.reshape(mass_vec, (-1, 1)))
    dinp_vec = wt_xyz.mean(axis=0, keepdims=True) - ref_pt

    # Endo is determined by if dien_vec has the same direction as the dinp_vec
    # using the normal vector of the reference plane as a reference direction
    endo = ((norm_vec @ dien_vec.T) * (norm_vec @ dinp_vec.T)).item() > 0
    return endo


def guess_rxn_from_normal_mode(xyz: np.array,
                               symbols: np.array,
                               disp: np.array,
                               amplitude: Union[float, list] = 0.25,
                               weights: Union[bool, np.array] = True,
                               backend: str = 'openbabel',
                               multiplicity: int = 1):
    """
    Guess reaction according to the normal mode analysis for a TS.

    Args:
        xyz (np.array): The xyz coordinates of the transition state. It should have a
                        size of N x 3.
        symbols (np.array): The symbols of each atoms. It should have a size of N.
        disp (np.array): The displacement of the normal mode. It should have a size of
                        N x 3.
        amplitude (float): The amplitude of the motion. If a single value is provided then the guess
                           will be unique (if available). 0.25 will be the default. Otherwise, a list
                           can be provided, and all possible results will be returned.
        weights (bool or np.array): If ``True``, use the sqrt(atom mass) as a scaling factor to the displacement.
                              If ``False``, use the identity weights. If a N x 1 ``np.array` is provided, then
                              The concern is that light atoms (e.g., H) tend to have larger motions
                              than heavier atoms.
        backend (str): The backend used to perceive xyz. Defaults to ``'openbabel'``.
        multiplicity (int): The spin multiplicity of the transition states. Defaults to 1.

    Returns:
        list: a list of potential reactants
        list: a list of potential products
    """
    if isinstance(amplitude, float):
        amplitude = [amplitude]

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array([PT.GetAtomicWeight(PT.GetAtomicNumber(symbol)) for symbol in symbols]).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((xyz.shape[0], 1))

    mols = {'r': [], 'p': []}
    for amp in amplitude:
        xyzs = xyz - amp * disp * weights, xyz + amp * disp * weights

        # Create the reactant complex and the product complex
        for xyz_mod, side in zip(xyzs, ['r', 'p']):
            xyz_str = ''
            for symbol, coords in zip(symbols, xyz_mod):
                xyz_str += f'{symbol:4}{coords[0]:14.8f}{coords[1]:14.8f}{coords[2]:14.8f}\n'
            try:
                mols[side].append(RDKitMol.FromXYZ(xyz_str, header=False, backend=backend))
                mols[side][-1].SaturateMol(multiplicity=multiplicity)
            except:
                # Need to provide a more precise error in the future
                # Cannot generate the molecule from XYZ
                pass

    # Pairwise isomorphic comparison
    for side, ms in mols.items():
        prune = []
        len_mol = len(ms)
        for i in range(len_mol):
            if i in prune:
                continue
            for j in range(i + 1, len_mol):
                if j not in prune and ms[i].GetSubstructMatch(ms[j]):
                    prune.append(j)
        mols[side] = [mol for i, mol in enumerate(ms) if i not in prune]

    return tuple(mols.values())


def examine_normal_mode(r_mol: RDKitMol,
                        p_mol: RDKitMol,
                        ts_xyz: np.array,
                        disp: np.array,
                        amplitude: Union[float, list] = 0.25,
                        weights: Union[bool, np.array] = True,
                        verbose: bool = True,
                        as_factors: bool = True):
    """
    Examine a TS's imaginary frequency given a known reactant complex and a
    product complex. The function checks if the bond changes are corresponding
    to the most significant change in the normal mode. The reactant and product
    complex need to be atom mapped.

    Args:
        r_mol ('RDKitMol'): the reactant complex.
        p_mol ('RDKitMol'): the product complex.
        ts_xyz (np.array): The xyz coordinates of the transition state. It should have a
                           size of N x 3.
        disp (np.array): The displacement of the normal mode. It should have a size of
                         N x 3.
        amplitude (float): The amplitude of the motion. Defaults to 0.25.
        weights (bool or np.array): If ``True``, use the sqrt(atom mass) as a scaling factor to the displacement.
                              If ``False``, use the identity weights. If a N x 1 ``np.array` is provided, then
                              The concern is that light atoms (e.g., H) tend to have larger motions
                              than heavier atoms.
        verbose (bool): If print detailed information. Defaults to ``True``.
        as_factors (bool): If return the value of factors instead of a judgment.
                           Defaults to ``False``

    Returns:
        - bool: ``True`` for pass the examination, ``False`` otherwise.
        - list: If `as_factors == True`, two factors will be returned.
    """
    # Analyze connectivity
    broken, formed, changed = get_all_changing_bonds(r_mol, p_mol)
    reacting_bonds = broken + formed + changed

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array(r_mol.GetAtomMasses()).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((ts_xyz.shape[0], 1))

    # Generate conformer instance according to the displacement
    xyzs = ts_xyz - amplitude * disp * weights, ts_xyz + amplitude * disp * weights
    r_copy = r_mol.Copy(); r_copy.SetPositions(xyzs[0])
    p_copy = p_mol.Copy(); p_copy.SetPositions(xyzs[1])
    r_conf, p_conf = r_copy.GetConformer(), p_copy.GetConformer()

    # Calculate bond distance change
    formed_and_broken_diff = [abs(r_conf.GetBondLength(bond) - p_conf.GetBondLength(bond))
                              for bond in broken + formed]
    changed_diff = [abs(r_conf.GetBondLength(bond) - p_conf.GetBondLength(bond))
                    for bond in changed]
    other_bonds_diff = [abs(r_conf.GetBondLength(bond) - p_conf.GetBondLength(bond))
                        for bond in r_copy.GetBondsAsTuples() if bond not in reacting_bonds]

    # We expect bonds that are formed or broken in the reaction
    # have relatively large changes; For bonds that change their bond order
    # in the reaction may have a smaller factor.
    # In this function, we only use the larger factor as a check.
    # The smaller factor is less deterministic, considering the change in
    # other bonds due to the change of atom hybridization or bond conjugation.
    baseline = np.max(other_bonds_diff)
    std = np.std(other_bonds_diff)
    larger_factor = (np.min(formed_and_broken_diff) - baseline) / std
    if changed_diff:
        # There might be no bond that only changes its order
        smaller_factor = (np.min(changed_diff) - baseline) / std
    else:
        smaller_factor = 0

    if verbose:
        print(f'The min. bond distance change for bonds that are broken or formed'
              f' is {np.min(formed_and_broken_diff)} A and is {larger_factor:.1f} STD off the baseline.')
        if changed_diff:
            print(f'The min. bond distance change for bonds that are changed'
                  f' is {np.min(changed_diff)} A and is {smaller_factor:.1f} STD off the baseline.')

    if as_factors:
        return larger_factor, smaller_factor

    if larger_factor > 3:
        return True
    return False


class NaiveAlign(object):
    """
    This is a naive alignment algorithm aligns reactant conformers.
        For 1 reactant system, the algorithm simply put the center of the reactant at the origin.
        For 2 reactant system, the resulting alignment has the following charateristics:
        - the centroid for fragment 1 is at the origin.
        - the centroid for fragment 2 is at (R1 + R2 + D), where R1 is the radius of fragment 1, R2 is the radius
          of fragment 2, and D is a pre-set distance value defined by the user.
        - the centroid of the reacting atoms in fragment 1 should be around the line (0,0,0) => (1,0,0).
        - the centroid of the reacting atoms in fragment 2 should be around the line (0,0,0) => (1,0,0).
        - the distance between atoms to form bonds are minimized.
        # TODO: [long term goal]
        For 3 reactant system, the feature is under-development. There should be two cases: close-linear alignment and triangle alignment,
        depending on the max number of interactions among fragments.
        For 4 reactant system, the feature is under-development. There should be three cases: close-linear, square, and tetrahedron alignment,
        depending on the max number of interactions among fragments.
    """

    dist = 2.0

    def __init__(self,
                 coords: np.array,
                 atom_maps: List[List],
                 formed_bonds: List[tuple],
                 broken_bonds: List[tuple],
                 ) -> 'NaiveAlign':
        """
        Initialize the alignment algorithm.

        Args:
            coords (np.array): The coordinates of the reactant complex.
            atom_maps (List[List]): The atom map in the complex. E.g., ([1,2,5], [3,4]) indicates the 1st,
                                    2nd, and 5th atoms are in the first molecule and 3th and 4th atoms are
                                    in the second molecule.
            formed_bonds (List[tuple]): The bonds that are formed in the reaction. E.g., [(1,2)] indicates
                                        atoms 1 and 2 will form a bond in the reaction.
            broken_bonds (List[tuple]): The bonds that are broken in the reaction. E.g., [(1,2)] indicates
                                        the bond between atoms 1 and 2 will be broken in the reaction.
        """
        self.coords = coords
        self.atom_maps = atom_maps
        self.formed_bonds = formed_bonds
        self.broken_bonds = broken_bonds
        self.reacting_atoms, self.non_reacting_atoms = self.get_reacting_atoms_in_fragments()
        # `interaction` is not used in the current scheme, but can be helpful when dealing with
        # multiple mol alignemnt problems
        self.interactions = [0] * len(atom_maps)
        for formed_bond in self.formed_bonds:
            for i, reacting_atom in enumerate(self.reacting_atoms):
                if (formed_bond[0] in reacting_atom and formed_bond[1] not in reacting_atom) \
                        or (formed_bond[0] not in reacting_atom and formed_bond[0] not in reacting_atom):
                    self.interactions[i] += 1

    def get_fragment_radii(self,) -> List[float]:
        """
        Get the radius of each fragment defined by the distance between the centroid to the farthest element.

        Returns:
            list: A list of radius.
        """
        return [get_max_distance_from_center(self.coords[atom_map, :]) for atom_map in self.atom_maps]

    def get_reacting_atoms_in_fragments(self,) -> List[list]:
        """
        Get the reacting atoms in each reactant.

        Returns:
            - A list of the list of reacting atoms in each reactant.
            - A list of the list of non reacting atoms in each reactant.
        """
        # Only formed bonds are considered. Bond breaking only takes place within the molecule and they are
        # less important when aligning molecules. If they are usually because the atoms that are connected by
        # the broken bonds are both reacting with the other reactant molecule. Also neglect bonds that are formed
        # within a single molecule
        new_formed_bonds = []
        for bond in self.formed_bonds:
            for atom_map in self.atom_maps:
                if np.logical_xor(bond[0] in atom_map, bond[1] in atom_map):
                    new_formed_bonds.append(bond)
                    break
        self.formed_bonds = new_formed_bonds
        reacting_atoms = set([i for bond in self.formed_bonds for i in bond])
        return ([[i for i in atom_map if i in reacting_atoms] for atom_map in self.atom_maps],
                [[i for i in atom_map if i not in reacting_atoms] for atom_map in self.atom_maps])

    @classmethod
    def from_reactants(cls,
                       mols: List['RDKitMol'],
                       formed_bonds: List[tuple],
                       broken_bonds: List[tuple],
                       conf_ids: List[int] = None,
                       ):
        """
        Create a complex in place by stacking the molecules together.

        Args:
            mols (RDKitMol): A list of reactants.
            formed_bonds (List[tuple]): bonds formed in the reaction.
            broken_bonds (List[tuple]): bonds broken in the reaction.
            conf_id1 (int, optional): The conformer id to be used in `mol1`. Defaults to 0.
            conf_id2 (int, optional): The conformer id to be used in `mol2`. Defaults to 0.
        """
        if conf_ids is None:
            conf_ids == [0] * len(mols)
        elif len(conf_ids) != len(mols):
            raise ValueError(f'The conf_ids\'s length (currently {len(conf_ids)}) should be '
                             f'the same as the length of moles (currently {len(mols)}.')
        coord_list = [mol.GetPostions(id=conf_id) for mol, conf_id in zip(mols, conf_ids)]
        coords  = np.concatenate(coord_list, axis=0)
        atom_maps, counter = [], 0
        for mol in mols:
            atom_maps.append(list(range(counter, counter + mol.GetNumAtoms())))
            counter += mol.GetNumAtoms()
        return cls(coords, atom_maps, formed_bonds, broken_bonds)

    @classmethod
    def from_complex(cls,
                     r_complex: 'RDKitMol',
                     formed_bonds: List[tuple],
                     broken_bonds: List[tuple],
                     conf_id: int = 0):
        """
        Initialize from a reactant complex.

        Args:
            r_complex (RDKitMol): The reactant complex.
            formed_bonds (List[tuple]): bonds formed in the reaction.
            broken_bonds (List[tuple]): bonds broken in the reaction.
            conf_id (int, optional): The conformer id to be used in the `complex`. Defaults to 0.
        """
        coords = r_complex.GetPositions(id=conf_id)
        atom_maps = [list(atom_map) for atom_map in r_complex.GetMolFrags()]
        return cls(coords, atom_maps, formed_bonds, broken_bonds)

    @classmethod
    def from_r_and_p_complex(cls,
                             r_complex: 'RDKitMol',
                             p_complex: 'RDKitMol',
                             conf_id: int = 0,
                             ):
        """
        Initialize from the reactant complex and the product complex. The product complex
        does not need to have conformers embedded, however, it should be atom mapped with
        the reactant complex.

        Args:
            r_complex (RDKitMol): The reactant complex.
            p_complex (RDKitMol): The product complex.
            conf_id (int, optional): The conformer id to be used in the reactant complex `r_complex`. Defaults to 0.
        """
        coords = r_complex.GetPositions(id=conf_id)
        atom_maps = [list(atom_map) for atom_map in r_complex.GetMolFrags()]
        formed_bonds, broken_bonds = get_formed_and_broken_bonds(r_complex, p_complex)
        return cls(coords, atom_maps, formed_bonds, broken_bonds)

    def rotate_fragment_separately(self,
                                   *angles: np.array,
                                   about_reacting: bool = False,
                                   ) -> np.array:
        """
        Rotate the molecule fragments in the complex by angles. The length of angles should be same as the length of
        `self.atom_maps`.

        Args:
            angles (np.array): Rotation angles for molecule fragment 1. It should be an array with a
                               size of (1,3) indicate the rotation angles about the x, y, and z axes, respectively.
            about_reacting (bool, optional): If rotate about the reactor center instead of the centroid. Defaults to False.

        Returns:
            np.array: The coordinates after the rotation operation.
        """
        coords = np.copy(self.coords)
        for i in range(len(angles)):
            atom_map = self.atom_maps[i]
            if about_reacting:
                kwargs = {'about': get_centroid(coords[self.reacting_atoms[i], :])}
            else:
                kwargs = {'about_center': True}
            coords[atom_map, :] = rotate(self.coords[atom_map, :],
                                         angles[i],
                                         **kwargs)
        return coords

    def initialize_align(self,
                         dist: float = None,
                         ):
        """
        Initialize the alignment for the reactants. Currently only available for 1 reactant and 2 reactant systems.

        Args:
            dist (float, optional): The a preset distance used to separate molecules. Defaults to None meaning using the value of `self.dist`.
        """
        if dist is not None and dist > 0:
            self.dist = dist

        if len(self.atom_maps) == 1:
            self.coords = translate_centroid(self.coords,
                                             np.zeros(3))
        elif len(self.atom_maps) == 2:
            pos = [np.zeros(3), np.array([self.dist, 0., 0.])]
            for i in [0, 1] :
                atom_map = self.atom_maps[i]
                # Make the first fragment centered at (0, 0, 0)
                # Make the second fragment centered at (R1 + R2 + dist)
                self.coords[atom_map, :] = translate(self.coords[atom_map, :],
                                                     -get_centroid(self.coords[self.reacting_atoms[i], :] + pos[i]))
        else:
            raise NotImplementedError('Hasn\'t been implemented for 3 and 4 reactant systems.')

    def score_bimolecule(self,
                         angles: np.array,
                         ) -> float:
        """
        Calculate the score of bimolecule alignment.

        Args:
            angles (np.array): an array with 6 elements. The first 3 angles correspond to the rotation of the first fragment,
                               and the last 3 angles correspond to the rotation of the second fragment.

        Returns:
            float: The score value.
        """
        angles = angles[:3].reshape(1, -1), angles[3:].reshape(1, -1)
        # Rotate fragment according to the input angle
        coords = self.rotate_fragment_separately(*angles, about_reacting=True)
        # A reference distance matrix
        dist_ref = self.dist

        # An orientation related score. This makes parallel alignment more favorable then normal alignment
        # This is motivated by the fact that when computing the distance between a point and two points
        # while fixing the distance between their centroid. The two points always favor a normal alignment
        # if min d^1 or no favor over any alignment if min d^2
        score1 = 0.
        v1 = np.array([1., 0., 0.])
        if len(self.reacting_atoms[0]) == 2:
            v2 = coords[self.reacting_atoms[0][0], :] - coords[self.reacting_atoms[0][1], :]
            v2_norm = np.linalg.norm(v2, ord=2)
            score1 += (v1 @ v2 / v2_norm) ** 2  # range from 0 - 1
        if len(self.reacting_atoms[1]) == 2:
            v2 = coords[self.reacting_atoms[1][0], :] - coords[self.reacting_atoms[1][1], :]
            v2_norm = np.linalg.norm(v2, ord=2)
            score1 += (v1 @ v2 / v2_norm) ** 2  # range from 0 - 1

        # An bonding related score. This makes the atoms that are forming bonds tend to get closer.
        # Square euclideans distance is used as score for each bond.
        score2 = 0.
        for bond in self.formed_bonds:
                # Only bonds form between fragments will help decrease this score3
                # Bonds formed inside a fragment will not change this score since molecules are rigid and the distances are fixed
            score2 += np.sum((coords[bond[0], :] - coords[bond[1], :]) ** 2)
        score2 = score2 / len(self.formed_bonds) / dist_ref

        # An interaction related score. Briefly, it (may) be more favorable to have non-reacting atoms in one fragment
        # being away from the reacting fragment. Use a coulomb like interaction as the score 1/r^2. This calculation scales with N.
        score3 = 0.
        # Get the centroids of reacting centers
        react_atom_center = [get_centroid(coords[self.reacting_atoms[i], :]) for i in range(2)]
        for i in [0, 1]:
            if len(self.non_reacting_atoms[i]):
                score3 += np.sum(1 / distance.cdist(coords[self.non_reacting_atoms[i],:], react_atom_center, 'sqeuclidean'))

        return score1 + score2 + score3

    def get_alignment_coords(self,
                             dist: float = None,):
        """
        Get coordinates of the alignment.

        Args:
            dist (float, optional): The a preset distance used to separate molecules. Defaults to None meaning using the value of `self.dist`.
        """
        self.initialize_align(dist=dist,)

        if len(self.atom_maps) == 1:
            return self.coords
        elif len(self.atom_maps) == 2:
            result = optimize.minimize(self.score_bimolecule,
                                       2 * np.pi * (np.random.rand(6) - 0.5),
                                       method='BFGS',
                                       options={'maxiter': 5000, 'disp': False})
            angles = result.x[:3].reshape(1, -1), result.x[3:].reshape(1, -1)
            return self.rotate_fragment_separately(*angles, about_reacting=True)
        else:
            raise NotImplementedError('Hasn\'t been implemented for 3 and 4 reactant systems.')

    def __call__(self, dist: float = None,):
        """
        Get coordinates of the alignment. Same as `self.get_alignment`

        Args:
            dist (float, optional): The a preset distance used to separate molecules. Defaults to None meaning using the value of `self.dist`.
        """
        return self.get_alignment_coords(dist=dist)
