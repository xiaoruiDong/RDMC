#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for reaction path analysis.
"""

from typing import List, Set, Tuple, Union

import numpy as np
from scipy import optimize
from scipy.spatial import distance

from rdmc import RDKitMol
from rdmc.mathlib.geom import get_centroid, get_max_distance_from_center, rotate, translate, translate_centroid
from rdmc.ts import get_broken_bonds, get_formed_and_broken_bonds
from rdmc.forcefield import OpenBabelFF


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


def reset_pmol(r_mol, p_mol):
    """
    Reset the product mol to best align with the reactant. This procedure consists of initializing the product 3D
    structure with the reactant coordinates and then 1) minimizing the product structure with constraints for broken
    bonds and 2) performing a second minimization with no constraints

    Args:
        r_mol ('RDKitMol' or 'Mol'): a RDKit Mol object
        p_mol ('RDKitMol' or 'Mol'): a RDKit Mol object

    Returns
        new_p_mol: The new product mol with changed coordinates
    """
    # copy current pmol and set new positions
    p_mol_new = p_mol.Copy(quickCopy=True)
    p_mol_new.SetPositions(r_mol.GetPositions())

    # setup first minimization with broken bond constraints
    obff = OpenBabelFF(force_field="uff")
    obff.setup(p_mol_new)
    broken_bonds = get_broken_bonds(r_mol, p_mol)
    r_conf = r_mol.GetConformer()
    current_distances = [r_conf.GetBondLength(b) for b in broken_bonds]
    [obff.add_distance_constraint(b, 1.5*d) for b, d in zip(broken_bonds, current_distances)]
    obff.optimize(max_step=2000)

    # second minimization without constraints
    obff.constraints = None
    obff.optimize(max_step=2000)
    p_mol_intermediate = obff.get_optimized_mol()

    # third optimization with MMFF94s
    obff = OpenBabelFF(force_field="mmff94s")
    obff.setup(p_mol_intermediate)
    obff.optimize(max_step=2000)

    return obff.get_optimized_mol()


def prepare_mols(r_mol, p_mol, align_bimolecular=True):
    """
    Prepare mols for reaction path analysis. If reactant has multiple fragments, first orient reactants in reacting
    orientation. Then, reinitialize coordinates of product using reset_pmol function

    Args:
        r_mol ('RDKitMol' or 'Mol'): a RDKit Mol object
        p_mol ('RDKitMol' or 'Mol'): a RDKit Mol object
        align_bimolecular (bool, optional): Whether or not to use alignment algorithm on bimolecular reactions
                                            (defaults to True)

    Returns
        r_mol, new_p_mol: The new reactant and product mols
    """
    if len(r_mol.GetMolFrags()) == 2:
        if align_bimolecular:
            r_mol = align_reactant_fragments(r_mol, p_mol)
    p_mol_new = reset_pmol(r_mol, p_mol)  # reconfigure p_mol as if starting from SMILES
    return r_mol, p_mol_new


def align_reactant_fragments(r_mol, p_mol):
    """
    Given reactant and product mols, find details of formed and broken bonds and generate reacting reactant complex

    Args:
        r_mol ('RDKitMol' or 'Mol'): a RDKit Mol object
        p_mol ('RDKitMol' or 'Mol'): a RDKit Mol object

    Returns
        r_mol_naive_align: The new reactant with aligned fragments
    """
    formed_bonds, broken_bonds = get_formed_and_broken_bonds(r_mol, p_mol)
    if len(formed_bonds + broken_bonds) == 0:
        print("Careful! No broken or formed bonds in this reaction! Returning input reactants")
        return r_mol
    naive_align = NaiveAlign.from_complex(r_mol, formed_bonds, broken_bonds)
    r_mol_naive_align = r_mol.Copy(quickCopy=True)
    r_mol_naive_align.SetPositions(naive_align())
    return r_mol_naive_align
