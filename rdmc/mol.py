#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit RWMol, Mol.
"""

from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, RWMol, Conformer
from rdkit.Geometry.rdGeometry import Point3D

from rdmc.conf import RDKitConf
from rdmc.utils import *
from rdmc.external.xyz2mol import parse_xyz_by_jensen


# Additional notes:
# Although current .py does not contain openbabel, but actually
# openbabel import is put behind rdkit looking globally
# This relates to https://github.com/rdkit/rdkit/issues/2292
# For Mac, with anaconda build:
# rdkit -  2020.03.2.0 from rdkit channel
# openbabel -  2.4.1 from rmg channel
# works fine without sequence problem
# For Linux,
# rdkit - 2020.03.3 from rdkit channel
# openbabel - 2.4.1 from rmg channel does not work


# Keep the representation method from rdchem.Mol
KEEP_RDMOL_ATTRIBUTES = ['_repr_html_',
                         '_repr_png_',
                         '_repr_svg_']

class RDKitMol(object):
    """
    A helpful wrapper for rdchem.Mol.
    The method nomenclature follows the Camel style to be consistent with RDKit.
    It keeps almost all of the orignal method of Chem.rdchem.Mol/RWMol, but add few useful
    shortcuts, so that a user doesn't need to refer to other RDKit modules.
    """

    def __init__(self,
                 mol: Union[Mol, RWMol],
                 keepAtomMap: bool=True):
        """
        Generate an RDKitMol Molecule instance from a RDKit ``Chem.rdchem.Mol`` or ``RWMol`` molecule.

        Args:
            mol (Union[Mol, RWMol]): The RDKit ``Chem.rdchem.Mol`` / ``RWmol`` molecule to be converted.
            keepAtomMap (bool, optional): Whether keep the original atom mapping. Defaults to True.
                                          If no atom mapping is stored in the molecule, atom mapping will
                                          be created based on atom indexes.
        """
        # keep the link to original molecule so we can easily recover it if needed.
        if isinstance(mol, RWMol):
            self._mol = mol
        elif isinstance(mol, Mol):
            self._mol = RWMol(mol)   # Make the original Mol a writable object
        else:
            raise ValueError(f'mol should be rdkit.Chem.rdchem.Mol / RWMol. '
                             f'Got: {type(mol)}')

        # Link methods of rdchem.Mol to the new instance
        for attr in dir(self._mol):
            # Not reset private properties and repeated properties
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(self._mol, attr,))
            elif attr in KEEP_RDMOL_ATTRIBUTES:
                setattr(self, attr, getattr(self._mol, attr,))

        # Set atom map number if it is not stored in mol or if it is invalid.
        if keepAtomMap:
            if not np.any(self.GetAtomMapNumbers()):
                # No stored atom mapping, so set it anyway
                self.SetAtomMapNumbers()
        else:
            self.SetAtomMapNumbers()

        # Perceive rings
        self.GetSymmSSSR()

    def AddRedundantBonds(self, bonds: Iterable):
        """
        Add redundant bonds (not originally exist in the molecule) for
        the convenience for some molecule operation or analysis. This function
        will only generate a copy of the molecule and no change is conducted inplace.

        Args:
            bonds: a list of length-2 Iterables containing the index of the ended atoms.
        """
        mol_cp = self.Copy()
        for bond in bonds:
            mol_cp.AddBond(*bond, Chem.BondType.SINGLE)
        return mol_cp

    def AlignMol(self,
                 prbMol: Union['RDKitMol', 'RWMol', 'Mol'] = None,
                 refMol: Union['RDKitMol', 'RWMol', 'Mol'] = None,
                 prbCid: int = 0,
                 refCid: int = 0,
                 reflect: bool = False,
                 atomMaps: list = None,
                 maxIters: int = 1000,
                 weights: list = [],
                 ) -> float:
        """
        Align molecules based on a reference molecule. This function will also return the RMSD value for the best alignment.
        When leaving both ``prbMol`` and ``refMol`` blank, then the function will match the current molecules' conformers, and
        PrbCid or refCid must be provided.

        Args:
            refMol (Mol): RDKit molecule as a reference. Should not be provided with ``prbMol``.
            prbMol (Mol): RDKit molecules to align to the current molecule. Should not be provided with ``refMol``.
            prbCid (int, optional): The conformer id to be aligned. Defaults to ``0``.
            refCid (int, optional): The id of reference conformer. Defaults to ``0``.
            reflect (bool, optional): Whether to reflect the conformation of the probe molecule.
                                      Defaults to ``False``.
            atomMap (list, optional): a vector of pairs of atom IDs ``(probe AtomId, ref AtomId)``
                                      used to compute the alignments. If this mapping is not
                                      specified an attempt is made to generate on by substructure matching.
            maxIters (int, optional): maximum number of iterations used in mimizing the RMSD. Defaults to ``1000``.

        Returns:
            float: RMSD value.
        """
        if prbMol != None and refMol != None:
            raise ValueError('`refMol` and `prbMol` should not be provided simultaneously.')
        elif prbMol == None and refMol == None and prbCid == refCid:
            raise ValueError('Cannot match the same conformer for the given molecule. `prbCid` and `refCid` needs'
                             'to be different if either `prbMol` or `refMol` is not provided.')

        refMol  = refMol or self
        prbMol = prbMol or self

        if atomMaps is None:
            atomMaps = [list(enumerate(range(self.GetNumAtoms())))]
        if reflect:
            prbMol.Reflect(id=prbCid)
        rmsd = np.inf
        for atom_map in atomMaps:
            cur_rmsd = Chem.rdMolAlign.AlignMol(refMol=refMol._mol,
                                                prbMol=prbMol._mol,
                                                prbCid=prbCid,
                                                refCid=refCid,
                                                atomMap=atom_map,
                                                reflect=reflect,
                                                maxIters=maxIters,
                                                weights=weights,
                                                )
            if cur_rmsd < rmsd:
                rmsd = cur_rmsd
            if reflect:
                prbMol.Reflect(id=prbCid)
        return rmsd

    def CalcRMSD(self,
                prbMol: 'RDKitMol',
                prbCid: int = 0,
                refCid: int = 0,
                reflect: bool = False,
                atomMaps: list = None,
                weights: list = [],
               ):
        """
        Calculate the RMSD for between conformers of two molecules. Note this function will not align molecule, thus molecules geometries
        in the calculation are not translated or rotated. You can expect a larger number compared to the RMSD from AlignMol.

        Args:
            prbMol ('RDKitMol'): The other molecule to compare with. This can be the instance as the current molecule.
            prbCid (int, optional): The conformer ID of the current molecule to calculate RMSD. Defaults to 0.
            refCid (int, optional): The conformer ID of the other molecule to calculate RMSD. Defaults to 0.
            reflect (bool, optional): Whether to reflect the conformation of the prbMol. Defaults to ``False``.
            atomMaps (list, optional): Provide an atom mapping to calculate the RMSD. By default, prbMol and current molecule
                                      will be assumed to have the same atom order.
            weights (list, optional): Specify weights to each atom pairs. E.g., use atom weights to highlight the importance of
                                      heavy atoms.
        """
        if atomMaps is None:
            atomMaps = [list(enumerate(range(self.GetNumAtoms())))]

        if reflect:
            prbMol.Reflect(id=prbCid)
        try:
            rmsd = Chem.rdMolAlign.CalcRMS(refMol=self._mol,
                                           prbMol=prbMol.ToRWMol(),
                                           prbId=prbCid,
                                           refId=refCid,
                                           map=atomMaps,
                                           weights=weights,
                                           )
        except AttributeError:
            raise NotImplementedError(f'The RDKit version used doesn\'t support this calculation.')
        if reflect:
            prbMol.Reflect(id=prbCid)
        return rmsd

    def AssignStereochemistryFrom3D(self, confId: int = 0):
        """
        Assign the chiraltype to a molecule's atoms.
        """
        Chem.rdmolops.AssignStereochemistryFrom3D(self._mol, confId=confId)

    def CombineMol(self,
                   molFrag: 'Mol',
                   offset: Union[list, tuple, float] = 0,
                   ) -> 'RDKitMol':
        """
        A function to combine the current molecule with the given ``molFrag`` (another molecule
        or fragment). It will return a new RDKitMol instance without changing the input molecules.

        Args:
            molFrag (Mol): the molecule or fragment to be combined into the current one.
            offset:
                - (list, tuple): A 3D vector used to define the offset.
                - (float): An ratio times the distance vector between the two centroids as the offset.

        Returns:
            RDKitMol: The combined molecule.
        """
        try:
            mol = molFrag.ToRWMol()
        except AttributeError:
            mol = molFrag

        # Manually assign a 3D vector for offset
        if isinstance(offset, (list, tuple)) and len(offset) == 3:
            vector = Point3D()
            for i, coord in enumerate(['x', 'y', 'z']):
                setattr(vector, coord, offset[i])

        # Assign an offset according to the centroid vector
        elif isinstance(offset, (int, float)):
            conf = self.GetConformer()
            conf_from_mol = mol.GetConformer()
            vector = Chem.rdMolTransforms.ComputeCentroid(conf_from_mol, ignoreHs=False) - \
                     Chem.rdMolTransforms.ComputeCentroid(conf.ToConformer(), ignoreHs=False)
            for coord in ['x', 'y', 'z']:
                setattr(vector, coord, getattr(vector, coord) * offset)

        else:
            raise ValueError(f'Invalid input for offset. Got: {offset}')

        combined = Chem.rdmolops.CombineMols(self._mol, mol, vector)
        return RDKitMol(combined)

    def Copy(self) -> 'RDKitMol':
        """
        Make a copy of the RDKitMol.

        Returns:
            RDKitMol: a copied molecule
        """
        return self.RenumberAtoms(list(range(self.GetNumAtoms())),
                                  updateAtomMap=False)

    def EmbedConformer(self,
                       embed_null: bool = True,
                       **kwargs):
        """
        Embed a conformer to the ``RDKitMol``. This will overwrite current conformers. By default, it
        will first try embeding a 3D conformer; if failed, it will then try to compute 2D coordinates
        and use that for the conformer structure; if both approaches fail and allow embedding a null
        conformer, then a conformer with all zero coordinates will be embedded. The last one is still
        helpful if you have the coordinates, so you can use `SetPositions` to set them, or if you want to
        optimize the geometry using things like forcefields.

        Args:
            embed_null (bool): If Embed 3D and 2D coordinates fails, whether to embed a conformer
                               with all null coordinates: (0, 0, 0) for each atom. Defaults to ``True``.
        """
        try:
            return_code = AllChem.EmbedMolecule(self._mol, **kwargs)
        except Chem.AtomValenceException:
            try:
                AllChem.Compute2DCoords(self._mol)
                return_code = 0
            except:  # To be replaced by a more specific error type
                return_code = -1

        if return_code == -1:
            if embed_null:
                self.EmbedNullConformer()
            else:
                raise RuntimeError('Cannot embed conformer for this molecule!')

    def EmbedMultipleConfs(self,
                           n: int = 1,
                           embed_null: bool = True,
                           **kwargs):
        """
        Embed conformers to the ``RDKitMol``. This will overwrite current conformers. By default, it
        will first try embeding 3D conformers; if failed, it will then try to compute 2D coordinates
        and use that for the conformer structures; if both approaches fail and allow embedding null
        conformers, then conformers with all zero coordinates will be embedded. The last one is still
        helpful if you have the coordinates, so you can use `SetPositions` to set them, or you want to
        optimize the geometry using things like forcefields.

        Args:
            n (int): The number of conformers to be embedded. The default is ``1``.
            embed_null (bool): If embeding fails, whether to embed null conformers. Defaults to ``True``.
        """
        try:
            results = AllChem.EmbedMultipleConfs(self._mol, numConfs=n, **kwargs)
        except Chem.AtomValenceException:
            results = []

        if len(results) == 0:
            if embed_null:
                results = self.EmbedMultipleNullConfs(n=n)
            else:
                raise RuntimeError('Cannot embed conformer for this molecule!')

    def EmbedNullConformer(self,
                           random: bool = True):
        """
        Embed a conformer with meaningless atom coordinates. This helps the cases where a conformer
        can not be successfully embeded. You can choose to generate all zero coordinates or random coordinates.
        You can set to all-zero coordinates, if you will set coordinates later; You should set to random
        coordinates, if you want to optimize this molecule by forcefield (RDKit force field cannot optimize
        all-zero coordinates).

        Args:
            random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
                                     coordinates. Defaults to ``True``.
        """
        self.EmbedMultipleNullConfs(n=1, random=random)

    def EmbedMultipleNullConfs(self,
                               n: int = 10,
                               random: bool = True):
        """
        Embed conformers with meaningless atom coordinates. This helps the cases where a conformer
        can not be successfully embedded. You can choose to generate all zero coordinates or random coordinates.
        You can set to all-zero coordinates, if you will set coordinates later; You should set to random
        coordinates, if you want to optimize this molecule by forcefield (RDKit force field cannot optimize
        all-zero coordinates).

        Args:
            n (int): The number of conformers to be embedded. Defaults to ``10``.
            random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
                                     coordinates. Defaults to ``True``.
        """
        self._mol.RemoveAllConformers()
        num_atoms = self.GetNumAtoms()
        for i in range(n):
            conf = Conformer()
            coords = np.random.rand(num_atoms, 3) if random else np.zeros((num_atoms, 3))
            set_rdconf_coordinates(conf, coords)
            conf.SetId(i)
            self._mol.AddConformer(conf)

    @ classmethod
    def FromOBMol(cls,
                  obMol: 'openbabel.OBMol',
                  removeHs: bool = False,
                  sanitize: bool = True,
                  embed: bool = True,
                  ) -> 'RDKitMol':
        """
        Convert a OpenBabel Mol to an RDKitMol object.

        Args:
            obMol (Molecule): An OpenBabel Molecule object for the conversion.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, Defaults to ``False``.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to ``True``.
            embed (bool, optional): Whether to embeb 3D conformer from OBMol. Defaults to ``True``.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the input OpenBabel Molecule object.
        """
        rw_mol = openbabel_mol_to_rdkit_mol(obMol, removeHs, sanitize, embed)
        return cls(rw_mol)

    @classmethod
    def FromMol(cls,
                mol: Union[Mol, RWMol],
                keepAtomMap: bool = True,
                ) -> 'RDKitMol':
        """
        Convert a RDKit ``Chem.rdchem.Mol`` molecule to ``RDKitMol`` Molecule.

        Args:
            rdmol (Union[Mol, RWMol]): The RDKit ``Chem.rdchem.Mol`` / ``RWMol`` molecule to be converted.
            keepAtomMap (bool, optional): Whether keep the original atom mapping. Defaults to True.
                                          If no atom mapping is stored in the molecule, atom mapping will
                                          be created based on atom indexes.

        Returns:
            RDKitMol: An RDKitMol molecule.
        """
        return cls(mol, keepAtomMap=keepAtomMap)

    @classmethod
    def FromSmiles(cls,
                   smiles: str,
                   removeHs: bool = False,
                   sanitize: bool = True,
                   allowCXSMILES: bool = True,
                   keepAtomMap: bool = True,
                   ) -> 'RDKitMol':
        """
        Convert a SMILES to an ``RDkitMol`` object.

        Args:
            smiles (str): A SMILES representation of the molecule.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.
            allowCXSMILES (bool, optional): Whether to recognize and parse CXSMILES. Defaults to ``True``.
            keepAtomMap (bool, optional): Whether to keep the Atom mapping contained in the SMILES. Defaults
                                          Defaults to ``True``.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the SMILES.
        """
        params = Chem.SmilesParserParams()
        params.removeHs = removeHs
        params.sanitize = sanitize
        params.allowCXSMILES = allowCXSMILES
        mol = Chem.MolFromSmiles(smiles, params)

        # By default, for a normal SMILES (e.g.,[CH2]CCCO) other than H indexed SMILES
        # (e.g., [C+:1]#[C:2][C:3]1=[C:7]([H:10])[N-:6][O:5][C:4]1([H:8])[H:9]),
        # no hydrogens are automatically added. So, we need to add H atoms.
        if not removeHs:
            mol = Chem.rdmolops.AddHs(mol)

        # Create RDKitMol
        mol = cls(mol, keepAtomMap=keepAtomMap)

        # By default in RDKit, the atom index may not have the same sequence as the
        # atom map number. For convenience, renumber atoms.
        if keepAtomMap:
            for idx, map_num in enumerate(mol.GetAtomMapNumbers()):
                # Check if the current index order is the same as the atom mapping
                # 1. The atom mapping numbers defaults to starting from 1
                # 2. If the atom map numbers are not from 0 or they are discontinued numbers
                #    The new molecule's index will have the sequence as the atom map number
                #    But the atom map number will be kept. E.g., [H:9][C:2]([C:4]([H:1])[H:3])([H:5])[H:8]
                #    will result in a molecule with indexes and atom map numbers as
                #    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (6, 9)]. This kind of smiles
                #    may be from reaction templates.
                if idx + 1 != map_num:
                    new_order = np.argsort(mol.GetAtomMapNumbers()).tolist()
                    return mol.RenumberAtoms(new_order, updateAtomMap=False)
        return mol

    @classmethod
    def FromSmarts(cls,
                   smarts):
        """
        Convert a SMARTS to an ``RDKitMol`` object.

        Args:
            smarts (str): A SMARTS string of the molecule
        """
        mol = Chem.MolFromSmarts(smarts)
        return cls(mol)

    @classmethod
    def FromRMGMol(cls,
                   rmgMol: 'rmgpy.molecule.Molecule',
                   removeHs: bool = False,
                   sanitize: bool = True,
                   ) -> 'RDKitMol':
        """
        Convert an RMG ``Molecule`` to an ``RDkitMol`` object.

        Args:
            rmgMol ('rmg.molecule.Molecule'): An RMG ``Molecule`` instance.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the RMG Molecule.
        """
        return cls(rmg_mol_to_rdkit_mol(rmgMol,
                                        removeHs,
                                        sanitize))

    @classmethod
    def FromXYZ(cls,
                xyz: str,
                backend: str = 'openbabel',
                header: bool = True,
                correctCO: bool = True,
                **kwargs):
        """
        Convert xyz string to RDKitMol.

        Args:
            xyz (str): A XYZ String.
            backend (str): The backend used to perceive molecule. Defaults to ``openbabel``.
                           Currently, we only support ``openbabel`` and ``jensen``.
            header (bool, optional): If lines of the number of atoms and title are included.
                                     Defaults to ``True.``
            supported kwargs:
                jensen:
                    - charge: The charge of the species. Defaults to ``0``.
                    - allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to False.
                    - use_graph: ``True`` to use networkx module for accelerate. Defaults to True.
                    - use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to False.
                    - embed_chiral: ``True`` to embed chiral information. Defaults to True.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the xyz.
        """
        if not header:
            xyz = f"{len(xyz.splitlines())}\n\n{xyz}"

        # Openbabel support read xyz and perceive atom connectivities
        if backend.lower() == 'openbabel':
            obmol = parse_xyz_by_openbabel(xyz, correct_CO=correctCO)
            return cls.FromOBMol(obmol)

        # https://github.com/jensengroup/xyz2mol/blob/master/xyz2mol.py
        # provides an approach to convert xyz to mol
        elif backend.lower() == 'jensen':
            mol = parse_xyz_by_jensen(xyz, correct_CO=correctCO, **kwargs)
            return cls(mol)

        else:
            raise NotImplementedError(f'Backend ({backend}) is not supported. Only `openbabel` and `jensen`'
                                      f' are supported.')

    def GetAtomicNumbers(self):
        """
        Get the Atomic numbers of the molecules. The atomic numbers are sorted by the atom indexes.

        Returns:
            list: A list of atomic numbers.
        """
        return [atom.GetAtomicNum() for atom in self.GetAtoms()]

    def GetBestAlign(self,
                     refMol,
                     prbCid: int = 0,
                     refCid: int = 0,
                     atomMap: list = [],
                     maxIters: int = 1000,
                     keepBestConformer: bool = True):
        """
        This is a wrapper function for calling `AlignMol` twice, with ``reflect`` to ``True``
        and ``False``, respectively.

        Args:
            refMol (Mol): RDKit molecule as a reference.
            prbCid (int, optional): The conformer id to be aligned. Defaults to ``0``.
            refCid (int, optional): The id of reference conformer. Defaults to ``0``.
            reflect (bool, optional): Whether to reflect the conformation of the probe molecule.
                                      Defaults to ``False``.
            atomMap (list, optional): a vector of pairs of atom IDs ``(probe AtomId, ref AtomId)``
                                      used to compute the alignments. If this mapping is not
                                      specified an attempt is made to generate on by substructure matching.
            maxIters (int, optional): maximum number of iterations used in mimizing the RMSD. Defaults to ``1000``.
            keepBestConformer (bool, optional): Whether to keep the best Conformer structure. Defaults to ``True``.
                                                This is less helpful when you are comparing different atom mappings.

        Returns:
            float: RMSD value.
            bool: if reflected conformer gives a better result.
        """
        # Align without refection
        rmsd = self.AlignMol(refMol=refMol,
                             prbCid=prbCid,
                             refCid=refCid,
                             atomMap=atomMap,
                             maxIters=maxIters)

        # Align with refection
        rmsd_r = self.AlignMol(refMol=refMol,
                               prbCid=prbCid,
                               refCid=refCid,
                               atomMap=atomMap,
                               maxIters=maxIters,
                               reflect=True)

        # The conformation is reflected, now reflect back
        self.Reflect(id=prbCid)

        reflect = True if rmsd > rmsd_r else False

        # Make sure the resulted conformer is the one with the lowest RMSD
        if keepBestConformer:
            rmsd = self.AlignMol(refMol=refMol,
                                 prbCid=prbCid,
                                 refCid=refCid,
                                 atomMap=atomMap,
                                 maxIters=maxIters,
                                 reflect=reflect,)
        else:
            rmsd = rmsd if rmsd <= rmsd_r else rmsd_r

        return rmsd, reflect

    def GetBondsAsTuples(self):
        """
        Generate a list of length-2 sets indicating the bonding atoms
        in the molecule
        """
        return [tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in self.GetBonds()]

    def GetElementSymbols(self):
        """
        Get the element symbols of the molecules. The element symbols are sorted by the atom indexes.

        Returns:
            list: A list of element symbols.
        """
        return get_element_symbols(self.GetAtomicNumbers())

    def GetAtomMasses(self):
        """
        Get the mass of each atom. The order is consistent with the atom indexes.

        Returns:
            list: A list of atom masses.
        """
        return get_atom_masses(self.GetAtomicNumbers())

    def GetConformer(self,
                     id: int = 0) -> 'RDKitConf':
        """
        Get the embedded conformer according to ID.

        Args:
            id (int): The ID of the conformer to be obtained. The default is ``0``.

        Raises:
            ValueError: Bad id assigned.

        Returns:
            RDKitConf: A conformer corresponding to the ID.
        """
        try:
            conformer = self._mol.GetConformer(id)
        except ValueError as e:
            raise ValueError(f"{e}: {id}")
        rdkitconf = RDKitConf(conformer)
        rdkitconf.SetOwningMol(self)
        return rdkitconf

    def GetConformers(self,
                      ids: Union[list, tuple] = [0],
                      ) -> List['RDKitConf']:
        """
        Get the embedded conformers according to IDs.

        Args:
            ids (Union[list, tuple]): The ids of the conformer to be obtained.
                                      The default is ``[0]``.

        Raises:
            ValueError: Bad id assigned.

        Returns:
            List[RDKitConf]: A list of conformers corresponding to the IDs.
        """
        conformers = list()
        for id in ids:
            try:
                conformer = self.GetConformer(id)
            except ValueError as e:
                raise
            else:
                conformers.append(conformer)
        return conformers

    def GetAllConformers(self) -> List['RDKitConf']:
        """
        Get all of the embedded conformers.

        Returns:
            List['RDKitConf']: A list all of conformers.
        """
        return self.GetConformers(list(range(self.GetNumConformers())))

    def GetDistanceMatrix(self, id: int = 0) -> np.ndarray:
        return Chem.rdmolops.Get3DDistanceMatrix(self._mol, confId=id)

    def GetPositions(self, id: int = 0) -> np.ndarray:
        """
        Get atom positions of the embeded conformer.

        Args:
            id (int, optional): The conformer ID to extract atom positions from.
                                Defaults to ``0``.

        Returns:
            np.ndarray: a 3 x N matrix containing atom coordinates.
        """
        conf = self.GetConformer(id=id)
        return conf.GetPositions()

    def GetSymmSSSR(self):
        """
        Get a symmetrized SSSR for a molecule.

        Returns:
            tuple: a sequence of sequences containing the rings found as atom ids
        """
        return Chem.GetSymmSSSR(self._mol)

    def GetSubstructMatch(self,
                          query: Union['RDKitMol', 'RWMol', 'Mol'],
                          useChirality: bool = False,
                          useQueryQueryMatches: bool = False
                          ) -> tuple:
        """
        Returns the indices of the molecule's atoms that match a substructure query.

        Args:
            query (Mol): a RDkit Molecule.
            useChirality (bool, optional): enables the use of stereochemistry in the matching. Defaults to ``False``.
            useQueryQueryMatches (bool, optional): use query-query matching logic. Defaults to ``False``.

        Returns:
            tuple: A tuple of matched indices.
        """
        try:
            return self._mol.GetSubstructMatch(query.ToRWMol(), useChirality, useQueryQueryMatches)
        except AttributeError:
            return self._mol.GetSubstructMatch(query, useChirality, useQueryQueryMatches)

    def GetSubstructMatches(self,
                            query: Union['RDKitMol', 'RWMol', 'Mol'],
                            uniquify: bool = True,
                            useChirality: bool = False,
                            useQueryQueryMatches: bool = False,
                            maxMatches: int = 1000,
                            ) -> tuple:
        """
        Returns tuples of the indices of the molecule's atoms that match a substructure query.

        Args:
            query (Mol): a Molecule.
            uniquify (bool, optional): determines whether or not the matches are uniquified. Defaults to True.
            useChirality (bool, optional): enables the use of stereochemistry in the matching. Defaults to False.
            useQueryQueryMatches (bool, optional): use query-query matching logic. Defaults to False.
            maxMatches: The maximum number of matches that will be returned to prevent a combinatorial explosion.

        Returns:
            tuple: A tuple of tuples of matched indices.
        """
        try:
            return self._mol.GetSubstructMatches(query.ToRWMol(), uniquify, useChirality,
                                                 useQueryQueryMatches, maxMatches)
        except AttributeError:
            return self._mol.GetSubstructMatches(query, uniquify, useChirality,
                                                 useQueryQueryMatches, maxMatches)

    def GetMolFrags(self,
                    asMols: bool = False,
                    sanitize: bool = True,
                    frags: Optional[list] = None,
                    fragsMolAtomMapping: Optional[list] = None,
                    ) -> tuple:
        """
        Finds the disconnected fragments from a molecule. For example, for the molecule ‘CC(=O)[O-].[NH3+]C’,
        this function will split the molecules into a list of ‘CC(=O)[O-]' and '[NH3+]C'. By defaults,
        this function will return a list of atom mapping, but options are available for getting mols.

        Args:
            asMols (bool, optional): Whether the fragments will be returned as molecules instead of atom ids.
                                     Defaults to ``True``.
            sanitize (bool, optional): Whether the fragments molecules will be sanitized before returning them.
                                       Defaults to ``True``.
            frags (list, optional): If this is provided as an empty list, the result will be mol.GetNumAtoms()
                                    long on return and will contain the fragment assignment for each Atom.
            fragsMolAtomMapping (list, optional): If this is provided as an empty list, the result will be a
                                                  numFrags long list on return, and each entry will contain the
                                                  indices of the Atoms in that fragment: [(0, 1, 2, 3), (4, 5)].values

        Returns:
            tuple: a tuple of atom mapping or a tuple of split molecules (RDKitMol).
        """
        frags = Chem.rdmolops.GetMolFrags(self._mol, asMols=asMols, sanitizeFrags=sanitize,
                                          frags=frags, fragsMolAtomMapping=fragsMolAtomMapping)
        if asMols:
            return tuple(RDKitMol(mol) for mol in frags)
        return frags

    def GetTorsionalModes(self,
                          excludeMethyl: bool = False,
                          ) -> list:
        """
        Get all of the torsional modes (rotors) from the molecule.

        Args:
            excludeMethyl (bool): Whether exclude the torsions with methyl groups. Defaults to ``False``.

        Returns:
            list: A list of four-atom-indice to indicating the torsional modes.
        """
        return find_internal_torsions(self._mol,
                                      exclude_methyl=excludeMethyl)

    def GetVdwMatrix(self, threshold=0.4) -> Optional[np.ndarray]:
        """
        Get the derived Van der Waals matrix, which can be used to analyze
        the collision of atoms. More information can be found from ``generate_vdw_mat``.

        Args:
            threshold: float indicating the threshold to use in the vdw matrix

        Returns:
            Optional[np.ndarray]: A 2D array of the derived Van der Waals Matrix, if the
                                  the matrix exists, otherwise ``None``.
        """
        try:
            return self._vdw_mat
        except AttributeError:
            self.SetVdwMatrix(threshold=threshold)
            return self._vdw_mat

    def HasCollidingAtoms(self, threshold=0.4) -> np.ndarray:
        """
        Args:
            threshold: float indicating the threshold to use in the vdw matrix
        """

        dist_mat = np.triu(self.GetDistanceMatrix())
        # if the distance is smaller than a threshold, the atom has a high chance of colliding
        return not np.all(self.GetVdwMatrix(threshold=threshold) <= dist_mat)

    def PrepareOutputMol(self,
                          removeHs: bool = False,
                          sanitize: bool = True,
                          ) -> Mol:
        """
        Generate a RDKit Mol instance for output purpose, to ensure that the original molecule is not modified.

        Args:
            removeHs (bool, optional): Remove less useful explicity H atoms. E.g., When output SMILES, H atoms,
                if explicitly added, will be included and reduce the readablity. Defaults to ``False``.
                Note, following Hs are not removed:

                    1. H which aren’t connected to a heavy atom. E.g.,[H][H].
                    2. Labelled H. E.g., atoms with atomic number=1, but isotope > 1.
                    3. Two coordinate Hs. E.g., central H in C[H-]C.
                    4. Hs connected to dummy atoms
                    5. Hs that are part of the definition of double bond Stereochemistry.
                    6. Hs that are not connected to anything else.

            sanitize (bool, optional): whether to sanitize the molecule. Defaults to ``True``.

        Returns:
            Mol: A Mol instance used for output purpose.
        """
        if removeHs:
            mol = self.RemoveHs(sanitize=sanitize)
        elif sanitize:
            mol = self.GetMol()
            Chem.rdmolops.SanitizeMol(mol)  # mol is modified in place
        return mol

    def RemoveHs(self,
                 sanitize: bool=True):
        """
        Remove H atoms. Useful when trying to match heavy atoms.py

        Args:
            sanitize (bool, optional): Whether to sanitize the molecule. Defaults to ``True``.
        """
        return Chem.rdmolops.RemoveHs(self._mol, sanitize=sanitize)

    def RenumberAtoms(self,
                      newOrder: list,
                      updateAtomMap: bool = True,
                      )-> 'RDKitMol':
        """
        Return a new copy of RDKitMol that has atom (index) reordered.

        Args:
            newOrder (list): the new ordering the atoms (should be numAtoms long). E.g,
                if newOrder is [3,2,0,1], then atom 3 in the original molecule
                will be atom 0 in the new one.
            updateAtomMap (bool): Whether to update the atom map number based on the
                                  new order.

        Returns:
            RDKitMol: Molecule with reordered atoms.
        """
        rwmol = Chem.rdmolops.RenumberAtoms(self._mol, newOrder)
        # Correct the AtomMapNum
        if updateAtomMap:
            [rwmol.GetAtomWithIdx(i).SetAtomMapNum(i + 1) for i in range(rwmol.GetNumAtoms())]
        return RDKitMol(rwmol)

    def Sanitize(self):
        """
        Sanitize the molecule.
        """
        Chem.rdmolops.SanitizeMol(self._mol)

    def SetAtomMapNumbers(self,
                          atomMap: Optional[Sequence[int]] = None,):
        """
        Set the atom mapping number. By defaults, atom indexes are used. It can be helpful
        when plotting the molecule in a 2D graph.

        Args:
            atomMap(list, tuple, optional): A sequence of integers for atom mapping.
        """
        num_atoms = self.GetNumAtoms()
        if atomMap is not None:
            if len(atomMap) != num_atoms:
                raise ValueError('Invalid atomMap provided. It should have the same length as atom numbers.')
        else:
            # Set a atom map numbers based on the order of atom index
            # As suggested by the developer of RDKit
            # It is better to start atom map number from 1
            atomMap = list(range(1, num_atoms + 1))

        for idx in range(num_atoms):
            atom = self.GetAtomWithIdx(idx)
            atom.SetAtomMapNum(atomMap[idx])

    def GetAtomMapNumbers(self) -> tuple:
        """
        Get the atom mapping.

        Returns:
            tuple: atom mapping numbers in the sequence of atom index.
        """
        return tuple(atom.GetAtomMapNum() for atom in self.GetAtoms())

    def Reflect(self, id: int = 0):
        """
        Reflect the atom coordinates of a molecule, and therefore its mirror image.

        Args:
            id (int, optional): The conformer id to reflect.
        """
        self.AlignMol(refMol=self,
                      prbCid=id,
                      maxIters=0,
                      reflect=True)

    def SetPositions(self,
                     coords: Union[Sequence, str],
                     id: int = 0,
                     header: bool = False):
        """
        Set the atom positions to one of the conformer.

        Args:
            coords (sequence): A tuple/list/ndarray containing atom positions;
                               or a string with the typical XYZ formating.
            id (int, optional): Conformer ID to assign the Positions to. Defaults to ``1``.
            header (bool): When the XYZ string has an header. Defaults to ``False``.
        """
        if isinstance(coords, str):
            xyz_lines = coords.splitlines()[2:] if header else coords.splitlines()
            coords = np.array([[float(atom) for atom in line.strip().split()[1:]] for line in xyz_lines])

        try:
            conf = self.GetConformer(id=id)
        except ValueError as e:
            if id == 0:
                try:
                    self.EmbedConformer()
                except RuntimeError:
                    self.EmbedNullConformer()
                conf = self.GetConformer()
            else:
                raise
        conf.SetPositions(np.array(coords, dtype=float))

    def ToOBMol(self) -> 'openbabel.OBMol':
        """
        Convert RDKitMol to a OBMol.

        Returns:
            OBMol: The corresponding openbabel OBMol.
        """
        return rdkit_mol_to_openbabel_mol(self)

    def ToRWMol(self) -> RWMol:
        """
        Convert the RDKitMol Molecule back to a RDKit Chem.rdchem.RWMol.

        returns:
            RWMol: A RDKit Chem.rdchem.RWMol molecule.
        """
        return self._mol

    def ToSDFFile(self, path: str):
        """
        Write molecule information to .sdf file.

        Args:
            path (str): The path to save the .sdf file.
        """
        writer = Chem.rdmolfiles.SDWriter(path)
        # Not sure what may cause exceptions and errors here
        # If any issues found, add try...except...finally
        writer.write(self._mol)
        writer.close()

    def ToSmiles(self,
                 stereo: bool = True,
                 kekule: bool = False,
                 canonical: bool = True,
                 removeAtomMap: bool = True,
                 removeHs: bool = True,
                 ) -> str:
        """
        Convert RDKitMol to a SMILES string.

        Args:
            stereo (bool, optional): Whether keep stereochemistry information. Defaults to ``True``.
            kekule (bool, optional): Whether use Kekule form. Defaults to ``False``.
            canonical (bool, optional): Whether generate a canonical SMILES. Defaults to ``True``.
            removeAtomMap (bool, optional): Whether to remove map id information in the SMILES. Defaults to ``True``.
            removeHs (bool, optional): Whether to remove H atoms to make obtained SMILES clean. Defaults to ``True``.

        Returns:
            str: The smiles string of the molecule.
        """
        mol = self.PrepareOutputMol(removeHs=removeHs, sanitize=True)

        # Remove atom map numbers, otherwise the smiles string is long and non-readable
        if removeAtomMap:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

        return Chem.rdmolfiles.MolToSmiles(mol,
                                           isomericSmiles=stereo,
                                           kekuleSmiles=kekule,
                                           canonical=canonical)

    def ToXYZ(self,
              confId: int = -1,
              header: bool = True,
              ) -> str:
        """
        Convert RDKitMol to a xyz string.

        Args:
            confId (int): The conformer ID to be exported.
            header (bool, optional): Whether to include header (first two lines).
                                     Defaults to ``True``.

        Returns:
            str: The xyz of the molecule.
        """
        xyz = Chem.MolToXYZBlock(self._mol, confId)
        if not header:
            xyz = '\n'.join(xyz.splitlines()[2:])
        return xyz

    def ToMolBlock(self,
                   confId: int = -1,
                   ) -> str:
        """
        Convert RDKitMol to a mol block string.

        Args:
            confId (int): The conformer ID to be exported.

        Returns:
            str: The mol block of the molecule.
        """
        return Chem.MolToMolBlock(self._mol, confId=confId)

    def GetFormalCharge(self):
        """
        Get formal charge of the molecule.

        Returns:
            int : Formal charge.
        """
        return Chem.GetFormalCharge(self._mol)

    def GetInternalCoordinates(self,
                               nonredundant: bool = True,
                               ) -> list:
        """
        Get internal coordinates of the molecule.
        """
        bonds, angles, torsions = get_internal_coords(self.ToOBMol(),
                                                      nonredundant=nonredundant)
        return [[[element - 1 for element in item]
                  for item in ic]
                  for ic in [bonds, angles, torsions]]

    def GetSpinMultiplicity(self):
        """
        Get spin multiplicity of a molecule. The spin multiplicity is calculated
        using Hund's rule of maximum multiplicity defined as 2S + 1.

        Returns:
            int : Spin multiplicity.
        """
        num_radical_elec = 0
        for atom in self.GetAtoms():
            num_radical_elec += atom.GetNumRadicalElectrons()
        return num_radical_elec + 1

    def SaturateBiradicalSites12(self,
                                 multiplicity: int,
                                 verbose: bool = True):
        """
        A method help to saturate 1,2 biradicals to match the given
        molecule spin multiplicity. E.g.,
            *C - C* => C = C
        In the current implementation, no error will be raised,
        if the function doesn't achieve the goal. This function has not been
        been tested on nitrogenate.

        Args:
            multiplicity (int): The target multiplicity.
            verbose (bool): Whether to print additional information. Defaults to ``True``.
        """
        cur_multiplicity = self.GetSpinMultiplicity()
        if  cur_multiplicity == multiplicity:
            # No need to fix
            return
        elif cur_multiplicity < multiplicity:
            if verbose:
                print('It is not possible to match the multiplicity '
                      'by saturating 1,2 biradical sites.')
            return
        elif (cur_multiplicity - multiplicity) % 2:
            if verbose:
                print('It is not possible to match the multiplicity '
                      'by saturating 1,2 biradical sites.')
            return

        num_dbs = (cur_multiplicity - multiplicity) / 2

        # Find all radical sites and save in `rad_atoms` and `rad_atom_elec_nums`
        rad_atoms = []
        for atom in self.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                rad_atoms.append(atom.GetIdx())

        # a list to record connected atom pairs to avoid repeating
        connected = [(i, j) for i, j in combinations(rad_atoms, 2)
                     if self.GetBondBetweenAtoms(i, j)]

        # Naive way to saturate 1,2 biradicals
        while num_dbs and connected:

            for i, j in connected:
                bond = self.GetBondBetweenAtoms(i, j)

                # Find the correct new bond type
                new_bond_type = ORDERS.get(bond.GetBondTypeAsDouble() + 1)
                if not new_bond_type:
                    # Although a bond is found, cannot decide what bond to make between them, pass
                    # This may cause insuccess for aromatic molecules
                    # TODO: Test the adaptability for aromatic molecules
                    continue
                bond.SetBondType(new_bond_type)

                # Modify radical site properties
                for atom in [self.GetAtomWithIdx(j), self.GetAtomWithIdx(i)]:
                    # Decrease radical electron by 1
                    atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)
                    # remove it from the list if it is no longer a radical
                    if atom.GetNumRadicalElectrons() == 0:
                        rad_atoms.remove(atom.GetIdx())
                break
            else:
                # if no update is made at all, break
                # TODO: Change this to a log in the future
                break
            num_dbs -= 1
            connected = [pair for pair in connected
                         if pair[0] in rad_atoms and pair[1] in rad_atoms]

        # Update things including explicity / implicit valence, etc.
        self.UpdatePropertyCache(strict=False)

        if num_dbs:
            if verbose:
                print('Cannot match the multiplicity by saturating 1,2 biradical.')

    def SaturateBiradicalSitesCDB(self,
                                  multiplicity: int,
                                  chain_length: int = 8,
                                  verbose: bool = True):
        """
        A method help to saturate biradicals that have conjugated double bond in between
        to match the given molecule spin multiplicity. E.g, 1,4 biradicals can be saturated
        if there is a unsaturated bond between them:
            *C - C = C - C* => C = C - C = C
        In the current implementation, no error will be raised,
        if the function doesn't achieve the goal. This function has not been
        been tested on nitrogenate.

        Args:
            multiplicity (int): The target multiplicity.
            chain_length (int): How long the conjugated double bond chain is.
                                A larger value will result in longer time. defaults to 8.
            verbose (int): Whether to print additional information. Defaults to ``True``.
        """
        cur_multiplicity = self.GetSpinMultiplicity()
        if  cur_multiplicity == multiplicity:
            # No need to fix
            return
        elif cur_multiplicity < multiplicity:
            if verbose:
                print('It is not possible to match the multiplicity '
                    'by saturating conjugated biradical sites.')
            return
        elif (cur_multiplicity - multiplicity) % 2:
            if verbose:
                print('It is not possible to match the multiplicity '
                    'by saturating conjugatged biradical sites.')
            return

        num_dbs = (cur_multiplicity - multiplicity) / 2

        # Find all radical sites and save in `rad_atoms` and `rad_atom_elec_nums`
        rad_atoms = []
        for atom in self.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                rad_atoms.append(atom.GetIdx())

        # prepruning by atom numbers
        chain_length = min(self.GetNumAtoms(), chain_length)
        # Find all paths satisfy *C -[- C = C -]n- C*
        # n = 1, 2, 3 is corresponding to chain length = 4, 6, 8
        for path_length in range(4, chain_length+1, 2):

            if not num_dbs:
                # problem solved in the previous run
                break

            all_paths = [list(p) for p in \
                         list(Chem.rdmolops.FindAllPathsOfLengthN(self._mol,
                                                                  path_length,
                                                                  useBonds=False))]
            if not all_paths:
                # empty all_paths means cannot find chains such long
                break

            paths = []
            for path in all_paths:
                if path[0] in rad_atoms and path[-1] in rad_atoms:
                    for sec in range(1, path_length-2, 2):
                        bond = self.GetBondBetweenAtoms(path[sec],
                                                        path[sec+1])
                        if bond.GetBondTypeAsDouble() not in [2, 3]:
                            break
                    else:
                        paths.append(path)

            while num_dbs and paths:
                for path in paths:
                    bonds = [self.GetBondBetweenAtoms(*path[i:i+2])
                             for i in range(path_length-1)]

                    new_bond_types = [ORDERS.get(bond.GetBondTypeAsDouble() + (-1) ** i)
                                    for i, bond in enumerate(bonds)]

                    if any([bond_type is None for bond_type in new_bond_types]):
                        # Although a match is found, cannot decide what bond to make, pass
                        continue

                    for bond, bond_type in zip(bonds, new_bond_types):
                        bond.SetBondType(bond_type)

                    # Modify radical site properties
                    for atom in [self.GetAtomWithIdx(path[0]), self.GetAtomWithIdx(path[-1])]:
                        # Decrease radical electron by 1
                        atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)
                        # remove it from the list if it is no longer a radical
                        if atom.GetNumRadicalElectrons() == 0:
                            rad_atoms.remove(atom.GetIdx())

                    num_dbs -= 1
                    break
                else:
                    # if no update is made at all, break
                    # TODO: Change this to a log in the future
                    break
                paths = [path for path in paths
                        if path[0] in rad_atoms and path[-1] in rad_atoms]

        # Update things including explicity / implicit valence, etc.
            self.UpdatePropertyCache(strict=False)

        if num_dbs:
            if verbose:
                print('Cannot match the multiplicity by saturating biradical with conjugated double bonds.')

    def SaturateCarbene(self,
                        multiplicity: int,
                        verbose: bool = True):
        """
        A method help to saturate carbenes and nitrenes to match the given
        molecule spin multiplicity:
            *-C-* (triplet) => C-(**) (singlet)
        In the current implementation, no error will be raised,
        if the function doesn't achieve the goal. This function has not been
        been tested on nitrogenate.

        Args:
            multiplicity (int): The target multiplicity.
            verbose (int): Whether to print additional information. Defaults to ``True``.
        """
        cur_multiplicity = self.GetSpinMultiplicity()
        if  cur_multiplicity == multiplicity:
            # No need to fix
            return
        elif cur_multiplicity < multiplicity:
            if verbose:
                print('It is not possible to match the multiplicity '
                    'by saturating carbene sites.')
            return
        elif (cur_multiplicity - multiplicity) % 2:
            if verbose:
                print('It is not possible to match the multiplicity '
                    'by saturating carbene sites.')
            return
        elec_to_pair = cur_multiplicity - multiplicity

        carbene_atoms = self.GetSubstructMatches(CARBENE_PATTERN)  # Import from rdmc.utils
        if not carbene_atoms:
            if verbose:
                print('There is no carbene site in the molecule.')
            return

        for atom in carbene_atoms:
            atom = self.GetAtomWithIdx(atom[0])
            while atom.GetNumRadicalElectrons() >=2 and elec_to_pair:
                atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 2)
                elec_to_pair -= 2
            if not elec_to_pair:
                break

        if elec_to_pair:
            if verbose:
                print('Cannot match the multiplicity by saturating carbene(-like) atoms')

    def SaturateMol(self, multiplicity: int,
                    chain_length: int = 8,
                    verbose: bool = False):
        """
        A method help to saturate the molecule to match the given
        molecule spin multiplicity. This is just a wrapper to call both
        `SaturateBiradicalSites12`, `SaturateBiradicalSitesCDB`, and
        `SaturateCarbene`:
            *C - C* => C = C
            *C - C = C - C* => C = C - C = C
            *-C-* (triplet) => C-(**) (singlet)
        In the current implementation, no error will be raised,
        if the function doesn't achieve the goal. This function has not been
        been tested on nitrogenate.

        Args:
            multiplicity (int): The target multiplicity.
            chain_length (int): How long the conjugated double bond chain is.
                                A larger value will result in longer time.
            verbose (bool): Whether to print intermediate information.
                            Defaults to ``False``.
        """
        self.SaturateBiradicalSites12(multiplicity=multiplicity,
                                      verbose=verbose)
        self.SaturateBiradicalSitesCDB(multiplicity=multiplicity,
                                       chain_length=chain_length,
                                       verbose=verbose)
        self.SaturateCarbene(multiplicity=multiplicity, verbose=verbose)
        if self.GetSpinMultiplicity() != multiplicity:
            print('SaturateMol fails after trying all methods and you need to be cautious about the generated mol.')

    def SetVdwMatrix(self,
                     threshold: float = 0.4,
                     vdw_radii: dict = VDW_RADII):
        """
        Set the derived Van der Waals matrix, which is an upper triangle matrix
        calculated from a threshold usually around 0.4 of the Van der Waals Radii.
        Its diagonal elements are all zeros. The element (i, j) is calculated by
        threshold * sum( R(atom i) + R(atom j) ). If two atoms are bonded, the value is
        set to be zero. When threshold = 0.4, the value is close to the covalent bond
        length.

        Args:
            threshold (float): The threshold used to calculate the derived Van der Waals
                               matrix. A larger value results in a matrix with larger values;
                               When compared with distance matrix, it may overestiate the
                               overlapping between atoms. The default value is 0.4.
            vdw_radii (dict): A dict stores the Van der Waals radii of different elements.

        Raises:
            ValueError: Invalid threshold is supplied.
        """
        self._vdw_mat = generate_vdw_mat(self, threshold, vdw_radii)


def parse_xyz_or_smiles_list(mol_list,
                             with_3d_info: bool = False,
                             **kwargs):
    """
    A helper function to parse xyz and smiles and list if the
    conformational information is provided.

    Args:
        mol_list (list): a list of smiles or xyzs or tuples of (string, multiplicity)
                  to specify desired multiplicity.
                  E.g., ['CCC', 'H 0 0 0', ('[CH2]', 1)]
        with_3d_info (bool): Whether to indicate which entries are from 3D representations.
                             Defaults to False.

    """
    mols, is_3D = [], []
    for mol in mol_list:
        if isinstance(mol, (tuple, list)) and len(mol) == 2:
            mol, mult = mol
        else:
            mult = None
        try:
            rd_mol = RDKitMol.FromXYZ(mol, **kwargs)
        except ValueError:
            rd_mol = RDKitMol.FromSmiles(mol,)
            rd_mol.EmbedConformer()
            is_3D.append(False)
        else:
            is_3D.append(True)
        finally:
            if mult != None:
                rd_mol.SaturateMol(multiplicity=mult)
            mols.append(rd_mol)
    if with_3d_info:
        return mols, is_3D
    else:
        return mols

def generate_vdw_mat(rd_mol,
                     threshold: float = 0.4,
                     vdw_radii: dict = VDW_RADII):
    """
    Generate a derived Van der Waals matrix, which is an upper triangle matrix
    calculated from a threshold usually around 0.4 of the Van der Waals Radii.
    Its diagonal elements are all zeros. The element (i, j) is calculated by
    threshold * sum( R(atom i) + R(atom j) ). If two atoms are bonded, the value is
    set to be zero. When threshold = 0.4, the value is close to the covalent bond
    length.

    Args:
        threshold (float): The threshold used to calculate the derived Van der Waals
                            matrix. A larger value results in a matrix with larger values;
                            When compared with distance matrix, it may overestiate the
                            overlapping between atoms. The default value is 0.4.
        vdw_radii (dict): A dict stores the Van der Waals radii of different elements.

    Raises:
        ValueError: Invalid threshold is supplied.
    """
    if threshold <= 0:
        raise ValueError("The provided threshold is invalid.")

    # initialize a vdw matrix
    num_atom = rd_mol.GetNumAtoms()
    vdw_mat = np.zeros((num_atom, num_atom))
    # get all of the atom index
    atom_idx_list = range(num_atom)

    for atom1_ind in atom_idx_list:
        atom1 = rd_mol.GetAtomWithIdx(atom1_ind)
        bonded_atom_number = [nb.GetIdx() for nb in atom1.GetNeighbors()]
        for atom2_ind in atom_idx_list[atom1_ind + 1:]:
            if atom2_ind in bonded_atom_number:
                # set a small value for bonded atoms, so they won't arise the collision detector
                vdw_mat[atom1_ind, atom2_ind] = 0.
            else:
                atom2 = rd_mol.GetAtomWithIdx(atom2_ind)
                vdw_mat[atom1_ind, atom2_ind] = threshold * \
                    (vdw_radii[atom1.GetAtomicNum()] +
                        vdw_radii[atom2.GetAtomicNum()])
    return vdw_mat
