#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit RWMol, Mol.
"""

from typing import List, Optional, Sequence, Union

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
        if isinstance(mol, Mol):
            self._mol = RWMol(mol)  # Make the original Mol a writable object
        elif isinstance(mol, RWMol):
            self._mol = mol
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

        # Set atom map number
        if keepAtomMap:
            if not np.any(self.GetAtomMapNumbers()):
                # No stored atom mapping, so set it anyway
                self.SetAtomMapNumbers()
        else:
            self.SetAtomMapNumbers()

        # Perceive rings
        self.GetSymmSSSR()

    def AlignMol(self,
                 refMol: Union['RDKitMol', 'RWMol', 'Mol'],
                 prbCid: int = 0,
                 refCid: int = 0,
                 reflect: bool = False,
                 atomMap: list = [],
                 maxIters: int = 1000,
                 ) -> float:
        """
        Align molecules based on a reference molecule.

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

        Returns:
            float: RMSD value.
        """
        try:
            ref_mol = refMol.ToRWMol()
        except AttributeError:
            ref_mol = refMol
        return Chem.rdMolAlign.AlignMol(prbMol=self._mol,
                                        refMol=ref_mol,
                                        prbCid=prbCid,
                                        refCid=refCid,
                                        atomMap=atomMap,
                                        reflect=reflect,
                                        maxIters=maxIters)

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
        return self.RenumberAtoms(list(range(self.GetNumAtoms())))

    def EmbedConformer(self):
        """
        Embed a conformer to the ``RDKitMol``. This will overwrite current conformers.
        """
        AllChem.EmbedMolecule(self._mol)

    def EmbedMultipleConfs(self, n: int = 1):
        """
        Embed conformers to the ``RDKitMol``. This will overwrite current conformers.

        Args:
            n (int): The number of conformers to be embedded. The default is ``1``.
        """
        AllChem.EmbedMultipleConfs(self._mol, numConfs=n)

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
                ) -> 'RDKitMol':
        """
        Convert a RDKit ``Chem.rdchem.Mol`` molecule to ``RDKitMol`` Molecule.

        Args:
            rdmol (Union[Mol, RWMol]): The RDKit ``Chem.rdchem.Mol`` / ``RWMol`` molecule to be converted.

        Returns:
            RDKitMol: An RDKitMol molecule.
        """
        return cls(mol)

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
        return cls(mol, keepAtomMap=keepAtomMap)

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

    def GetElementSymbols(self):
        """
        Get the element symbols of the molecules. The element symbols are sorted by the atom indexes.

        Returns:
            list: A list of element symbols.
        """
        pt = Chem.GetPeriodicTable()
        return [pt.GetElementSymbol(atom.GetAtomicNum()) for atom in self.GetAtoms()]

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
                      newOrder: list):
        """
        Return a new copy of RDKitMol that has atom reordered.

        Args:
            newOrder (list): the new ordering the atoms (should be numAtoms long). E.g,
                if newOrder is [3,2,0,1], then atom 3 in the original molecule
                will be atom 0 in the new one.

        Returns:
            RDKitMol: Molecule with reordered atoms.
        """
        rwmol = Chem.rdmolops.RenumberAtoms(self._mol, newOrder)
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
        if atomMap:
            if len(atomMap) != num_atoms:
                raise ValueError('Invalid atomMap provided. It should have the same length as atom numbers.')
        else:
            atomMap = list(range(num_atoms))

        for idx in range(num_atoms):
            atom = self.GetAtomWithIdx(idx)
            atom.SetProp('molAtomMapNumber', str(atomMap[idx]))

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
                     coords: Sequence,
                     id: int = 0):
        """
        Set the atom positions to one of the conformer.

        Args:
            coords (sequence): A tuple/list/ndarray containing atom positions.
            id (int, optional): Conformer ID to assign the Positions to.
        """
        try:
            conf = self.GetConformer(id=id)
        except ValueError as e:
            if id == 0:
                self.EmbedConformer()
                conf = self.GetConformer()
            else:
                raise
        conf.SetPositions(coords)

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
                 RemoveAtomMap: bool = True,
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
        if RemoveAtomMap:
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


class RDKitTS(RDKitMol):
    """
    This is a wrapper for Chem.rdchem.Mol / Chem.rdchem.RWMol to deal with Transition States
    specifically. Since transition states are those cannot be sanitized, EmbedConformer method
    is not feasible. Modifications on ``Chem.AllChem.EmbedMolecule()`` and ``Chem.All.EmbedMultipleConfs``
    allows user assign xyz coordinates mannually. Generally, not recommended to use unless you have to do it.
    """

    def EmbedConformer(self):
        """
        Embed a conformer to the RDKitMol. This will overwrite current conformers.
        The coordinates will be initialized to zeros.
        """
        self.EmbedMultipleConfs()

    def EmbedMultipleConfs(self, n: int = 1):
        """
        Embed conformers to the RDKitMol. This will overwrite current conformers.
        All coordinates will be initialized to zeros.

        Args:
            n (int): The number of conformers to be embedded. Defaults to ``1``.
        """
        self._mol.RemoveAllConformers()
        num_atoms = self._mol.GetNumAtoms()
        for i in range(n):
            conf = Conformer()
            set_rdconf_coordinates(conf, np.zeros((num_atoms, 3)))
            conf.SetId(i)
            self._mol.AddConformer(conf)
