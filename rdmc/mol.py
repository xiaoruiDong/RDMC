#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit RWMol, Mol.
"""

from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, RWMol

# openbabel import is currently put behind rdkit.
# This relates to https://github.com/rdkit/rdkit/issues/2292
# For Mac, with anaconda build:
# rdkit -  2020.03.2.0 from rdkit channel
# openbabel -  2.4.1 from rmg channel
# works fine without sequence problem
# For Linux,
# rdkit - 2020.03.3 from rdkit channel
# openbabel - 2.4.1 from rmg channel does not work
import openbabel as ob
import pybel

from rdmc.conf import RDKitConf
from rdmc.utils import *
from rdmc.external.xyz2mol import int_atom, xyz2mol

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

    def __init__(self, mol: Union[Mol, RWMol]):
        """
        Generate an RDKitMol Molecule instance from a RDKit Chem.rdchem.Mol or RWMol molecule.

        Args:
            mol (Union[Mol, RWMol]): The RDKit Chem.rdchem.Mol / RWmol molecule to be converted.
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
        self.SetAtomMapNumbers()

        # Perceive rings
        Chem.GetSymmSSSR(self._mol)

    def AlignMol(self,
                 refMol: Union['RDKitMol', 'RWMol', 'Mol'],
                 confid: int = 0,
                 refCid: int = 0,
                 reflect: bool = False,
                 atomMap: list = [],
                 maxIters: int = 1000,
                 ) -> float:
        """
        Align molecules based on a reference molecule.

        Args:
            refMol (Mol): RDKit molecule as a reference.
            confid (int, optional): The conformer id to be aligned. Defaults to 0.
            refCid (int, optional): The id of reference conformer. Defaults to 0.
            reflect (bool, optional): Whether to reflect the conformation of the probe molecule.
                                      Defaults to False.
            atomMap (list, optional): a vector of pairs of atom IDs (probe AtomId, ref AtomId)
                                      used to compute the alignments. If this mapping is not
                                      specified an attempt is made to generate on by substructure matching.
            maxIters (int, optional): maximum number of iterations used in mimizing the RMSD. Defaults to 1000.
        """
        try:
            return Chem.rdMolAlign.AlignMol(prbMol=self._mol,
                                            refMol=refMol.ToRWMol(),
                                            prbCid=confid,
                                            refCid=refCid,
                                            atomMap=atomMap,
                                            reflect=reflect,
                                            maxIters=maxIters)
        except AttributeError:
            return Chem.rdMolAlign.AlignMol(prbMol=self._mol,
                                            refMol=refMol,
                                            prbCid=confid,
                                            refCid=refCid,
                                            atomMap=atomMap,
                                            reflect=reflect,
                                            maxIters=maxIters)

    def Copy(self) -> 'RDKitMol':
        """
        Make a copy of the RDKitMol.

        Returns:
            RDKitMol: a copied molecule
        """
        return self.RenumberAtoms(list(range(self.GetNumAtoms())))

    def EmbedConformer(self):
        """
        Embed a conformer to the RDKitMol. This will overwrite current conformers.
        """
        AllChem.EmbedMolecule(self._mol)

    def EmbedMultipleConfs(self, n: int = 1):
        """
        Embed conformers to the RDKitMol. This will overwrite current conformers.

        Args:
            n (int): The number of conformers to be embedded. The default is 1.
        """
        AllChem.EmbedMultipleConfs(self._mol, numConfs=n)

    @ classmethod
    def FromOBMol(cls,
                  ob_mol: 'openbabel.OBMol',
                  remove_h: bool = False,
                  sanitize: bool = True,
                  embed: bool = True,
                  ) -> 'RDKitMol':
        """
        Convert a OpenBabel molecular structure to an RDKit RDMol object.

        Args:
            ob_mol (Molecule): An OpenBabel Molecule object for the conversion.
            remove_h (bool, optional): Whether to remove hydrogen atoms from the molecule, Defaults to False.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to True.
            embed (bool, optional): Whether to embeb 3D conformer from OBMol. Defaults to True.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the input OpenBabel Molecule object.
        """
        rw_mol = openbabel_mol_to_rdkit_mol(ob_mol, remove_h, sanitize, embed)
        return cls(rw_mol)

    @classmethod
    def FromMol(cls,
                mol: Union[Mol, RWMol],
                ) -> 'RDKitMol':
        """
        Convert a RDKit Chem.rdchem.Mol molecule to RDKitMol Molecule.

        Args:
            rdmol (Union[Mol, RWMol]): The RDKit Chem.rdchem.Mol / RWMol molecule to be converted.

        Returns:
            RDKitMol: An RDKitMol molecule.
        """
        return cls(mol)

    @classmethod
    def FromSmiles(cls,
                   smiles: str,
                   remove_h: bool = False,
                   sanitize: bool = True,
                   ) -> 'RDKitMol':
        """
        Convert a smiles to an RDkit Mol object.

        Args:
            smiles (str): A SMILES representation of the molecule.
            remove_h (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the SMILES.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not remove_h:
            mol = Chem.AddHs(mol)
        if sanitize:
            Chem.SanitizeMol(mol)
        return cls(mol)

    @classmethod
    def FromRMGMol(cls,
                   rmgmol: 'rmgpy.molecule.Molecule',
                   remove_h: bool = False,
                   sanitize: bool = True,
                   ) -> 'RDKitMol':
        """
        Convert an RMG Molecule to an RDkit Mol object.

        Args:
            smiles (str): An RMG Molecule instance.
            remove_h (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the RMG Molecule.
        """
        return cls(rmg_mol_to_rdkit_mol(rmgmol,
                                        remove_h,
                                        sanitize))

    @classmethod
    def FromXYZ(cls,
                xyz: str,
                backend: str = 'pybel',
                header: bool = True,
                **kwargs):
        """
        Convert xyz string to RDKitMol.

        Args:
            xyz (str): A XYZ String.
            backend (str): The backend used to perceive molecule. Defaults to "pybel".
                           Currently, we only support "pybel" and "jensen".
            header (bool, optional): If lines of the number of atoms and title are included.
                                     Defaults to True.
            supported kwargs:
                jensen: - charge: The charge of the species. Defaults to 0.
                        - allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to False.
                        - use_graph: ``True`` to use networkx module for accelerate. Defaults to True.
                        - use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to False.
                        - embed_chiral: ``True`` to embed chiral information. Defaults to True.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the xyz.
        """
        if not header:
            xyz = f"{len(xyz.splitlines())}\n\n{xyz}"

        # Pybel support read xyz and perceive atom connectivities
        if backend.lower() == 'pybel':
            obmol = pybel.readstring('xyz', xyz).OBMol
            return cls.FromOBMol(obmol)

        # https://github.com/jensengroup/xyz2mol/blob/master/xyz2mol.py
        # provides an approach to convert xyz to mol
        elif backend.lower() == 'jensen':
            atoms, coords = [], []
            for line in xyz.splitlines()[2:]:
                sections = line.split()
                atoms.append(int_atom(sections[0]))
                coords.append([float(i) for i in sections[1:]])
            if 'allow_charged_fragments' not in kwargs:
                kwargs['allow_charged_fragments'] = False
            try:
                mol = xyz2mol(atoms, coords, **kwargs)[0]
            except IndexError:
                raise ValueError(f'Cannot perceive the xyz by the backend ({backend}).')
            else:
                return cls(mol)

        else:
            raise NotImplementedError(f'Backend ({backend}) is not supported. Only `pybel` and `jensen`'
                                      f' are supported.')

    def GetAtomicNumbers(self):
        """
        Get the Atomic numbers of the molecules. The atomic numbers are sorted by the atom indexes.

        Returns:
            list: A list of atomic numbers.
        """
        return [atom.GetAtomicNum() for atom in self.GetAtoms()]

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
        Get the embedded conformer according to ID. The relationship:
                       mol
            RDKitMol ======== RWMol
           owing ||           || owing
             mol ||           || mol
            RDKitConf ====== Conformer
                     conformer
        Args:
            id (int): The ID of the conformer to be obtained. The default is 0.

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
                                      The default is [0].

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

    def GetSubstructMatch(self,
                          query: Union['RDKitMol', 'RWMol', 'Mol'],
                          useChirality: bool = False,
                          useQueryQueryMatches: bool = False
                          ) -> tuple:
        """
        Returns the indices of the molecule's atoms that match a substructure query.

        Args:
            query (Mol): a Molecule.
            useChirality (bool, optional): enables the use of stereochemistry in the matching. Defaults to False.
            useQueryQueryMatches (bool, optional): use query-query matching logic. Defaults to False.

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

    def GetTorsionalModes(self,
                          exclude_methyl: bool = False,
                          ) -> list:
        """
        Get all of the torsional modes (rotors) from the molecule.

        Args:
            exclude_methyl (bool): Whether exclude the torsions with methyl groups.

        Returns:
            list: A list of four-atom-indice to indicating the torsional modes.
        """
        return find_internal_torsions(self._mol,
                                      exclude_methyl=exclude_methyl)

    def PrepareOutputMol(self,
                          remove_h: bool = False,
                          sanitize: bool = True,
                          ) -> Mol:
        """
        Generate a RDKit Mol instance for output purpose, to ensure that the original molecule is not modified.

        Args:
            remove_h (bool, optional): Remove less useful explicity H atoms. E.g., When output SMILES, H atoms,
                      if explicitly added, will be included and reduce the readablity. Note, following Hs are not removed:
                        1. H which aren’t connected to a heavy atom. E.g.,[H][H].
                        2. Labelled H. E.g., atoms with atomic number=1, but isotope > 1.
                        3. Two coordinate Hs. E.g., central H in C[H-]C.
                        4. Hs connected to dummy atoms
                        5. Hs that are part of the definition of double bond Stereochemistry.
                        6. Hs that are not connected to anything else.
                        Defaults to False:
            sanitize (bool, optional): whether to sanitize the molecule. Defaults to True.

        Returns:
            Mol: A Mol instance used for output purpose.
        """
        mol = self.GetMol()
        if remove_h:
            mol = Chem.rdmolops.RemoveHs(mol, sanitize=sanitize)
        elif sanitize:
            Chem.rdmolops.SanitizeMol(mol)  # mol is modified in place
        return mol

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

    def SetAtomMapNumbers(self):
        """
        Set the atom index to atom map number, so atom indexes are shown when plotting the molecule in a 2D graph.
        """
        for ind in range(self.GetNumAtoms()):
            atom = self.GetAtomWithIdx(ind)
            atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))

    def ToOBMol(self):
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
                 mapid: bool = False,
                 remove_h: bool = True,
                 ) -> str:
        """
        Convert RDKitMol to a SMILES string.

        Args:
            stereo (bool, optional): Whether keep stereochemistry information. Defaults to True.
            kekule (bool, optional): Whether use Kekule form. Defaults to False.
            canonical (bool, optional): Whether generate a canonical SMILES. Defaults to True.
            mapid (bool, optional): Whether to keep map id information in the SMILES. Defaults to False.
            remove_h (bool, optional): Whether to remove H atoms to make obtained SMILES clean. Defaults to True.

        Returns:
            str: The smiles string of the molecule.
        """
        mol = self.PrepareOutputMol(remove_h=remove_h, sanitize=True)

        # Remove atom map numbers, otherwise the smiles string is long and non-readable
        if not mapid:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

        return Chem.rdmolfiles.MolToSmiles(mol,
                                           isomericSmiles=stereo,
                                           kekuleSmiles=kekule,
                                           canonical=canonical)

    def ToXYZ(self,
              conf_id: int = -1,
              header: bool = True,
              ) -> str:
        """
        Convert RDKitMol to a SMILES string.

        Args:
            conf_id (int): The conformer ID to be exported.
            header (bool, optional): Whether to include header (first two lines).
                                     Defaults to True.

        Returns
            str: The xyz of the molecule.
        """
        xyz = Chem.MolToXYZBlock(self._mol, conf_id)
        if not header:
            xyz = '\n'.join(xyz.splitlines()[2:])
        return xyz
