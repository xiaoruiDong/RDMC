from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields, rdMolTransforms
from rdmc.mol import RDKitMol
from rdmc.ts import get_broken_bonds, NaiveAlign, get_formed_and_broken_bonds
from rdmc.forcefield import OpenBabelFF
from scipy.optimize import differential_evolution
from rmsd import kabsch_rmsd
import numpy as np
import copy


def mmff_constraints(mol, bonds, confid=0):
    ffps = ChemicalForceFields.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s', mmffVerbosity=0)
    if ffps is None:
        return mol  # TODO: need to replace this
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffps, confId=confid, ignoreInterfragInteractions=True)
    for a1, a2 in bonds:
        ff.MMFFAddDistanceConstraint(a1, a2, True, 0., 0., 1e05)
    res = ff.Minimize(maxIts=200)
    return mol


def reset_pmol(r_mol, p_mol):

    # copy current pmol and set new positions
    p_mol_new = copy.deepcopy(p_mol)
    p_mol_new.SetPositions(r_mol.GetPositions())

    # setup first minimization with broken bond constraints
    obff = OpenBabelFF()
    obff.setup(p_mol_new)
    broken_bonds = get_broken_bonds(r_mol, p_mol)
    r_dmat = r_mol.GetDistanceMatrix()
    current_distances = [r_dmat[b] for b in broken_bonds]
    [obff.add_distance_constraint(b, d) for b, d in zip(broken_bonds, current_distances)];
    obff.optimize(max_step=2000)

    # second minimization without constraints
    obff.constraints = None
    obff.optimize(max_step=2000)
    return obff.get_optimized_mol()


def realistic_mol_prep(mols):
    r_mol, ts_mol, p_mol = mols
    r_rdmc, p_rdmc = RDKitMol.FromMol(r_mol), RDKitMol.FromMol(p_mol)
    if len(r_rdmc.GetMolFrags()) == 2:
        r_rdmc = align_reactant_fragments(r_rdmc, p_rdmc)
    p_mol_new = reset_pmol(r_rdmc, p_rdmc)  # reconfigure pmol as if starting from SMILES
    # p_mol_new = optimize_rotatable_bonds(r_rdmc, p_mol_new)  # optimize rotatable bonds
    return r_rdmc.ToRWMol(), ts_mol, p_mol_new.ToRWMol()


def align_reactant_fragments(r_rdmc, p_rdmc):
    formed_bonds, broken_bonds = get_formed_and_broken_bonds(r_rdmc, p_rdmc)
    naive_align = NaiveAlign.from_complex(r_rdmc, formed_bonds, broken_bonds)
    r_rdmc_naive_align = r_rdmc.Copy()
    r_rdmc_naive_align.SetPositions(naive_align())
    return r_rdmc_naive_align


def optimize_rotatable_bonds(r_mol, p_mol, seed=0, popsize=150, maxiter=500, mutation=(0.5, 1), recombination=0.8):
    # Set optimization function
    opt = optimze_conformation(r_mol=r_mol.ToRWMol(), p_mol=p_mol.ToRWMol(), n_particles=1, seed=seed)

    # Define bounds for optimization
    max_bound = np.concatenate([[np.pi] * 3, [0, 0, 0], [np.pi] * len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi] * 3, [0, 0, 0], [-np.pi] * len(opt.rotable_bonds)], axis=0)
    bounds = (min_bound, max_bound)

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, list(zip(bounds[0], bounds[1])), maxiter=maxiter,
                                    popsize=int(np.ceil(popsize / (len(opt.rotable_bonds) + 6))),
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)
    opt_p_mol = apply_changes(opt.p_mol, result['x'], opt.rotable_bonds)  # might need FF opt

    return opt_p_mol


class optimze_conformation():
    def __init__(self, r_mol, p_mol, n_particles, save_molecules=False, seed=None):
        super(optimze_conformation, self).__init__()
        if seed:
            np.random.seed(seed)

        self.opt_mols = []
        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([p_mol])
        self.save_molecules = save_molecules
        self.r_mol = r_mol
        self.p_mol = p_mol

    def score_conformation(self, values):
        """
        Parameters
        ----------
        values : numpy.ndarray
            set of inputs of shape :code:`(n_particles, dimensions)`
        Returns
        -------
        numpy.ndarray
            computed cost of size :code:`(n_particles, )`
        """
        if len(values.shape) < 2: values = np.expand_dims(values, axis=0)
        p_mols = [copy.copy(self.p_mol) for _ in range(self.n_particles)]

        # Apply changes to molecules
        # apply rotations
        [SetDihedral(p_mols[m].GetConformer(), self.rotable_bonds[r], values[m, 6 + r]) for r in
         range(len(self.rotable_bonds)) for m in range(self.n_particles)]

        # apply transformation matrix
        [rdMolTransforms.TransformConformer(p_mols[m].GetConformer(), GetTransformationMatrix(values[m, :6])) for m in
         range(self.n_particles)]

        # calculate rmsd
        # rmsds_list = [Rotation.align_vectors(m.GetConformer().GetPositions(), self.r_mol.GetConformer().GetPositions())[1] for m in p_mols]
        # rmsds_list = [kabsch_rmsd(self.r_mol.GetConformer().GetPositions(), m.GetConformer().GetPositions(), W=None, translate=True) for m in p_mols]
        rmsds_list = []
        for r in self.rotable_bonds:
            weights = np.zeros([self.p_mol.GetNumAtoms()])
            weights.put(r, 1.)
            rmsds = [kabsch_rmsd(self.r_mol.GetConformer().GetPositions(), m.GetConformer().GetPositions(), W=weights,
                                 translate=True) for m in p_mols]
            rmsds_list.append(rmsds)
        rmsds_array = np.array(rmsds_list).sum(axis=0)

        # save
        if self.save_molecules: self.opt_mols.append(p_mols[np.argmin(rmsds_array)])

        return rmsds_array


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def GetTransformationMatrix(transformations):
    if len(transformations) == 6:
        x, y, z, disp_x, disp_y, disp_z = transformations
    elif len(transformations) == 3:
        x, y, z = transformations
        disp_x, disp_y, disp_z = 0., 0., 0.
    transMat = np.array([[np.cos(z) * np.cos(y), (np.cos(z) * np.sin(y) * np.sin(x)) - (np.sin(z) * np.cos(x)),
                          (np.cos(z) * np.sin(y) * np.cos(x)) + (np.sin(z) * np.sin(x)), disp_x],
                         [np.sin(z) * np.cos(y), (np.sin(z) * np.sin(y) * np.sin(x)) + (np.cos(z) * np.cos(x)),
                          (np.sin(z) * np.sin(y) * np.cos(x)) - (np.cos(z) * np.sin(x)), disp_y],
                         [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x), disp_z],
                         [0, 0, 0, 1]], dtype=np.double)
    return transMat


def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    # if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #     or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #     continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def get_random_conformation(mol, rotable_bonds=None, seed=None):
    if isinstance(mol, Chem.Mol):
        # Check if ligand it has 3D coordinates, otherwise generate them
        try:
            mol.GetConformer()
        except:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
    else:
        raise Exception('mol should be an RDKIT molecule')
    if seed:
        np.random.seed(seed)
    if rotable_bonds is None:
        rotable_bonds = get_torsions([mol])
    new_conf = apply_changes(mol, np.random.rand(len(rotable_bonds, ) + 6) * 10, rotable_bonds)
    Chem.rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())
    return new_conf


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[6 + r]) for r in range(len(rotable_bonds))]

    # apply transformation matrix
    rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol
