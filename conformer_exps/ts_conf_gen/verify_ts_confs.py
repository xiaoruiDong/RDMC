import os
from argparse import ArgumentParser
from rdmc import RDKitMol
from rdmc.conformer_generation.ts_verifiers import XTBFrequencyVerifier, OrcaIRCVerifier

parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--rxn_idx", type=int)


def optimize_conf(args):

    save_dir = os.path.join(args.exp_dir, str(args.rxn_idx))
    ts_file = os.path.join(save_dir, "ts_optimized_confs.sdf")
    opt_ts_mol = RDKitMol.FromFile(ts_file, sanitize=False, sameMol=True)

    r_smi = RDKitMol.FromFile(os.path.join(save_dir, "reactant_confs.sdf"))[0].GetProp("_Name")
    p_smi = RDKitMol.FromFile(os.path.join(save_dir, "product_confs.sdf"))[0].GetProp("_Name")
    rxn_smiles = r_smi + ">>" + p_smi

    freq_verifier = XTBFrequencyVerifier()
    irc_verifier = OrcaIRCVerifier()

    keep_ids = [True] * opt_ts_mol.GetNumConformers()
    keep_ids = freq_verifier(opt_ts_mol, keep_ids=keep_ids, save_dir=save_dir, rxn_smiles=rxn_smiles)
    keep_ids = irc_verifier(opt_ts_mol, keep_ids=keep_ids, save_dir=save_dir, rxn_smiles=rxn_smiles)


if __name__ == "__main__":
    args = parser.parse_args()
    optimize_conf(args)
