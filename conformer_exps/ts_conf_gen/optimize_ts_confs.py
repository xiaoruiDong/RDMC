import os
from argparse import ArgumentParser
from rdmc import RDKitMol
import pickle
from rdmc.conformer_generation.ts_optimizers import SellaOptimizer

parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--ts_file", type=str)
parser.add_argument("--val_ids", type=str)
parser.add_argument("--rxn_idx", type=int)
parser.add_argument("--opt_method", type=str, default="GFN2-xTB")


def optimize_conf(args):

    ts_mols = RDKitMol.FromFile(args.ts_file, sanitize=False)
    with open(args.val_ids, "rb") as f:
        val_ids = pickle.load(f)
    ts_idx = val_ids[args.rxn_idx]
    ts_mol = ts_mols[ts_idx]

    optimizer = SellaOptimizer(track_stats=True, method=args.opt_method)
    save_dir = os.path.join(args.exp_dir, args.opt_method, str(ts_idx))
    opt_ts_mol = optimizer(ts_mol, save_dir=save_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    optimize_conf(args)
