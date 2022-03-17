import os
from argparse import ArgumentParser
import pandas as pd
from rdmc.conformer_generation.ts_generators import TSConformerGenerator
from rdmc.conformer_generation.ts_guessers import TSEGNNGuesser
from rdmc.conformer_generation.ts_optimizers import SellaOptimizer

parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--rxns_path", type=str)
parser.add_argument("--rxn_idx", type=int)
parser.add_argument("--n_ts_conformers", type=int, default=20)
parser.add_argument("--opt_method", type=str, default="GFN2-xTB")


def generate_conf(args):

    rxn_data = pd.read_csv(args.rxns_path)
    rxn_smiles = rxn_data.iloc[args.rxn_idx].item()

    # trained_model_dir = "~/code/RDMC/rdmc/external/ts_egnn/trained_models/2022_02_07/"
    trained_model_dir = "../../external/ts_egnn/trained_models/2022_02_07"
    embedder = TSEGNNGuesser(trained_model_dir, track_stats=True)
    optimizer = SellaOptimizer(track_stats=True, method=args.opt_method)
    save_dir = os.path.join(args.exp_dir, str(args.rxn_idx))

    ts_gen = TSConformerGenerator(
        rxn_smiles=rxn_smiles,
        embedder=embedder,
        optimizer=optimizer,
        save_dir=save_dir
    )

    opt_ts_mol = ts_gen(args.n_ts_conformers)


if __name__ == "__main__":
    args = parser.parse_args()
    generate_conf(args)
