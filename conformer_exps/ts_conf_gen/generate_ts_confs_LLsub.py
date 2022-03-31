import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pickle
from rdmc.conformer_generation.ts_generators import TSConformerGenerator
from rdmc.conformer_generation.ts_guessers import TSEGNNGuesser, RMSDPPGuesser, AutoNEBGuesser, TSGCNGuesser
from rdmc.conformer_generation.ts_verifiers import XTBFrequencyVerifier, OrcaIRCVerifier, GaussianIRCVerifier
from rdmc.conformer_generation.ts_optimizers import SellaOptimizer, OrcaOptimizer, GaussianOptimizer

parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--rxns_path", type=str)
parser.add_argument("--split_path", type=str)
parser.add_argument("--n_ts_conformers", type=int, default=20)
parser.add_argument("--opt_method", type=str, default="GFN2-xTB")
parser.add_argument("--guess_method", type=str, default="ts_egnn")
parser.add_argument("--task_id", type=int)
parser.add_argument("--num_tasks", type=int)


def generate_conf(args):

    all_rxns = pd.read_csv(args.rxns_path)
    _ids = np.load(args.split_path, allow_pickle=True)[2]
    rxn_ids = _ids[args.task_id:len(_ids):args.num_tasks]

    for rxn_idx in rxn_ids:

        try:

            rxn_data = all_rxns.iloc[rxn_idx]
            rxn_smiles = rxn_data["rsmi"] + ">>" + rxn_data["psmi"]

            if args.guess_method == "ts_egnn":
                trained_model_dir = "~/code/RDMC/rdmc/external/ts_egnn/trained_models/ts_egnn/2022_02_07/"
                # trained_model_dir = "../../external/ts_egnn/trained_models/2022_02_07"
                embedder = TSEGNNGuesser(trained_model_dir, track_stats=True)
            elif args.guess_method == "ts_gcn":
                trained_model_dir = "~/code/RDMC/rdmc/external/ts_egnn/trained_models/ts_gcn/2022_03_18/"
                embedder = TSGCNGuesser(trained_model_dir, track_stats=True)
            elif args.guess_method == "rmsd_pp":
                embedder = RMSDPPGuesser(track_stats=True)
            elif args.guess_method == "auto_neb":
                embedder = AutoNEBGuesser(track_stats=True)

            optimizer = GaussianOptimizer(track_stats=True, method=args.opt_method, nprocs=2)
            verifiers=[XTBFrequencyVerifier(track_stats=True), GaussianIRCVerifier(track_stats=True, nprocs=2)]
            save_dir = os.path.join(args.exp_dir, args.guess_method, str(rxn_idx))

            ts_gen = TSConformerGenerator(
                rxn_smiles=rxn_smiles,
                embedder=embedder,
                optimizer=optimizer,
                verifiers=verifiers,
                save_dir=save_dir
            )

            opt_ts_mol = ts_gen(args.n_ts_conformers)

            stats = [x for y in [ts_gen.embedder.stats] + [ts_gen.optimizer.stats] + [v.stats for v in ts_gen.verifiers] for x in y]
            with open(os.path.join(save_dir, "stats.pkl"), "wb") as f:
                pickle.dump(stats, f)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = parser.parse_args()
    generate_conf(args)
