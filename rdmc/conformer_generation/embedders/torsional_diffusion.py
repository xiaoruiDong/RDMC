import csv
from pathlib import Path
import pickle

from rdmc import Mol
from rdmc.conformer_generation.embedders.base import ConfGenEmbedder
from rdmc.conformer_generation.utils import subprocess_runner

from rdtools.atommap import has_atom_map_numbers, reset_atom_map_numbers
from rdtools.conf import add_conformer


class TorsionalDiffusionEmbedder(ConfGenEmbedder):
    """
    Embed conformers using TorsionalDiffusion. In the current implementation Torsional-Diffusion will always yield even-number
    of conformers.

    Args:
        repo_dir (str): Directory to the cloned torsion-diffusion repository.
        model_dir (str): Directory of the trained model.
        track_stats (bool, optional): Whether to track the statistics of the conformer generation. Defaults to ``False``.
    """

    def __init__(
        self,
        repo_dir: str,
        model_dir: str,
        track_stats: bool = False,
    ):
        super().__init__(track_stats)

        self.repo_dir = Path(repo_dir)
        self.model_dir = Path(model_dir)

        self.check_availability()

    def is_available(self):
        """
        Check if Torsional-Diffusion is available.

        Returns:
            bool
        """
        return self.repo_dir.exists() and self.model_dir.exists()

    def run(self, smiles: str, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            smiles (str): SMILES string of the molecule.
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        # Create an input file for Torsional-Diffusion
        with open(self.work_dir / "smiles.csv", "w", newline="") as csvfile:
            fieldnames = ["smiles", "n_conformers", "corrected_smiles"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "smiles": smiles,
                    "n_conformers": n_conformers // 2,
                    "corrected_smiles": smiles,
                }
            )  # according to the doc, the number of generated conformers will be doubled

        # Run Torsional-Diffusion
        output_file = self.work_dir / "conformers.pkl"
        command = [
            "python",
            str(self.repo_dir / "generate_confs.py"),
            "--test_csv",
            str(self.work_dir / "smiles.csv"),
            "--inference_steps",
            "20",  # temporarily hardcoded, the default models are trained with 20 steps
            "--model_dir",
            str(self.model_dir),
            "--out",
            str(output_file),
            "--batch_size",
            "128",
            "--no_energy",
        ]
        subprocess_returncode = subprocess_runner(
            command,
            self.work_dir / "torsional_diffusion.log",
            self.work_dir,
        )

        # Read the generated conformers
        try:
            with open(output_file, "rb") as f:
                # output is a dict with smiles as keys and a list of conformers as values
                mols = pickle.load(f)[smiles]
            assert len(mols) == n_conformers

            # Check if the original smiles has atom map numbers
            # If so, reorder the conformers positions according to the atom map numbers
            if has_atom_map_numbers(mols[0]):
                mol = Mol.FromSmiles(smiles)
                atommap = mols[0].GetSubstructMatch(mol)
                for conf_id in range(n_conformers):
                    coords = mols[conf_id].GetConformer().GetPositions()
                    add_conformer(mol, coords=coords[atommap, :], conf_id=conf_id)

            else:
                mol = Mol(mols[0])
                for conf_id in range(1, n_conformers):
                    add_conformer(
                        mol, conf=mols[conf_id].GetConformer(), conf_id=conf_id
                    )
                reset_atom_map_numbers(mol)  # Assign Atom Map numbers

            mol.KeepIDs = {i: True for i in range(n_conformers)}

        except (FileNotFoundError, AssertionError) as e:
            print("Torsional-Diffusion failed to generate conformers.")
            print(e)
            mol = Mol.FromSmiles(smiles)
            mol.EmbedMultipleNullConfs(n_conformers, random=True)
            mol.KeepIDs = {i: False for i in range(n_conformers)}

        return mol
