from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from rdkit import Chem
from rdmc.conformer_generation.task.basetask import BaseTask


class TSInitialGuesser(BaseTask):
    """
    The abstract class for TS initial Guesser.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def run(self, mols, **kwargs):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        n_conf = len(mols)

        try:
            positions, successes = self.generate_ts_guesses(mols, **kwargs)
        except BaseException as e:
            print("Run into errors when generating TS guesses.")
            print(e)
            # todo: add a log
            positions = {i: None for i in range(n_conf)}
            successes = {i: False for i in range(n_conf)}

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(n_conf)
        [
            ts_mol.GetEditableConformer(i).SetPositions(p)
            for i, p in positions.items()
            if p is not None
        ]

        if self.save_dir:
            self.save_guesses(mols, ts_mol)

        ts_mol.KeepIDs = successes

        return ts_mol

    def generate_ts_guesses(self, mols, **kwargs) -> Tuple[dict, dict]:
        """
        Generate TS guesses. This method includes a workflow to sequentially generate TS guess for each pair of reactant and product.
        You can either `implement generate_ts_guess` or choose to re-implement this method in the child class.

        Args:
            mols (list): A list of reactant and product pairs.

        Returns:
            Tuple[dict, dict]: The generated guesses positions and the success status. The keys are the conformer IDs. The values are the positions and the success status.
        """
        positions, successes = {}, {}
        for i, (r_mol, p_mol) in enumerate(mols):

            try:
                pos, success = self.generate_ts_guess(r_mol, p_mol, conf_id=i, **kwargs)
                positions[i] = pos
                successes[i] = success
            except BaseException:
                positions[i] = None
                successes[i] = False
        return positions, successes

    def generate_ts_guess(
        self, r_mol, p_mol, conf_id: int = 0, **kwargs
    ) -> Tuple[np.ndarray, bool]:
        """
        Generate a single TS guess.

        Args:
            rmol (RDKitMol): The reactant molecule in RDKitMol with 3D conformer saved with the molecule.
            pmol (RDKitMol): The product molecule in RDKitMol with 3D conformer saved with the molecule.

        Returns:
            Tuple[np.ndarray, bool]: The generated guess positions and the success status.
        """
        raise NotImplementedError

    def save_guesses(self, rp_combos: list, ts_mol: "RDKitMol"):
        """
        Save the generated guesses into the ``self.save_dir``.

        Args:
            rp_combos (list): A list of reactant and product complex pairs used to generate transition states.
            ts_mol (RDKitMol): The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """

        save_dir = self.save_dir

        # Save reactants and products into SDF format
        r_path = save_dir / "reactant_confs.sdf"
        p_path = save_dir / "product_confs.sdf"
        try:
            r_writer = Chem.rdmolfiles.SDWriter(str(r_path))
            p_writer = Chem.rdmolfiles.SDWriter(str(p_path))

            for reactant, product in rp_combos:

                try:
                    if reactant.GetProp("Identity") == "product":
                        reactant, product = product, reactant
                except KeyError:
                    # No identity prop
                    reactant.SetProp("Identity", "reactant")
                    product.SetProp("Identity", "product")

                reactant.SetProp("_Name", f"{Chem.MolToSmiles(reactant)}")
                product.SetProp("_Name", f"{Chem.MolToSmiles(product)}")
                r_writer.write(reactant)
                p_writer.write(product)

        except BaseException:
            raise RuntimeError("Unknown Error when saving TS guess into SDF files.")
        finally:
            r_writer.close()
            p_writer.close()

        # save TS initial guesses
        ts_path = save_dir / "ts_initial_guess_confs.sdf"
        try:
            ts_writer = Chem.rdmolfiles.SDWriter(str(ts_path))
            for i in range(ts_mol.GetNumConformers()):
                ts_writer.write(ts_mol, confId=i)
        except Exception:
            raise RuntimeError("Unknown Error when saving TS guess into SDF files.")
        finally:
            ts_writer.close()
