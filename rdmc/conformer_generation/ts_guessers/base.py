import os
from time import time
from typing import Optional

from rdkit import Chem


class TSInitialGuesser:
    """
    The abstract class for TS initial Guesser.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    _avail_ = True

    def __init__(
        self,
        track_stats: Optional[bool] = False,
    ):
        """
        Initialize the TS initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        assert (
            self._avail
        ), f"The dependency requirement needs to be fulfilled to use {self.__class__.__name__}. Please install the relevant dependencies and try again.."
        self.track_stats = track_stats
        self.n_success = None
        self.percent_success = None
        self.stats = []

    def generate_ts_guesses(
        self,
        mols: list,
        save_dir: Optional[str] = None,
    ) -> "RDKitMol":
        """
        The key function used to generate TS guesses. It varies by the actual classes and need to implemented inside each class.
        The function should at least take mols and save_dir as input arguments. The returned value should be a RDKitMol with TS
        geometries.

        Args:
            mols (list): A list of reactant and product pairs.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None`` for not saving.

        Returns:
            RDKitMol: The TS molecule in ``RDKitMol`` with 3D conformer saved with the molecule.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
        """
        raise NotImplementedError

    def save_guesses(self, save_dir: str, rp_combos: list, ts_mol: "RDKitMol"):
        """
        Save the generated guesses into the given ``save_dir``.

        Args:
            save_dir (str): The path to the directory to save the results.
            rp_combos (list): A list of reactant and product complex pairs used to generate transition states.
            ts_mol (RDKitMol): The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """

        # Save reactants and products into SDF format
        r_path = os.path.join(save_dir, "reactant_confs.sdf")
        p_path = os.path.join(save_dir, "product_confs.sdf")
        try:
            r_writer = Chem.rdmolfiles.SDWriter(r_path)
            p_writer = Chem.rdmolfiles.SDWriter(p_path)

            for r, p in rp_combos:

                if r.GetProp("Identity") == "reactant":
                    reactant = r
                    product = p
                elif r.GetProp("Identity") == "product":
                    reactant = p
                    product = r

                reactant, product = reactant, product
                reactant.SetProp("_Name", f"{Chem.MolToSmiles(reactant)}")
                product.SetProp("_Name", f"{Chem.MolToSmiles(product)}")
                r_writer.write(reactant)
                p_writer.write(product)

        except Exception:
            raise
        finally:
            r_writer.close()
            p_writer.close()

        # save TS initial guesses
        ts_path = os.path.join(save_dir, "ts_initial_guess_confs.sdf")
        try:
            ts_writer = Chem.rdmolfiles.SDWriter(ts_path)
            for i in range(ts_mol.GetNumConformers()):
                ts_writer.write(ts_mol, confId=i)
        except Exception:
            raise
        finally:
            ts_writer.close()

    def __call__(
        self,
        mols: list,
        multiplicity: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        """
        The workflow to generate TS initial guesses.

        Args:
            mols (list): A list of molecules.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None`` for not setting.
            save_dir (str, optional): The path to save results. Defaults to ``None`` for not saving.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        time_start = time()
        ts_mol_data = self.generate_ts_guesses(mols, multiplicity, save_dir)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return ts_mol_data
