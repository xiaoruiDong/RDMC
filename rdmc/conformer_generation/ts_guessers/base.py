from abc import abstractmethod
from typing import Optional


from rdkit import Chem
from rdmc.conformer_generation.task.basetask import BaseTask


class TSInitialGuesser(BaseTask):
    """
    The abstract class for TS initial Guesser.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self,
        track_stats: Optional[bool] = False,
    ):
        """
        Initialize the TS initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super().__init__(track_stats)
        self.n_success = None
        self.percent_success = None

    @abstractmethod
    def run(self):
        """
        The key function used to generate TS guesses. It varies by the actual classes and need to implemented inside each class.
        The function should at least take mols and multiplicity as input arguments. The returned value should be a RDKitMol with TS
        geometries.

        Args:
            mols (list): A list of reactant and product pairs.

        Returns:
            RDKitMol: The TS molecule in ``RDKitMol`` with 3D conformer saved with the molecule.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
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
