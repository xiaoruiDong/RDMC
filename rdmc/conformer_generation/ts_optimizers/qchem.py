from rdmc.conformer_generation.optimizers.qchem import (
    QChemOptimizer as ConfQChemOptimizer,
)


class QChemOptimizer(ConfQChemOptimizer):
    """
    The class to optimize TS geometries using the Baker's eigenvector-following (EF) algorithm built in QChem.
    You have to have the QChem package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the method available in QChem.
                                Defaults to ``"wB97x-d3"``.
        basis (str, optional): The method to be used for TS optimization. you can use the basis available in QChem.
                                Defaults to ``"def2-tzvp"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    optimize_ts = True


# Xiaorui's note: keep this module to make sure backward compatibility
