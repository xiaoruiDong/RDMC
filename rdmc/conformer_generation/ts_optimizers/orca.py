from rdmc.conformer_generation.optimizers.orca import (
    OrcaOptimizer as ConfOrcaOptimizer,
)


class OrcaOptimizer(ConfOrcaOptimizer):
    """
    The class to optimize TS geometries using the Berny algorithm built in Orca.
    You have to have the Orca package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                Defaults to ``"XTB2"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    optimize_ts = True


# Xiaorui's note: keep this module to make sure backward compatibility
