from rdmc.conformer_generation.optimizers.gaussian import (
    GaussianOptimizer as ConfGaussianOptimizer,
)


class GaussianOptimizer(ConfGaussianOptimizer):
    """
    The class to optimize TS geometries using the Berny algorithm built in Gaussian.
    You have to have the Gaussian package installed to run this optimizer

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do.
                                Defaults to ``"GFN2-xTB"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    optimize_ts = True


# Xiaorui's note: keep this module to make sure backward compatibility
