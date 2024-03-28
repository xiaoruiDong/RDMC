from rdmc.conformer_generation.comp_env import xtb_available


class XTBTask:

    def __init__(
        self,
        method: str = "GFN2-xTB",
    ):
        self.method = method

    def is_available(self):
        return xtb_available
