from typing import Optional


class FakeModule:

    def __init__(self, module_name: str, package_name: Optional[str] = None):
        package_name = package_name or module_name
        self.module_name = module_name
        self.package_name = package_name

    def __getattribute__(self, __name: str):
        raise RuntimeError(
            f"The method ({self.module_name}.{__name}) cannot be run as {self.package_name} is not installed. "
            f"Please install the {self.package_name} and restart the python Kernel."
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"The function ({self.module_name}) cannot be run as {self.package_name} is not installed. "
            f"Please install the {self.package_name} and restart the python Kernel."
        )


def get_fake_module(module_name: str, package_name: Optional[str] = None):
    """
    Returns a fake module that raises an error when used.
    """
    return FakeModule(module_name, package_name)
