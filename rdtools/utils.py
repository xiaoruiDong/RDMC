from typing import Optional

FAKE_PACKAGES = set()


class FakeModule:

    def __init__(self, module_name: str, package_name: Optional[str] = None):

        object.__setattr__(self, "module_name", module_name)
        object.__setattr__(self, "package_name", package_name or module_name)
        FAKE_PACKAGES.add(package_name or module_name)

    def __getattribute__(self, __name: str):

        module_name = object.__getattribute__(self, "module_name")
        package_name = object.__getattribute__(self, "package_name")

        raise RuntimeError(
            f"The method ({module_name}.{__name}) cannot be run as {package_name} is not installed. "
            f"Please install the {package_name} and restart the python Kernel."
        )

    def __call__(self, *args, **kwargs):

        module_name = object.__getattribute__(self, "module_name")
        package_name = object.__getattribute__(self, "package_name")

        raise RuntimeError(
            f"The method ({module_name}()) cannot be run as {package_name} is not installed. "
            f"Please install the {package_name} and restart the python Kernel."
        )


def get_fake_module(module_name: str, package_name: Optional[str] = None):
    """
    Returns a fake module that raises an error when used.
    """
    return FakeModule(module_name, package_name)
