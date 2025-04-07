"""Utils used in the RDTools package."""

from typing import Any, Optional

FAKE_PACKAGES = set()


class FakeModule:
    """Fake module that raises an error when used.

    This is to allow users to import modules that are not installed, but run into
    an error when they try to use them. This is useful for optional dependencies.

    Args:
        module_name (str): The name of the module.
        package_name (Optional[str], optional): The name of the package. Defaults to None.
    """

    def __init__(self, module_name: str, package_name: Optional[str] = None):
        object.__setattr__(self, "module_name", module_name)
        object.__setattr__(self, "package_name", package_name or module_name)
        FAKE_PACKAGES.add(package_name or module_name)

    def __getattribute__(self, __name: str) -> None:
        """Get the attribute of the module.

        Args:
            __name (str): The name of the attribute.

        Raises:
            RuntimeError: If the module is not installed.
        """
        module_name = object.__getattribute__(self, "module_name")
        package_name = object.__getattribute__(self, "package_name")

        raise RuntimeError(
            f"The method ({module_name}.{__name}) cannot be run as {package_name} is not installed. "
            f"Please install the {package_name} and restart the python Kernel."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Call the module.

        Args:
            *args (Any): The arguments to the module.
            **kwargs (Any): The keyword arguments to the module.

        Raises:
            RuntimeError: If the module is not installed.
        """
        module_name = object.__getattribute__(self, "module_name")
        package_name = object.__getattribute__(self, "package_name")

        raise RuntimeError(
            f"The method ({module_name}()) cannot be run as {package_name} is not installed. "
            f"Please install the {package_name} and restart the python Kernel."
        )


def get_fake_module(module_name: str, package_name: Optional[str] = None) -> FakeModule:
    """Returns a fake module that raises an error when used.

    Args:
        module_name (str): The name of the module.
        package_name (Optional[str], optional): The name of the package. Defaults to None.

    Returns:
        FakeModule: The fake module.
    """
    return FakeModule(module_name, package_name)
