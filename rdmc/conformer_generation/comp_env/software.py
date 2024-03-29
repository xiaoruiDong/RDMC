import importlib
from pathlib import Path
import shutil
from typing import Optional

from rdtools.utils import FakeModule


package_available = {}
binary_available = {}


def try_import(
    full_module_name: str,
    alias: Optional[str] = None,
    namespace: Optional[dict] = None,
    package_name: Optional[str] = None,
):
    """
    A helper function to import a module, function, or class. If the module is not available,
    a fake module will be created.

    Args:
        full_module_name (str): The full name of the module, function, or class.
        alias (str, optional): The alias of the module, function, or class. Defaults to ``None``.
        namespace (dict, optional): The namespace where the module, function, or class will be
            imported. Defaults to ``None``.
        package_name (str, optional): The name of the package. Defaults to ``None``. This is used
            in providing instructions if the module is not available.

    Examples:
        >>> from rdtools.software import try_import
        >>> try_import("rdmc.conformer_generation.ts_guessers.TSEGNNGuesser")
        >>> try_import("ase.Atoms")
    """

    if namespace is None:
        namespace = globals()

    module_name, _, attribute = full_module_name.rpartition(".")
    # rdmc.conformer_generation.ts_guessers => "rdmc.conformer_generation", "ts_guessers"
    # ase => "", "ase"
    # rdkit.Chem => "rdkit", "Chem"

    if not module_name:
        module_name, attribute = attribute, module_name
        package_name = package_name or module_name
    else:
        package_name = package_name or module_name.split(".")[0]

    try:
        if module_name and attribute:
            module = importlib.import_module(module_name)
            try:
                attr = getattr(module, attribute)
                namespace[alias or attribute] = attr

                package_available[package_name] = (
                    package_available.get(package_name) or True
                )

            except AttributeError:
                namespace[alias or attribute] = FakeModule(
                    full_module_name, package_name
                )

                # Not overwriting it, it is possible a module no longer available
                # but the software is installed
                package_available[package_name] = (
                    package_available.get(package_name) or False
                )
        else:
            module = importlib.import_module(full_module_name)
            namespace[alias or module_name] = module
    except ImportError:
        namespace[alias or attribute or module_name] = FakeModule(
            full_module_name, package_name
        )

        # Not overwriting it, it is possible a module no longer available
        # but the software is installed
        package_available[package_name] = package_available.get(package_name) or False


def register_binary(binary: str) -> str:
    """
    Get the path to the binary of a software.

    Args:
        binary (str): Name of the binary.

    Returns:
        str: Path to the binary if available, otherwise an empty string.
    """
    path = shutil.which(binary) or ""
    binary_available[binary] = path if path != "" else None
    return path


def set_binary(binary: str, path: str) -> None:
    """
    Set the path to the binary of a software.

    Args:
        binary (str): Name of the binary.
        path (str): Path to the binary.
    """
    if Path(path).is_file():
        binary_available[binary] = path
    else:
        # todo: Probably raises an warning.
        pass
    return None


def get_binary(binary: str) -> str:
    """
    Get the path to the binary of a software.

    Args:
        binary (str): Name of the binary.

    Returns:
        str: Path to the binary if available, otherwise an empty string.
    """
    if binary not in binary_available:
        register_binary(binary)
    return binary_available.get(binary, None)


def has_binary(binary: str) -> bool:
    """
    Check if the binary is available.

    Args:
        binary (str): Name of the binary.

    Returns:
        bool: True if the binary is available, otherwise False.
    """
    return get_binary(binary) is not None
