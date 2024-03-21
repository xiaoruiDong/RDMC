from pathlib import Path
import shutil
from contextlib import contextmanager


module_available = {}
binary_available = {}


@contextmanager
def register_module(software):
    """
    Register the availability of external software.
    """
    if module_available.get(software) is False:
        return
    try:
        yield
    except ImportError:
        module_available[software] = False
    else:
        module_available[software] = True


def register_binary(binary: str) -> str:
    """
    Get the path to the binary of a software.

    Args:
        binary (str): Name of the binary.

    Returns:
        str: Path to the binary if available, otherwise an empty string.
    """
    path = shutil.which(binary) or ''
    binary_available[binary] = path if path != '' else None
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
