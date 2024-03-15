from rdmc.rdtools.view.base import base_viewer


# Arrow in freq view indicates the force direction
default_arrow_spec = {
    "color": "black",
    "radius": 0.08,
}

def freq_viewer(
    xyz: str,
    frames: int = 10,
    amplitude: float = 1.0,
    **kwargs,
) -> "py3Dmol.view":
    """
    Create a viewer for viewing frequency modes. The function only accepts xyz string with dx, dy, dz
    properties appended. A typical line in the xyz file looks like ``"C 0. 0. 0. 0.1 0.2 0.0"``. This
    viewer accepts additional keyword arguments following the same way base_viewer is implemented.

    Args:
        obj (str): The xyz string with dx, dy, dz properties.
        frames (int, optional): Number of frames to be created. Defaults to ``10``.
        amplitude (float, optional): amplitude of distortion. Defaults to ``1.0``.

    Returns:
        py3Dmol.view: The molecule frequency viewer.
    """
    viewer = base_viewer(
        xyz, 'xyz',
        model_extra={"vibrate": {"frames": frames, "amplitude": amplitude}},
        animate={'loop': 'backAndForth'},
        **kwargs,
        )
    viewer.vibrate(
        frames,
        amplitude,
        True,  # vibrate bothways
        default_arrow_spec,
    )
    return viewer


def merge_xyz_dxdydz(
    xyz: str,
    dxdydz: list,
) -> str:
    """
    A helper function to create input for freq_viewer.
    Merge the xyz string with the dxdydz information.

    Args:
        xyz (str): The xyz string.
        dxdydz (list): The dx dy dz in a 3 x N matrix like list or array.

    Returns:
        str: The xyz string with dxdydz information.
    """
    lines = xyz.strip().splitlines()

    assert len(lines) - 2 == len(dxdydz), (
        f"The number of atoms doesn't match xyz ({len(lines) - 2}) "
        f"and dxdydz ({len(dxdydz)})"
    )

    new_lines = lines[0:2] + [
        line.strip() + "\t\t" + "".join([f"{dx:<12}" for dx in dxdydz_onerow])
        for line, dxdydz_onerow in zip(lines[2:], dxdydz)
    ]

    return "\n".join(new_lines)
