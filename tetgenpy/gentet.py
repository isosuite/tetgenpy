from tetgenpy import tetgenpy_core as core
from tetgenpy.plc import PLC


def tetgen_exe(argv):
    """
    Calls tetgen executable with argv as a list.
    This call performs the same routine as tetgen's main().
    The first element of argv is usually the executable name and tetgen also
    expects that. However, this can be a dummy.

    Parameters
    ----------
    argv: list

    Returns
    -------
    exit_code: int
      Will return 0 if everything went well. -1 with RuntimeError without
      any informative error message.
    """
    try:
        # only list
        if not isinstance(argv, list):
            raise TypeError("argv should be a list.")

        if len(argv) == 0:
            raise ValueError("argv needs at least one element")

        return core.main(argv)

    except RuntimeError:
        return -1


def _to_tetgenio(maybe_plc):
    """
    Convert input to TetgenIO: interpretable type for tetrahedralize().
    """
    if isinstance(maybe_plc, core.TetgenIO):
        return maybe_plc

    elif isinstance(maybe_plc, PLC):
        return maybe_plc.to_tetgenio()

    else:
        raise TypeError(
            "inputs for tetraheralize should be either TetgenIO or PLC."
            f"Given {type(maybe_plc)}"
        )


def tetrahedralize(
    switches, tetgenio_in, additional_points=None, background_mesh=None
):
    """
    Calls tetrahedralize from the core with appropriate parameters.
    Input and output is based on TetgenIO.

    Parameters
    ----------
    switches: str
    tetgenio_in: TetgenIO
    additional_points: TetgenIO
    background_mesh: TetgenIO

    Returns
    -------
    out: TetgenIO
    """
    out = core.TetgenIO()

    # fill dummies
    if additional_points is None:
        additional_points = core.TetgenIO()

    if background_mesh is None:
        background_mesh = core.TetgenIO()

    # type
    switches = str(switches)
    tetgenio_in = _to_tetgenio(tetgenio_in)
    additional_points = _to_tetgenio(additional_points)
    background_mesh = _to_tetgenio(background_mesh)

    # call
    core.tetrahedralize(
        switches, tetgenio_in, out, additional_points, background_mesh
    )

    return out
