import pathlib


def makepath(path: str) -> None:
    """Checks if path exists and makes it if it doesn't

    Args:
        path (str): path address
    """
    if not pathlib.Path(path).exists():
        pathlib.Path(path).mkdir(parents=True)
