import pandas as pd


def str2bool(var: bool | int | str) -> bool:
    """
    Convert a string to a boolean argument
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(var, bool):
        return var
    elif isinstance(var, int):
        if var == 1:
            return True
        elif var == 0:
            return False

    elif var.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif var.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError("Boolean or equivalent value expected.")
