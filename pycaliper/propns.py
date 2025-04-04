"""
File: pycaliper/propns.py

This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley

Namespace for properties
"""

# Properties for top module
TOP_INPUT_2T_PROP = "input"
TOP_STATE_2T_PROP = "state"
TOP_OUTPUT_2T_PROP = "output"

TOP_INPUT_1T_PROP = "input_inv"
TOP_STATE_1T_PROP = "state_inv"
TOP_OUTPUT_1T_PROP = "output_inv"

STEP_PROP = "step"


def TOP_STEP_PROP(fn_name: str, k: int) -> str:
    """Gets the property name for a given step.

    Args:
        fn_name (str): The function name.
        k (int): The step number.

    Returns:
        str: The property name for the given step.
    """
    return f"{STEP_PROP}_{fn_name}_{k}"


def get_as_assm(prop: str) -> str:
    """Gets the assumption name for a given property.

    Args:
        prop (str): The property name.

    Returns:
        str: The assumption name for the given property.
    """
    return f"A_{prop}"


def get_as_prop(prop: str) -> str:
    """Gets the assertion name for a given property.

    Args:
        prop (str): The property name.

    Returns:
        str: The assertion name for the given property.
    """
    return f"P_{prop}"


def eq_sva(s: str) -> str:
    """Gets the equality SVA for a given string.

    Args:
        s (str): The string to use.

    Returns:
        str: The equality SVA.
    """
    return f"eq_{s}"


def condeq_sva(s: str) -> str:
    """Gets the conditional equality SVA for a given string.

    Args:
        s (str): The string to use.

    Returns:
        str: The conditional equality SVA.
    """
    return f"condeq_{s}"
