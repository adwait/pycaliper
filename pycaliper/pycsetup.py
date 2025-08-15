"""
File: pycaliper/pycsetup.py

This file provides setup and workflow management for PyCaliper CLI tasks.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

import os
import sys
import logging
import random

import tempfile
import importlib

import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from pycaliper.jginterface import jasperclient as jgc

from pydantic import BaseModel
from .pycconfig import PYConfig, DesignConfig, JasperConfig
from .per.per import SpecModule

logger = logging.getLogger(__name__)


class PYCArgs(BaseModel):
    """Arguments for PyCaliper tasks.

    This class defines the arguments that can be passed to PyCaliper tasks.
    It uses Pydantic for validation and default values.

    Attributes:
        specpath (str): Path to the specification module.
        jgcpath (str): Path to the Jasper configuration file.
        dcpath (str): Path to the design configuration file.
        params (str): Parameters for the specification module.
        sdir (str): Directory to save results to.
        tdir (str): Directory containing trace files.
        onetrace (bool): Whether to verify only one-trace properties.
        bmc (str): Bounded model checking configuration.
        requires_jasper (bool): Whether this task requires Jasper connection.
    """

    specpath: str = ""  #: Path to the specification module
    jgcpath: str = ""  #: Path to the Jasper configuration file
    dcpath: str = ""  #: Path to the design configuration file
    params: str = ""  #: Parameters for the specification module
    sdir: str = ""  #: Directory to save results to
    tdir: str = ""  #: Directory containing trace files
    onetrace: bool = False  #: Whether to verify only one-trace properties
    bmc: str = ""  #: Bounded model checking configuration
    requires_jasper: bool = True  #: Whether this task requires Jasper connection


# Import configuration functions from proofmanager to avoid circular imports
from .proofmanager import get_jgconfig, get_designconfig, mock_or_connect


def get_specmodname(specmod):
    """Extract the specification module name from a path.

    Args:
        specmod (str): Path to the specification module.

    Returns:
        str: Name of the specification module.
    """
    if "/" in specmod:
        module_name = specmod.rsplit("/", 1)[1]
        if "." in module_name:
            module_name = module_name.rsplit(".", 1)[0]
        return module_name
    return specmod


def create_module(specpath, args):
    """Dynamically import the spec module and create an instance of it."""

    params = {}
    if args.params:
        for pair in args.params.split(","):
            key, value = pair.split("=")
            params[key] = int(value)

    # Split the module name into the module name and the parent package
    module_path, module_name = specpath.rsplit("/", 1)

    # Check if the path exists
    if not os.path.isdir(module_path):
        logger.error(f"Path '{module_path}' does not exist.")
        exit(1)
    # Add the module path to sys.path
    sys.path.append(module_path)

    try:
        if "." in module_name:
            module_name, class_name = module_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            logger.debug(
                f"Successfully imported module: {module_name} from {module_path}"
            )
            return getattr(module, class_name)(**params)
        else:
            # Import the module using importlib
            module = importlib.import_module(module_name)
            logger.debug(
                f"Successfully imported module: {module_name} from {module_path}"
            )
            return getattr(module, module_name)(**params)
    except ImportError as e:
        logger.error(f"Error importing module {module_name} from {module_path}: {e}")
        return None
    finally:
        # Clean up: remove the path from sys.path to avoid potential side effects
        sys.path.remove(module_path)


def get_pyconfig(args: PYCArgs) -> PYConfig:
    """Create a PyCaliper configuration from arguments.

    This function creates a PyCaliper configuration from the provided arguments,
    including loading and validating Jasper and design configurations.

    Args:
        args (PYCArgs): PyCaliper arguments.

    Returns:
        PYConfig: PyCaliper configuration.
    """

    if args.jgcpath != "":
        jasperc = get_jgconfig(args.jgcpath)
    else:
        jasperc = JasperConfig()

    if args.dcpath != "":
        designc = get_designconfig(args.dcpath)
    else:
        designc = DesignConfig()

    return PYConfig(
        # Working directory
        sdir=args.sdir,
        # Is this a mock run
        mock=(args.jgcpath == ""),
        # Jasper configuration
        jgc=jasperc,
        # Spec config
        pycspec=get_specmodname(args.specpath),
        # Trace directory
        tdir=args.tdir,
        onetrace=args.onetrace,
        dc=designc,
    )


def start(args: PYCArgs) -> tuple[PYConfig, SpecModule]:
    """Start a PyCaliper task.

    This function sets up all components for a PyCaliper task and checks that
    the task can be run in the current mode.

    Args:
        args (PYCArgs): PyCaliper arguments.

    Returns:
        tuple: Tuple of (pyconfig, module).
    """
    pyconfig = get_pyconfig(args)
    is_connected = mock_or_connect(pyconfig.mock, pyconfig.jgc.port)

    module = create_module(args.specpath, args)
    assert module is not None, f"SpecModule {args.specpath} not found."

    if args.requires_jasper and not is_connected:
        logger.error(
            "This task requires Jasper sockets, cannot be run in mock mode."
        )
        sys.exit(1)

    return pyconfig, module
