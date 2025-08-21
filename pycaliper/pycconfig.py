"""
File: pycaliper/pycconfig.py

This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

from pydantic import BaseModel


class JasperConfig(BaseModel):
    """Configuration for Jasper Gold formal verification tool.

    This class holds configuration parameters for interacting with the Jasper Gold
    formal verification tool.

    Attributes:
        jdir (str): Jasper working directory relative to the pycaliper directory.
        script (str): TCL script relative to the Jasper working directory.
        pycfile (str): Location of the generated SVA file relative to the Jasper working directory.
        context (str): Proof node context.
        design_list (str): Design list file.
        port (int): Port number to connect to Jasper server.
    """

    jdir: str = ""
    script: str = ""
    pycfile: str = ""
    context: str = ""
    design_list: str = "design.lst"
    port: int = 8080

    def pycfile_abspath(self) -> str:
        """Gets the absolute path to the PyCaliper specification file.

        Returns:
            str: Absolute path to the PyCaliper specification file.
        """
        return f"{self.jdir}/{self.pycfile}"


class Design:
    """Base class for design representations.

    This class serves as a base for different design representations in PyCaliper.

    Attributes:
        name (str): Name of the design.
    """

    def __init__(self, name: str) -> None:
        """Initializes a design.

        Args:
            name (str): Name of the design.
        """
        self.name = name

    def __hash__(self):
        """Hashes the Design object.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class DesignConfig(BaseModel):
    """Configuration for design verification.

    This class holds configuration parameters for design verification.

    Attributes:
        cpy1 (str): Hierarchy prefix for the first copy of the design.
        cpy2 (str): Hierarchy prefix for the second copy of the design.
        lang (str): Hardware description language used for the design.
        topmod (str): Name of the top module in the design.
        clk (str): Name of the clock signal in the design.
    """

    cpy1: str = "a"
    cpy2: str = "b"
    lang: str = "sv12"
    topmod: str = ""
    clk: str = "clk"


class PYConfig(BaseModel):
    """PyCaliper configuration class.

    This class holds the main configuration for PyCaliper tasks.

    Attributes:
        sdir (str): Directory to save results to.
        mock (bool): Whether to run in mock mode without Jasper access.
        jgc (JasperConfig): Jasper configuration.
        onetrace (bool): Whether to verify only one-trace properties.
        tdir (str): Directory containing VCD trace files.
        dc (DesignConfig): Design configuration.
    """

    # Saving directory
    sdir: str = ""

    # Is this a mock run (without Jasper access)?
    mock: bool = False
    # Jasper configuration
    jgc: JasperConfig = JasperConfig()

    # Use only one trace for verification
    onetrace: bool = False
    # Directory of VCD traces
    tdir: str = ""

    # Design configuration
    dc: DesignConfig = DesignConfig()
