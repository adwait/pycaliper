from pydantic import BaseModel


class DesignConfig(BaseModel):
    cpy1: str = "a"
    cpy2: str = "b"


class PYConfig(BaseModel):
    """PyCaliper configuration class"""

    # Saving directory
    sdir: str = ""

    # Jasper directory (relative to pycaliper dir)
    jdir: str = ""
    # Is this a mock run (without Jasper access)?
    mock: bool = False
    # Script to load in Jasper (relative to Jasper dir)
    script: str = ""
    # Verification context to use in Jasper
    context: str = ""
    # PyCaliper SVA filepath to use (relative to pycaliper dir)
    pycfile: str = ""
    # Port to use
    port: int = 8080

    # Specification location
    pycspec: str = ""
    # Use only one trace for verification
    onetrace: bool = False
    # Directory of VCD traces
    tdir: str = ""

    # Design configuration
    dc: DesignConfig = DesignConfig()
