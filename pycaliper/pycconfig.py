from pydantic import BaseModel


class JasperConfig(BaseModel):
    jdir: str = ""
    script: str = ""
    pycfile: str = ""
    context: str = ""
    design_list: str = "design.lst"
    port: int = 8080

    def pycfile_abspath(self):
        return f"{self.jdir}/{self.pycfile}"


class Design:
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self):
        raise NotImplementedError


class DesignConfig(BaseModel):
    cpy1: str = "a"
    cpy2: str = "b"
    lang: str = "sv12"
    topmod: str = ""
    clk: str = "clk"


class PYConfig(BaseModel):
    """PyCaliper configuration class"""

    # Saving directory
    sdir: str = ""

    # Is this a mock run (without Jasper access)?
    mock: bool = False
    # Jasper configuration
    jgc: JasperConfig = JasperConfig()

    # Specification location
    pycspec: str = ""
    # Use only one trace for verification
    onetrace: bool = False
    # Directory of VCD traces
    tdir: str = ""

    # Design configuration
    dc: DesignConfig = DesignConfig()
