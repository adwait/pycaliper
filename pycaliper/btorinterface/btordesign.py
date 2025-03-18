import enum
import hashlib
import dill as pickle
from btoropt import program as prg

from pycaliper.pycconfig import Design


class ClkEdge(enum.Enum):
    # Transitions are triggered on a positive edge (using "clk2fflogic")
    POSEDGE = 1
    # Transitions are triggered on a negative edge (using "clk2fflogic")
    NEGEDGE = 2
    # Clock removal based using "yosys chformal"
    CHFORMAL = 3


class BTORDesign(Design):
    def __init__(
        self, name: str, prgm: list[prg.Instruction], clkedge: ClkEdge = ClkEdge.POSEDGE
    ) -> None:
        self.name = name
        self.prgm = prgm
        self.clkedge = clkedge

    def __hash__(self):
        return hashlib.md5(pickle.dumps(self.prgm)).hexdigest()
