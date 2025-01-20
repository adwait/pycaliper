from btor2ex import BTOR2Ex, BoolectorSolver
from btor2ex.btor2ex.utils import parsewrapper

import btoropt

from pycaliper.per import *
from pycaliper.pycmanager import PYConfig
from pycaliper.verif.btorverifier import BTORVerifier1Trace
from pycaliper.btorinterface.pycbtorsymex import PYCBTORSymex

from myspecs.tage import boundary_spec

import logging

# Log to a debug file with overwriting
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="w")


prgm = btoropt.parse(parsewrapper("designs/tage/tage_predictor.btor"))
# prgm = btoropt.parse(parsewrapper("designs/tage/tage-predictor/btor/full_design.btor"))

# engine = BTOR2Ex(BoolectorSolver(), prgm)

# engine.execute()

pyconfig = PYConfig(clk="clk_i", k=2)

verifier = BTORVerifier1Trace(pyconfig, PYCBTORSymex(pyconfig, BoolectorSolver(), prgm))


# verifier.slv.preprocess()

# print(verifier.slv.names)

result = verifier.verify(boundary_spec())

input("Press Enter to continue...")
print("Verification result: ", "PASS" if result else "FAIL")
