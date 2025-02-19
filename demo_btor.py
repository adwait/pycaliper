from btor2ex import BTOR2Ex, BoolectorSolver
from btor2ex.btor2ex.utils import parsewrapper

import btoropt

from pycaliper.per import *
from pycaliper.pycmanager import PYConfig
from pycaliper.verif.btorverifier import BTORVerifier1Trace
from pycaliper.btorinterface.pycbtorsymex import PYCBTORSymex

from myspecs.demo import demo

import logging

# Log to a debug file with overwriting
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="w")


prgm = btoropt.parse(parsewrapper("designs/demo/btor/full_design.btor"))

# engine = BTOR2Ex(BoolectorSolver(), prgm)

# engine.execute()

pyconfig = PYConfig()

verifier = BTORVerifier1Trace(pyconfig, PYCBTORSymex(pyconfig, BoolectorSolver(), prgm))


# verifier.slv.preprocess()

# print(verifier.slv.names)

result = verifier.verify(demo())

print("Verification result: ", "PASS" if result else "FAIL")
