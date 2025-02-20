from btor2ex import BTOR2Ex, BoolectorSolver
from btor2ex.btor2ex.utils import parsewrapper

import btoropt

from pycaliper.per import *
from pycaliper.pycmanager import PYConfig
from pycaliper.verif.btorverifier import BTORVerifier1Trace

from myspecs.demo import demo

import logging

# Log to a debug file with overwriting and line number
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s::%(name)s::%(lineno)s::%(levelname)s::%(message)s",
)


prgm = btoropt.parse(parsewrapper("designs/demo/btor/full_design.btor"))

# engine = BTOR2Ex(BoolectorSolver(), prgm)

# engine.execute()

pyconfig = PYConfig()

verifier = BTORVerifier1Trace(pyconfig)


# verifier.slv.preprocess()

# print(verifier.slv.names)

result = verifier.verify(demo().instantiate(), prgm)

print("Verification result: ", "PASS" if result else "FAIL")
