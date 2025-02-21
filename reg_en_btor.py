from btor2ex import BTOR2Ex, BoolectorSolver
from btor2ex.btor2ex.utils import parsewrapper

import btoropt

from pycaliper.per import *
from pycaliper.pycconfig import PYConfig
from pycaliper.verif.btorverifier import BTORVerifier1Trace

from myspecs.regblock import reg_en

import logging

# Log to a debug file with overwriting
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="w")


prgm = btoropt.parse(parsewrapper("designs/regblock/btor/full_design.btor"))

# engine = BTOR2Ex(BoolectorSolver(), prgm)

# engine.execute()

pyconfig = PYConfig()

verifier = BTORVerifier1Trace()


# verifier.slv.preprocess()

# print(verifier.slv.names)

result = verifier.verify(reg_en().instantiate(), prgm, pyconfig.dc)

print("Verification result: ", "PASS" if result else "FAIL")
