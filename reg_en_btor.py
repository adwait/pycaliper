from pycaliper.per import *
from pycaliper.proofmanager import ProofManager

from myspecs.regblock import reg_en

import logging

# Log to a debug file with overwriting
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="w")

pm = ProofManager(cligui=True)

pm.mk_btor_design_from_file("designs/regblock/btor/full_design.btor", "regblock")
pm.mk_spec(reg_en, "reg_en_spec")
result = pm.mk_btor_proof_one_trace("reg_en_spec", "regblock")

print("Verification result: ", "PASS" if result.result else "FAIL")
