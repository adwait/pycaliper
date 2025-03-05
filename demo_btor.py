from pycaliper.proofmanager import ProofManager
from myspecs.demo import demo

import logging

# Log to a debug file with overwriting and line number
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s::%(name)s::%(lineno)s::%(levelname)s::%(message)s",
)

pm = ProofManager(cligui=True)
prgm = pm.mk_btor_design_from_file("designs/demo/btor/full_design.btor", "demo")
spec = pm.mk_spec(demo, "demo_spec")
pr = pm.mk_btor_proof_one_trace(spec, prgm)
