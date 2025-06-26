from pycaliper.proofmanager import mk_btordesign
from pycaliper.verif.btorverifier import BTORVerifier1Trace, BTORVerifierBMC
from pycaliper.pycconfig import DesignConfig
from tests.specs.demo import demo
import logging

logging.basicConfig(
    filename='debug.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Create a logger
logger = logging.getLogger(__name__)

# Create a BTOR design
prgm = mk_btordesign("demo", "examples/designs/demo/btor/full_design.btor")

# Instantiate the demo specification
spec = demo()
spec.instantiate()

# Perform a 1-trace inductive proof
verifier = BTORVerifier1Trace()
result = verifier.verify(spec, prgm, DesignConfig(cpy1="a"))
print("Proof result:", "SAFE" if result.verified else "BUG")

# Perform a 1-trace bmc proof for the "simstep" sequence of constraints
verifier = BTORVerifierBMC()
result = verifier.verify(spec, prgm, DesignConfig(cpy1="a"), "simstep")
print("Proof result:", "SAFE" if result.verified else "BUG")
