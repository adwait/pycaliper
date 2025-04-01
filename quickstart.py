from pycaliper.proofmanager import mk_btordesign
from pycaliper.verif.btorverifier import BTORVerifier1Trace
from pycaliper.pycconfig import DesignConfig
from tests.specs.demo import demo

# Create a BTOR design
prgm = mk_btordesign("demo", "examples/designs/demo/btor/full_design.btor")

# Instantiate the demo specification
spec = demo()
spec.instantiate()

# Perform the proof
verifier = BTORVerifier1Trace()
result = verifier.verify(spec, prgm, DesignConfig(cpy1="a"))
print("Proof result:", "SAFE" if result.verified else "BUG")
