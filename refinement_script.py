import logging

from pycaliper.verif.mmrverifier import RefinementMap
from pycaliper.proofmanager import ProofManager

from myspecs.refinement import FIFO, Counter

# debug to file, info to console
logging.basicConfig(level=logging.DEBUG, filename="debug.log", filemode="w")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

pm = ProofManager(cligui=True)

fifo: FIFO = pm.mk_spec(FIFO, "fifo")
counter: Counter = pm.mk_spec(Counter, "counter")

refinement_map = RefinementMap(
    mappings=[
        (fifo.push, counter.add),
        (fifo.pop, ~counter.add),
        (fifo.wr_ptr - fifo.rd_ptr, counter.count),
    ]
)

pm.check_refinement(fifo, counter, refinement_map)
