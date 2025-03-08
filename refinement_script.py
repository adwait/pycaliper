import logging

from pycaliper.verif.refinementverifier import RefinementMap
from pycaliper.proofmanager import ProofManager

from myspecs.refinement import FIFO, Counter
from specs.refiner_modules import refiner_module1, refiner_module2

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

pm.check_mm_refinement(fifo, counter, refinement_map)

refmod1: refiner_module1 = pm.mk_spec(refiner_module1, "refmod1")
refmod2: refiner_module2 = pm.mk_spec(refiner_module2, "refmod2")

pm.check_ss_refinement(refmod1, refmod1.simsched1, refmod1.simsched2)
pm.check_ss_refinement(refmod2, refmod2.simsched1, refmod2.simsched2, True)

from time import sleep

sleep(5)
