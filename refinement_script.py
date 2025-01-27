import logging

from pycaliper.verif.mmrverifier import MMRVerifier
from btor2ex import BoolectorSolver

from myspecs.refinement import FIFO, Counter

# debug to file, info to console
logging.basicConfig(level=logging.DEBUG, filename="debug.log", filemode="w")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


fifo = FIFO()
counter = Counter()

refinement_map = [
    (fifo.push, counter.add),
    (fifo.pop, ~counter.add),
    (fifo.wr_ptr - fifo.rd_ptr, counter.count),
]

MMRVerifier(BoolectorSolver()).check_refinement(fifo, counter, rmap=refinement_map)
