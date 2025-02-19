import sys
import logging
from time import time

from pycaliper.pycmanager import PYCArgs, setup_pyc_tmgr_jg
from pycaliper.verif.jgverifier import JGVerifier1Trace

from myspecs.tage import boundary_spec, tage_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_main(bw):

    BHTWIDTH = bw
    TAGEWIDTH = BHTWIDTH - 2

    args = PYCArgs(
        specpath="myspecs/tage.boundary_spec",
        jgcpath="designs/tage/config_boundary.json",
    )
    is_conn, pyconfig, tmgr = setup_pyc_tmgr_jg(args)

    time_start = time()

    tage_conf = tage_config(BHT_IDX_WIDTH=BHTWIDTH, TAGE_IDX_WIDTH=TAGEWIDTH)

    specmodule = boundary_spec(config=tage_conf)

    verifier = JGVerifier1Trace(pyconfig)
    logger.debug("Running two trace verification.")

    verifier.verify(specmodule)

    time_end = time()
    print("Time taken: ", time_end - time_start)


if __name__ == "__main__":
    # Take the BHTWIDTH as an argument from the command line
    bw = int(sys.argv[1])
    test_main(bw)
