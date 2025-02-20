import sys
import logging
from btor2ex import BoolectorSolver
from btor2ex.btor2ex.utils import parsewrapper

import btoropt

from pycaliper.per import *
from pycaliper.pycmanager import PYConfig
from pycaliper.verif.btorverifier import BTORVerifier1Trace

from myspecs.tage import boundary_spec, tage_config

# Log to a debug file with overwriting
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="w")


def test_main(bw):
    BHTWIDTH = bw
    TAGEWIDTH = BHTWIDTH - 2

    # prgm = btoropt.parse(parsewrapper("designs/tage/tage_predictor.btor"))
    prgm = btoropt.parse(
        parsewrapper(
            f"designs/tage/tage-predictor/btor/full_design_{BHTWIDTH}_{TAGEWIDTH}.btor"
        )
    )

    pyconfig = PYConfig()

    verifier = BTORVerifier1Trace(pyconfig)

    tage_conf = tage_config(BHT_IDX_WIDTH=BHTWIDTH, TAGE_IDX_WIDTH=TAGEWIDTH)

    # print(tage_conf.BHT_IDX_WIDTH)

    result = verifier.verify(boundary_spec(config=tage_conf).instantiate(), prgm)

    print(
        f"Verification result for {BHTWIDTH} {TAGEWIDTH}: ",
        "PASS" if result else "FAIL",
    )


if __name__ == "__main__":
    # Take the BHTWIDTH as an argument from the command line
    bw = int(sys.argv[1])
    test_main(bw)
