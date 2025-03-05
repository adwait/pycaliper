import sys
import logging
from pycaliper.proofmanager import ProofManager
from myspecs.tage import boundary_spec, tage_config

# Log to a debug file with overwriting
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="w")


def test_main(bw):
    bw = bw
    tw = bw - 2

    pm = ProofManager(cligui=True)
    pm.mk_btor_design_from_file(
        f"designs/tage/tage-predictor/btor/full_design_{bw}_{tw}.btor",
        f"tage_{bw}_{tw}",
    )
    pm.mk_spec(
        boundary_spec,
        "tage_boundary_spec",
        config=tage_config(BHT_IDX_WIDTH=bw, TAGE_IDX_WIDTH=tw),
    )
    pm.mk_btor_proof_one_trace("tage_boundary_spec", f"tage_{bw}_{tw}")


if __name__ == "__main__":
    # Take the BHTWIDTH as an argument from the command line
    bw = int(sys.argv[1])
    test_main(bw)
