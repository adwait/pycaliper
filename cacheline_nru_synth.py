import sys
import logging
from time import time

from pycaliper.pycmanager import PYCArgs, setup_pyc_tmgr_jg
from pycaliper.synth.persynthesis import (
    PERSynthesizer,
    SeqStrategy,
    RandomStrategy,
    LLMStrategy,
    SynthesisTree,
)

from myspecs.cacheline_nru import cacheline_nru

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

NUM_TRIALS = 20


def run_once(k_: int, strategy_: str, fuelbudget_=3, retries_=10, stepbudget_=10):

    args = PYCArgs(
        specpath="myspecs/cacheline_nru",
        jgcpath="designs/dawg/config_nru.json",
        params="MODE=0,k=4",
    )
    is_connected, pyconfig, tmgr = setup_pyc_tmgr_jg(args)

    assert is_connected, "JasperGold is not connected."

    time_start = time()

    cnru = cacheline_nru(MODE=0, k=k_)

    match strategy_:
        case "seq":
            strat = SeqStrategy()
        case "rand":
            strat = RandomStrategy()
        case "llm":
            strat = LLMStrategy()
        case _:
            logger.warning("Invalid strategy, using default strategy.")
            strat = SeqStrategy()

    SynthesisTree.counter = 0
    synthesizer = PERSynthesizer(pyconfig, strat, fuelbudget_, stepbudget_)
    finalmod = synthesizer.synthesize(cnru, retries_)

    tmgr.save_spec(finalmod)
    tmgr.save()

    time_end = time()
    print("Time taken: ", time_end - time_start)
    return finalmod, time_end - time_start


def test_main(k_, strategy_):

    results = []
    for i in range(NUM_TRIALS):
        _, t = run_once(k_, strategy_)
        results.append(t)

    print("Average time taken: ", sum(results) / NUM_TRIALS)
    # Variance
    print(
        "Variance: ",
        sum((t - sum(results) / NUM_TRIALS) ** 2 for t in results) / NUM_TRIALS,
    )


if __name__ == "__main__":
    # Take the BHTWIDTH as an argument from the command line

    strategy = sys.argv[1]
    k = int(sys.argv[2])
    test_main(k, strategy)
