import sys
import logging

from pycaliper.verif.jgverifier import (
    JGVerifier1Trace,
    JGVerifier2Trace,
    JGVerifier1TraceBMC,
)
from pycaliper.synth.persynthesis import PERSynthesizer
from pycaliper.synth.iis_strategy import SeqStrategy, RandomStrategy, LLMStrategy
from pycaliper.svagen import SVAGen
from pycaliper.synth.alignsynthesis import AlignSynthesizer
from pycaliper.pycmanager import start, PYCTask, PYCArgs

import typer
from typer import Argument, Option
from typing_extensions import Annotated

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.INFO)
h1.setFormatter(logging.Formatter("%(levelname)s::%(message)s"))

h2 = logging.FileHandler("debug.log", mode="w")
h2.setLevel(logging.DEBUG)
h2.setFormatter(logging.Formatter("%(asctime)s::%(name)s::%(levelname)s::%(message)s"))

# Add filename and line number to log messages
logging.basicConfig(level=logging.DEBUG, handlers=[h1, h2])

logger = logging.getLogger(__name__)

DESCRIPTION = "Invariant verification and synthesis using Jasper."
app = typer.Typer(help=DESCRIPTION)


@app.command("verif")
def verif_main(
    specpath: Annotated[str, Argument(help="Path to the PYC config file")] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    # Allow using --params
    params: Annotated[
        str, Option(help="Parameters for the spec module: (<key>=<intvalue>)+")
    ] = "",
    # Allow using -s or --sdir
    sdir: Annotated[str, Option(help="Directory to save results to.")] = "",
    # Allow using --onetrace
    onetrace: Annotated[bool, Option(help="Verify only one-trace properties.")] = False,
    # Allow using --bmc
    bmc: Annotated[
        str, Option(help="Perform verification with bounded model checking.")
    ] = "",
):
    args = PYCArgs(
        specpath=specpath,
        jgcpath=jgcpath,
        params=params,
        sdir=sdir,
        onetrace=onetrace,
        bmc=bmc,
    )
    if bmc == "":
        if onetrace:
            pconfig, tmgr, module = start(PYCTask.VERIF1T, args)
            verifier = JGVerifier1Trace(pconfig)
            logger.debug("Running single trace verification.")
        else:
            pconfig, tmgr, module = start(PYCTask.VERIF2T, args)
            verifier = JGVerifier2Trace(pconfig)
            logger.debug("Running two trace verification.")
        verifier.verify(module)
    else:
        pconfig, tmgr, module = start(PYCTask.VERIFBMC, args)
        verifier = JGVerifier1TraceBMC(pconfig)
        logger.debug("Running BMC verification.")
        verifier.verify(module, bmc)


@app.command("persynth")
def persynth_main(
    specpath: Annotated[str, Argument(help="Path to the PYC config file")] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    # Allow using --params
    params: Annotated[
        str, Option(help="Parameters for the spec module: (<key>=<intvalue>)+")
    ] = "",
    # Allow using -s or --sdir
    sdir: Annotated[
        str, Option("-s", "--sdir", help="Directory to save results to.")
    ] = "",
    # Allow using --strategy
    strategy: Annotated[
        str, Option(help="Strategy to use for synthesis ['seq', 'rand', 'llm']")
    ] = "seq",
    # Allow using --fuel
    fuelbudget: Annotated[int, Option(help="Fuel for the synthesis strategy")] = 3,
    # Allow using --retries
    retries: Annotated[int, Option(help="Number of retries for synthesis")] = 1,
    # Allow using --stepbudget
    stepbudget: Annotated[int, Option(help="Step budget for synthesis")] = 10,
):
    args = PYCArgs(specpath=specpath, jgcpath=jgcpath, params=params, sdir=sdir)
    pconfig, tmgr, module = start(PYCTask.PERSYNTH, args)

    match strategy:
        case "seq":
            strat = SeqStrategy()
        case "rand":
            strat = RandomStrategy()
        case "llm":
            strat = LLMStrategy()
        case _:
            logger.warning("Invalid strategy, using default strategy.")
            strat = SeqStrategy()

    synthesizer = PERSynthesizer(pconfig, strat, fuelbudget, stepbudget)
    finalmod = synthesizer.synthesize(module, retries)

    tmgr.save_spec(finalmod)
    tmgr.save()


@app.command("svagen")
def svagen_main(
    specpath: Annotated[str, Argument(help="Path to the PYC config file")] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    # Allow using --params
    params: Annotated[
        str, Option(help="Parameters for the spec module: (<key>=<intvalue>)+")
    ] = "",
    # Allow using -s or --sdir
    sdir: Annotated[str, Option(help="Directory to save results to.")] = "",
):
    args = PYCArgs(specpath=specpath, jgcpath=jgcpath, params=params, sdir=sdir)
    pconfig, tmgr, module = start(PYCTask.SVAGEN, args)

    svagen = SVAGen(module)
    svagen.create_pyc_specfile(filename=pconfig.pycfile)


@app.command("alignsynth")
def alignsynth_main(
    specpath: Annotated[str, Argument(help="Path to the PYC config file")] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    # Allow using --params
    params: Annotated[
        str, Option(help="Parameters for the spec module: (<key>=<intvalue>)+")
    ] = "",
    # Allow using -s or --sdir
    sdir: Annotated[str, Option(help="Directory to save results to.")] = "",
):
    args = PYCArgs(specpath=specpath, jgcpath=jgcpath, params=params, sdir=sdir)

    pconfig, tmgr, module = start(PYCTask.CTRLSYNTH, args)

    synthesizer = AlignSynthesizer(tmgr, pconfig)
    asmod = synthesizer.synthesize(module)

    tmgr.save_spec(asmod)
    tmgr.save()


@app.command("fullsynth")
def fullsynth_main(
    specpath: Annotated[str, Argument(help="Path to the PYC config file")] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    # Allow using --params
    params: Annotated[
        str, Option(help="Parameters for the spec module: (<key>=<intvalue>)+")
    ] = "",
    # Allow using -s or --sdir
    sdir: Annotated[str, Option(help="Directory to save results to.")] = "",
):
    args = PYCArgs(specpath=specpath, jgcpath=jgcpath, params=params, sdir=sdir)
    pconfig, tmgr, module = start(PYCTask.FULLSYNTH, args)

    # PER Synthesizer
    psynth = PERSynthesizer(pconfig)

    verif = JGVerifier1Trace(pconfig)

    # CA Synthesizer
    asynth = AlignSynthesizer(tmgr, pconfig)
    # Align synthesize module, save it and grab a copy
    asmod = asynth.synthesize(module)
    tmgr.save_spec(asmod)

    # Check that invariants pass
    res = verif.verify(asmod)
    if res:
        logger.info("Single trace verification passed, moving on to PER synthesis!")
    else:
        logger.error("Verification failed for single trace properties, quitting!")
        sys.exit(1)

    # PER Synthesize module
    finalmod = psynth.synthesize(asmod)
    tmgr.save_spec(finalmod)
    tmgr.save()


if __name__ == "__main__":
    app()
