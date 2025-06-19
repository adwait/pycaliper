"""
File: pycaliper/pycmain.py

This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

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
from pycaliper.pycmanager import (
    start,
    PYCTask,
    PYCArgs,
    get_jgconfig,
    get_designconfig,
    mock_or_connect,
)
from pycaliper.jginterface.hierarchyexplorer import HierarchyExplorer
from pycaliper.jginterface.jgsetup import setup_jasperharness

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

DESCRIPTION = "PyCaliper: Specification Synthesis and Verification Infrastructure."
app = typer.Typer(help=DESCRIPTION)


@app.command("verif")
def verif_main(
    specpath: Annotated[
        str, Argument(help="Path to the PyCaliper specification class")
    ] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    dcpath: Annotated[
        str, Option("-d", "--dc", help="Path to the design configuration file")
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
    """Verify invariants in a PyCaliper specification.

    This function performs formal verification of the properties specified in a PyCaliper
    specification class. It can perform either one-trace or two-trace verification,
    and supports bounded model checking.

    Args:
        specpath (str): Path to the PyCaliper specification class.
        jgcpath (str): Path to the Jasper configuration file.
        dcpath (str): Path to the design configuration file.
        params (str): Parameters for the specification module in the format key=value.
        sdir (str): Directory to save verification results to.
        onetrace (bool): If True, verify only one-trace properties.
        bmc (str): If provided, perform verification with bounded model checking.
    """
    args = PYCArgs(
        specpath=specpath,
        jgcpath=jgcpath,
        dcpath=dcpath,
        params=params,
        sdir=sdir,
        onetrace=onetrace,
        bmc=bmc,
    )
    if bmc == "":
        if onetrace:
            pyconfig, tmgr, module = start(PYCTask.VERIF1T, args)
            verifier = JGVerifier1Trace()
            logger.debug("Running single trace verification.")
        else:
            pyconfig, tmgr, module = start(PYCTask.VERIF2T, args)
            verifier = JGVerifier2Trace()
            logger.debug("Running two trace verification.")
        module.instantiate()
        verifier.verify(module, pyconfig)
    else:
        pyconfig, tmgr, module = start(PYCTask.VERIFBMC, args)
        verifier = JGVerifier1TraceBMC()
        logger.debug("Running BMC verification.")
        module.instantiate()
        verifier.verify(module, pyconfig, bmc)


@app.command("persynth")
def persynth_main(
    specpath: Annotated[
        str, Argument(help="Path to the PyCaliper specification class")
    ] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    dcpath: Annotated[
        str, Option("-d", "--dc", help="Path to the design configuration file")
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
    """Synthesize invariants using Partial Equivalence Relations (PER).

    This function performs synthesis of invariants for a PyCaliper specification
    using the PER synthesis approach. It supports different synthesis strategies
    and configurable parameters for the synthesis process.

    Args:
        specpath (str): Path to the PyCaliper specification class.
        jgcpath (str): Path to the Jasper configuration file.
        dcpath (str): Path to the design configuration file.
        params (str): Parameters for the specification module in the format key=value.
        sdir (str): Directory to save synthesis results to.
        strategy (str): Strategy to use for synthesis ('seq', 'rand', or 'llm').
        fuelbudget (int): Fuel budget for the synthesis strategy.
        retries (int): Number of retries for synthesis.
        stepbudget (int): Step budget for synthesis.
    """
    args = PYCArgs(
        specpath=specpath, jgcpath=jgcpath, dcpath=dcpath, params=params, sdir=sdir
    )
    pyconfig, tmgr, module = start(PYCTask.PERSYNTH, args)

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

    synthesizer = PERSynthesizer(pyconfig, strat, fuelbudget, stepbudget)
    module.instantiate()
    finalmod = synthesizer.synthesize(module, retries)

    tmgr.save_spec(finalmod)
    tmgr.save()


@app.command("svagen")
def svagen_main(
    specpath: Annotated[
        str, Argument(help="Path to the PyCaliper specification class")
    ] = "",
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    dcpath: Annotated[
        str, Option("-d", "--dc", help="Path to the design configuration file")
    ] = "",
    # Allow using --params
    params: Annotated[
        str, Option(help="Parameters for the spec module: (<key>=<intvalue>)+")
    ] = "",
    # Allow using -s or --sdir
    sdir: Annotated[str, Option(help="Directory to save results to.")] = "",
):
    """Generate SystemVerilog Assertions (SVA) from a PyCaliper specification.

    This function generates SVA specifications from a PyCaliper specification class.
    The generated SVA can be used for formal verification in SystemVerilog environments.

    Args:
        specpath (str): Path to the PyCaliper specification class.
        jgcpath (str): Path to the Jasper configuration file.
        dcpath (str): Path to the design configuration file.
        params (str): Parameters for the specification module in the format key=value.
        sdir (str): Directory to save the generated SVA file to.
    """
    args = PYCArgs(
        specpath=specpath, jgcpath=jgcpath, dcpath=dcpath, params=params, sdir=sdir
    )
    pyconfig, tmgr, module = start(PYCTask.SVAGEN, args)
    module.instantiate()
    svagen = SVAGen()
    svagen.create_pyc_specfile(
        module, filename=pyconfig.jgc.pycfile_abspath(), dc=pyconfig.dc
    )


@app.command("genhierarchy")
def genhierarchy_main(
    # Allow providing a configuration for Jasper
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    dcpath: Annotated[
        str, Option("-d", "--dc", help="Path to the design configuration file")
    ] = "",
    # Allow using -s or --sdir
    path: Annotated[str, Option(help="File to save results to.")] = "",
    # Root module
    root: Annotated[str, Option(help="Root module name.")] = "",
    # Hierarchy exploration depth
    depth: Annotated[
        int,
        Option(help="Depth of hierarchy exploration. Default is full exploration (-1)"),
    ] = -1,
):
    args = PYCArgs(
        specpath="",
        jgcpath=jgcpath,
        dcpath="",
        params="",
        sdir="",
    )
    jgconfig = get_jgconfig(jgcpath)
    dc = get_designconfig(dcpath)
    mock_or_connect(False, jgconfig.port)
    dexplorer = HierarchyExplorer(jgconfig, dc)
    mod = dexplorer.generate_skeleton(root, depth)
    # Save the module
    with open(path, "x") as f:
        f.write(mod.full_repr())
    logger.info(f"Specification written to {path}.")


@app.command("jasperharness")
def jasperharness_main(
    jgcpath: Annotated[
        str, Option("-j", "--jgc", help="Path to the Jasper config file")
    ] = "",
    dcpath: Annotated[
        str, Option("-d", "--dc", help="Path to the design configuration file")
    ] = "",
):
    args = PYCArgs(
        specpath="",
        jgcpath=jgcpath,
        dcpath=dcpath,
        params="",
        sdir="",
    )
    jgconfig = get_jgconfig(args.jgcpath)
    dc = get_designconfig(args.dcpath)
    setup_jasperharness(jgconfig, dc)


def main():
    """Main entry point for the PyCaliper command-line interface.

    This function initializes and runs the Typer application that provides
    the command-line interface for PyCaliper.
    """
    app()


if __name__ == "__main__":
    main()
