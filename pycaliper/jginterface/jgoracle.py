import enum
import logging
import os

from . import jasperclient as jgc
from ..svagen import SVAContext
from ..propns import TOP_STEP_PROP, TOP_SEQ_PROP

logger = logging.getLogger(__name__)


class ProofResult(enum.Enum):
    NONE = 0
    CEX = 1
    SAFE = 2
    PROVEN = 3
    MAX_TRACE_LENGTH = 4
    UNKNOWN = 5
    SIM = 6

    def __str__(self):
        return self.name


def prove(taskcon: str, prop: str) -> ProofResult:
    """Prove a property

    Args:
        taskcon (str): proof node the property is defined under
        prop (str): property name

    Returns:
        ProofResult: result of the proof
    """
    prop_wctx = get_wctx(taskcon, f"P_{prop}")
    logger.debug(f"Proving property: {prop_wctx}")
    cmd = f"prove -property {{ {prop_wctx} }}"
    res: str = jgc.eval(cmd)
    logger.debug(f"Proving property: {prop_wctx} returned {res}")
    return ProofResult[res.upper()]


def is_pass(res: ProofResult) -> bool:
    """Is the result a pass"""
    return res in [ProofResult.SAFE, ProofResult.MAX_TRACE_LENGTH, ProofResult.PROVEN]


def get_wctx(taskcon: str, prop: str) -> str:
    """Get the hierarchical assumption name for a property wire

    Args:
        taskcon (str): proof node name under which the property is defined
        prop (str): property name

    Returns:
        str: hierarchical assumption name
    """
    return f"{taskcon}.{prop}"


def disable_assm(taskcon: str, assm: str):
    """Disable an assumption

    Args:
        taskcon (str): proof node name
        assm (str): assumption name

    Returns:
        _type_: result of the JasperGold command
    """
    assm_wctx = get_wctx(taskcon, f"A_{assm}")
    logger.debug(f"Disabling assumption: {assm_wctx}")
    cmd = f"assume -disable {assm_wctx}"
    res = jgc.eval(cmd)
    logger.debug(f"Disabling assumption: {assm_wctx} returned {res}")
    return res


def enable_assm(taskcon: str, assm: str):
    """Enable an assumption

    Args:
        taskcon (str): proof node name
        assm (str): assumption name

    Returns:
        _type_: result of the JasperGold command
    """
    assm_wctx = get_wctx(taskcon, f"A_{assm}")
    logger.debug(f"Enabling assumption: {assm_wctx}")
    cmd = f"assume -enable {assm_wctx}"
    res = jgc.eval(cmd)
    logger.debug(f"Enabling assumption: {assm_wctx} returned {res}")
    return res


def disable_all_bmc(taskcon: str, svacon: SVAContext):
    """Disable all BMC assumptions"""
    for sched in svacon.assms_bmc:
        for assm in svacon.assms_bmc[sched]:
            disable_assm(taskcon, assm)


def set_assm_induction_1t(taskcon: str, svacon: SVAContext):
    """Enable only 1-trace assumptions (required for 1 trace properties)

    Args:
        taskcon (str): proof node name
    """
    for cand in svacon.holes:
        disable_assm(taskcon, cand)
    for assm in svacon.assms_2trace:
        disable_assm(taskcon, assm)
    for assm in svacon.assms_1trace:
        enable_assm(taskcon, assm)
    disable_all_bmc(taskcon, svacon)


def set_assm_induction_2t(taskcon: str, svacon: SVAContext):
    """Enable all assumptions required for 2 trace properties

    Args:
        taskcon (str): proof node name
    """
    # Disable all holes in the specification
    for cand in svacon.holes:
        disable_assm(taskcon, cand)
    for assm in svacon.assms_2trace:
        enable_assm(taskcon, assm)
    for assm in svacon.assms_1trace:
        enable_assm(taskcon, assm)
    disable_all_bmc(taskcon, svacon)


def set_assm_bmc(taskcon: str, svacon: SVAContext, sched: str):
    """Enable all assumptions required for 1 BMC trace properties"""
    # Disable all holes
    for cand in svacon.holes:
        disable_assm(taskcon, cand)
    for assm in svacon.assms_2trace:
        disable_assm(taskcon, assm)
    for assm in svacon.assms_1trace:
        disable_assm(taskcon, assm)
    for sched_ in svacon.assms_bmc:
        if sched == sched_:
            for assm in svacon.assms_bmc[sched_]:
                enable_assm(taskcon, assm)
        else:
            for assm in svacon.assms_bmc[sched_]:
                disable_assm(taskcon, assm)


def prove_out_induction_1t(taskcon) -> ProofResult:
    return prove(taskcon, "output_inv")


def prove_out_induction_2t(taskcon) -> ProofResult:
    return prove(taskcon, "output")


def prove_out_bmc(taskcon, svacon: SVAContext, sched: str) -> list[ProofResult]:
    results = []
    for i in range(len(svacon.asrts_bmc[sched])):
        results.append(prove(taskcon, TOP_STEP_PROP(sched, i)))
    return results


def prove_seq(taskcon, svacon: SVAContext, sched: str) -> ProofResult:
    return prove(taskcon, TOP_SEQ_PROP(sched))


def loadscript(script):
    # Get pwd
    cmd = f"include {script}"
    logger.info(f"Loading Jasper script: {cmd}")
    res = jgc.eval(cmd)
    return res


def create_vcd_trace(prop, filepath):
    traceoptcmd = "set_trace_optimization standard"
    windowcmd = f"visualize -violation -property {prop} -window visualize:trace"
    tracecmd = f"visualize -save -force -vcd {filepath} -window visualize:trace"
    logger.debug(f"Creating VCD trace for property: {prop}")
    traceoptres = jgc.eval(traceoptcmd)
    windowres = jgc.eval(windowcmd)
    traceres = jgc.eval(tracecmd)
    logger.debug(
        f"Creating VCD trace for property: {prop} returned {traceoptres}; {windowres}; {traceres}"
    )
    return


def setjwd(jwd):
    # Change the Jasper working directory
    pwd = os.getcwd()
    cmd = f"cd {pwd}/{jwd}"
    res = jgc.eval(cmd)
    logger.debug(f"Changing Jasper working directory to {jwd} returned {res}")
    return res


def set_trace_length(k: int) -> str:
    """Set the trace length for the proof.

    Args:
        k (int): Trace length.

    Returns:
        str: Trace length string.
    """
    res = jgc.eval(f"set_max_trace_length {k}")
    logger.debug(f"Setting trace length to {k}")
    return res


# def create_auxreg(name, width):
#     connectcmd = f"connect -bind auxreg {name} -parameter WIDTH {width}"
#     elabcmd = f"connect -elaborate"
#     connectres = jgc.eval(connectcmd)
#     elabres = jgc.eval(elabcmd)
#     logger.debug(f"Creating auxilliary register {name} of width {width} returned {connectres}; {elabres}")
#     return
