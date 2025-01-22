import logging
import sys
import copy
from tqdm import tqdm
import random

from ..per import Module, Logic, Const
from ..per.per import unroll, SimulationStep

from vcdvcd import VCDVCD
from pycaliper.vcdutils import XVALUE, get_subtrace

from ..pycmanager import PYConfig
from ..vcdutils import get_subtrace
from ..pycmanager import PYCManager

from ..verif.jgverifier import JGVerifier1TraceBMC

from ..propns import TOP_STEP_PROP, get_as_prop

from ..jginterface.jgoracle import prove, is_pass, create_vcd_trace

logger = logging.getLogger(__name__)


class SamplerModule(Module):
    def __init__(self, pyconf: PYConfig, **kwargs) -> None:
        super().__init__(name="", **kwargs)
        self.pyconf = pyconf
        self.tracepaths = []

    def add_trace_file(self, path: str):
        self.tracepaths.append(path)

    def shim_simstep(self, i: int):
        if i == self.pyconf.k - 1:
            self.pycassert(Const(0, 1))

    def simstep(self):
        for i in range(self.pyconf.k):
            self._pycinternal__simstep = SimulationStep()
            self.shim_simstep(i)
            self._pycinternal__simsteps.append(
                copy.deepcopy(self._pycinternal__simstep)
            )


class RandomDeviationSampler(SamplerModule):
    """A sampler module that samples traces, while randomly enforcing deviation from the set of previously collected traces"""

    def __init__(self, pyconf: PYConfig, signals: list[Logic], **kwargs) -> None:
        super().__init__(pyconf, **kwargs)
        self.prev_traces: list[list[dict[Logic, Const]]] = []
        self.signals = signals
        self.picked = False

    def add_trace(self, vcdpath: str):
        vcdobj: VCDVCD = VCDVCD(vcdpath)
        # Get the subtrace
        trace = get_subtrace(vcdobj, self.signals, range(0, self.pyconf.k), self.pyconf)
        trace_w_consts = [
            {k: Const(v.val, k.width) for k, v in step.items() if v != XVALUE}
            for step in trace
        ]
        self.prev_traces.append(trace_w_consts)

    def free_traces(self):
        self.prev_traces = []

    def shim_simstep(self, i: int):
        if i == self.pyconf.k - 1:
            self.pycassert(Const(0, 1))
            return
        if self.picked or len(self.prev_traces) == 0:
            return
        else:
            if random.choice([0, 1]):
                self.picked = True
                trace = random.choice(self.prev_traces)
                step = trace[i]
                for key, val in step.items():
                    self.pycassume(key != val)

    def simstep(self) -> None:
        self.picked = False
        for i in range(self.pyconf.k):
            self._pycinternal__simstep = SimulationStep()
            self.shim_simstep(i)
            self._pycinternal__simsteps.append(
                copy.deepcopy(self._pycinternal__simstep)
            )


class TraceSampler:
    def __init__(self, tmgr: PYCManager, pyconf: PYConfig, sm: SamplerModule) -> None:
        self.pyconf = pyconf
        self.tmgr = tmgr
        self.sm = sm
        self.tgprop = (
            f"{self.pyconf.context}.{get_as_prop(TOP_STEP_PROP(self.pyconf.k-1))}"
        )
        self.coverengine = JGVerifier1TraceBMC(self.pyconf)

    def new_trace(self) -> str:
        """Generate a simulation VCD trace from the design

        Returns:
            str: path to the generated VCD trace
        """
        if self.pyconf.mock:
            logger.info("Mock mode enabled, no new traces can be generated.")
            sys.exit(1)

        # Check property
        res = self.coverengine.verify(self.sm)
        if is_pass(res):
            logger.error("Property is SAFE, no traces!")
            sys.exit(1)

        # Grab the trace
        vcd_path = self.tmgr.create_vcd_path()
        res = create_vcd_trace(self.tgprop, vcd_path)
        logger.debug(f"Trace generated at {vcd_path}.")

        # Potentially update the sampler module instance
        self.sm.add_trace_file(vcd_path)
        return vcd_path

    def get_t_traces(self, t):
        """Generate t traces"""
        for _ in tqdm(range(t), desc="Generating traces"):
            self.new_trace()
