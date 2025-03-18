"""
    Synthesis for equality invariants using Jasper FV Interface
"""

import logging
from dataclasses import dataclass

from ..pycconfig import PYConfig

from ..per import SpecModule, PERHole, Context, Logic
from .iis_strategy import *

from pycaliper.svagen import SVAGen
from pycaliper.jginterface.jgsetup import setup_jasper, JasperConfig
from pycaliper.jginterface.jgoracle import (
    prove,
    prove_out_induction_2t,
    set_assm_induction_2t,
    is_pass,
    enable_assm,
    disable_assm,
    loadscript,
)
from pycaliper.pycconfig import Design, DesignConfig
from pycaliper.verif.jgverifier import JGDesign
from pycaliper.synth.btorsynthesizer import BTORVerifier2TraceIncremental
from pycaliper.verif.btorverifier import BTORDesign


logger = logging.getLogger(__name__)


class SynthesisTree:

    counter = 0

    def __init__(self, asrts=[], assms=[], parent=None, inherits=None):
        self.children: dict[str, SynthesisTree] = {}
        self.parent = parent
        self.inherits = inherits
        self.secondaries = []

        self.asrts: list[str] = list(asrts)
        self.assms: list[str] = assms + ([inherits] if inherits is not None else [])

        self.fuel = len(self.asrts) - len(self.assms)

        self.id = SynthesisTree.counter
        SynthesisTree.counter += 1

        self.checked = False

    def add_child(self, cand):
        self.children[cand] = SynthesisTree(self.asrts, self.assms, self, cand)

    def add_asrt(self, asrt):
        self.asrts.append(asrt)
        self.fuel += 1

    def add_secondary_assm(self, assm):
        if assm not in self.assms:
            self.assms.append(assm)
            self.fuel -= 1
            self.secondaries.append(assm)
            return True
        return False

    def is_self_inductive(self):
        return set(self.assms) == set(self.asrts)

    def __str__(self) -> str:
        return f"synnode::{self.id}(fuel={self.fuel})"


@dataclass
class HoudiniSynthesizerConfig:
    fuelbudget: int = 10
    stepbudget: int = 10
    retries: int = 1


@dataclass
class HoudiniSynthesizerStats:
    solvecalls: int
    steps: int
    minfuel: int
    success: bool


class HoudiniSynthesizer:
    def __init__(self):
        self.strategy = None
        self.synconfig: HoudiniSynthesizerConfig = None
        self.ctx = ""

        self.specmodule: SpecModule = None
        self.des: Design = None
        self.dc: DesignConfig = None

        # Running variables
        self.synstate: SynthesisTree = None
        self.depth: int = 0
        self.minfuel: int = 0
        self.solvecalls: int = 0

        self.candidates: dict[str, Logic] = {}

    def _setup_synthesis(self):
        raise NotImplementedError

    def _restart_synthesis(self):
        raise NotImplementedError

    def _can_add(self, cand) -> bool:
        raise NotImplementedError

    def _enable_assm(self, assm):
        raise NotImplementedError

    def _disable_assm(self, assm):
        raise NotImplementedError

    def _check_safe(self) -> bool:
        raise NotImplementedError

    def _saturate(self):
        added = True
        while added:
            added = False
            for cand in self.candidates:
                if cand not in self.synstate.asrts:
                    self.solvecalls += 1
                    if self._can_add(cand):
                        added = True
                        self.synstate.add_asrt(cand)
                        if self.synstate.add_secondary_assm(cand):
                            self._enable_assm(cand)
                        logger.debug(f"Added assertion {cand} to synthesis node")
                        break

    def _dive(self, cand):
        self.synstate.add_child(cand)
        self.synstate = self.synstate.children[cand]
        logger.debug(f"Dived to new state: {self.synstate} on candidate: {cand}")
        self.depth += 1
        self._enable_assm(cand)
        logger.debug(
            f"Saturating curr. synstate: {self.synstate}, with assms: {self.synstate.assms}"
        )
        self._saturate()
        self.minfuel = min(self.minfuel, self.synstate.fuel)
        logger.debug(
            f"Saturated curr. synstate: {self.synstate}, new assrts: {self.synstate.asrts}"
        )

    def _backtrack(self):
        if self.synstate.parent is not None:
            cand = self.synstate.inherits
            secondaries = self.synstate.secondaries
            self.synstate = self.synstate.parent
            logger.debug(
                f"Backtracked to state: {self.synstate} on "
                + f"inheritance: {cand}, and secondaries: {secondaries}"
            )
            for c in [cand] + secondaries:
                self._disable_assm(c)
            self.depth -= 1
            return True
        else:
            logger.warn(
                f"Cannot backtrack from root state. Synthesis failed: {self.synstate}"
            )
            return False

    def _safe(self):
        if not self.synstate.checked:
            self.synstate.checked = True
            self.solvecalls += 1
            return self._check_safe()
        return False

    def _synthesize(self, candidate_order):
        steps = 0
        while True:
            steps += 1
            if self.synstate.is_self_inductive():
                logger.debug(f"Synthesis node is self-inductive: {self.synstate.asrts}")
                if self._safe():
                    # Done
                    logger.debug(
                        f"Synthesis complete. Found invariant: {self.synstate.asrts}"
                    )
                    return self.synstate.asrts
                else:
                    logger.debug(f"Synthesis node is not safe: {self.synstate.asrts}")
                    unexplored_cands = [
                        cand
                        for cand in candidate_order
                        if cand not in self.synstate.children
                        and cand not in self.synstate.assms
                    ]
                    if unexplored_cands == []:
                        return None
                    else:
                        cand = unexplored_cands[0]
                        if steps < self.synconfig.stepbudget:
                            self._dive(cand)
                        else:
                            return None
            else:
                unexplored_cands = [
                    cand
                    for cand in candidate_order
                    if cand not in self.synstate.children
                    and cand not in self.synstate.assms
                ]
                if unexplored_cands == [] or (
                    self.synstate.fuel + self.synconfig.fuelbudget < 0
                ):
                    if not self._backtrack():
                        return None
                else:
                    cand = unexplored_cands[0]
                    if steps < self.synconfig.stepbudget:
                        self._dive(cand)
                    else:
                        return None

    def synthesize(
        self,
        topmod: SpecModule,
        des: Design,
        dc: DesignConfig,
        strategy: IISStrategy = SeqStrategy(),
        synconfig: HoudiniSynthesizerConfig = HoudiniSynthesizerConfig(),
    ) -> tuple[SpecModule, HoudiniSynthesizerStats]:

        logger.info(f"Using strategy: {strategy.__class__.__name__}")
        self.strategy: IISStrategy = strategy
        self.synconfig: HoudiniSynthesizerConfig = synconfig
        self.solvecalls = 0
        self.des = des
        self.dc: DesignConfig = dc
        assert topmod.is_instantiated(), "Module not instantiated."
        self.specmodule = topmod

        self._setup_synthesis()

        candidate_orders = []

        iter_ = 0
        while iter_ < self.synconfig.retries:
            iter_ += 1

            self._restart_synthesis()

            candidate_order = self.strategy.get_candidate_order(
                list(self.candidates.keys()),
                ctx=self.ctx,
                prev_attempts=candidate_orders,
            )
            logger.debug(f"Got candidate order: {candidate_order}")
            candidate_orders.append(candidate_order)

            invs = self._synthesize(candidate_order)

            if invs is None:
                logger.info(f"Synthesis failed at attempt {iter_}, retrying.")
            else:
                break

        if invs is None:
            logger.info(f"Synthesis failed after {iter_} attempts.")
            stats = HoudiniSynthesizerStats(
                solvecalls=self.solvecalls,
                steps=SynthesisTree.counter,
                minfuel=self.synconfig.fuelbudget + self.minfuel,
                success=False,
            )
        else:
            # Synthesis succeeded
            for c in topmod._pycinternal__perholes:
                c.deactivate()
            for inv in invs:
                topmod._eq(self.candidates[inv], Context.STATE)

            logger.info(self.strategy.get_stats())
            logger.info(
                f"Synthesized invariants {invs} in {iter_} attempts at depth: {self.depth}."
            )
            stats = HoudiniSynthesizerStats(
                solvecalls=self.solvecalls,
                steps=SynthesisTree.counter,
                minfuel=self.synconfig.fuelbudget + self.minfuel,
                success=True,
            )
        return topmod, stats


class HoudiniSynthesizerJG(HoudiniSynthesizer):
    def __init__(self):
        super().__init__()
        self.jgc: JasperConfig = None
        self.svagen: SVAGen = None

    def _setup_synthesis(self):
        assert isinstance(self.des, JGDesign), "Design not of type JGDesign."
        self.jgc: JasperConfig = self.des.pyc.jgc
        setup_jasper(self.specmodule, self.jgc, self.dc)
        self.svagen = SVAGen()
        self.svagen.create_pyc_specfile(
            self.specmodule, filename=self.jgc.pycfile_abspath(), dc=self.dc
        )
        self.candidates = self.svagen.holes
        state_invs = self.svagen.specs[self.svagen.topmod.path].state_spec_decl
        # Strip each line and concatenate
        self.ctx = " ".join([line.strip() for line in state_invs.split("\n")])

    def _restart_synthesis(self):
        # Load the script
        loadscript(self.jgc.script)
        # Enable and disable the right assumptions
        set_assm_induction_2t(self.jgc.context, self.svagen.property_context)

        self.synstate = SynthesisTree()
        self.depth = 0
        self.minfuel = 0

    def _can_add(self, cand) -> bool:
        return is_pass(prove(self.jgc.context, cand))

    def _enable_assm(self, cand):
        enable_assm(self.jgc.context, cand)

    def _disable_assm(self, cand):
        disable_assm(self.jgc.context, cand)

    def _check_safe(self) -> bool:
        return is_pass(prove_out_induction_2t(self.jgc.context))


class HoudiniSynthesizerBTOR(HoudiniSynthesizer):
    def __init__(self, gui=None):
        super().__init__()
        self.verifier = BTORVerifier2TraceIncremental(gui)

    def _setup_synthesis(self):
        assert isinstance(
            self.des, BTORDesign
        ), f"Design not of type BTORDesign, but is {type(self.des)}"
        self.candidates = {
            h.per._get_id_for_per_hole(): h.per.logic
            for h in self.specmodule._pycinternal__perholes
        }

    def _restart_synthesis(self):
        self.verifier = BTORVerifier2TraceIncremental()
        self.verifier.unroll(self.specmodule, self.des, self.dc)

        self.synstate = SynthesisTree()
        self.depth = 0
        self.minfuel = 0

    def _can_add(self, cand):
        # Add clause to engine and check if there is a CEX
        return self.verifier.can_add(cand)

    def _disable_assm(self, assm):
        # Remove clause from engine
        self.verifier.disable_hole_assm(assm)

    def _enable_assm(self, assm):
        # Add clause to engine
        self.verifier.enable_hole_assm(assm)

    def _check_safe(self):
        # Check if there is a CEX
        return self.verifier.check_safe()


class PERSynthesizer:
    def __init__(
        self,
        pyconfig: PYConfig,
        strategy: IISStrategy = SeqStrategy(),
        fuelbudget: int = 3,
        stepbudget: int = 10,
    ) -> None:
        self.pyconfig = pyconfig
        self.svagen = None
        self.candidates: dict[str, PERHole] = {}

        self.fuelbudget = fuelbudget
        self.stepbudget = stepbudget

        self.depth = 0
        self.minfuel = 0
        self.solvecalls = 0
        self.synstate: SynthesisTree = SynthesisTree()

        self.strategy = strategy
        self.state_invs = ""

    def reset_state(self):
        self.synstate = SynthesisTree()
        self.depth = 0
        self.minfuel = 0

    def _saturate(self):
        added = True
        while added:
            added = False
            for cand in self.candidates:
                if cand not in self.synstate.asrts:
                    self.solvecalls += 1
                    if is_pass(prove(self.pyconfig.jgc.context, cand)):
                        added = True
                        self.synstate.add_asrt(cand)
                        if self.synstate.add_secondary_assm(cand):
                            enable_assm(self.pyconfig.jgc.context, cand)
                        logger.debug(f"Added assertion {cand} to synthesis node")
                        break

    def _dive(self, cand):
        self.synstate.add_child(cand)
        self.synstate = self.synstate.children[cand]
        logger.debug(f"Dived to new state: {self.synstate} on candidate: {cand}")
        self.depth += 1
        enable_assm(self.pyconfig.jgc.context, cand)
        logger.debug(
            f"Saturating curr. synstate: {self.synstate}, with assms: {self.synstate.assms}"
        )
        self._saturate()
        self.minfuel = min(self.minfuel, self.synstate.fuel)
        logger.debug(
            f"Saturated curr. synstate: {self.synstate}, new assrts: {self.synstate.asrts}"
        )

    def _backtrack(self):
        if self.synstate.parent is not None:
            cand = self.synstate.inherits
            secondaries = self.synstate.secondaries
            self.synstate = self.synstate.parent
            logger.debug(
                f"Backtracked to state: {self.synstate} on "
                + f"inheritance: {cand}, and secondaries: {secondaries}"
            )
            for c in [cand] + secondaries:
                disable_assm(self.pyconfig.jgc.context, c)
            self.depth -= 1
            return True
        else:
            logger.warn(
                f"Cannot backtrack from root state. Synthesis failed: {self.synstate}"
            )
            return False

    def safe(self):
        if not self.synstate.checked:
            self.synstate.checked = True
            self.solvecalls += 1
            return is_pass(prove_out_induction_2t(self.pyconfig.jgc.context))
        return False

    def _synthesize(self, candidate_order):
        steps = 0
        while True:
            steps += 1
            if self.synstate.is_self_inductive():
                logger.debug(f"Synthesis node is self-inductive: {self.synstate.asrts}")
                if self.safe():
                    # Done
                    logger.debug(
                        f"Synthesis complete. Found invariant: {self.synstate.asrts}"
                    )
                    return self.synstate.asrts
                else:
                    logger.debug(f"Synthesis node is not safe: {self.synstate.asrts}")
                    unexplored_cands = [
                        cand
                        for cand in candidate_order
                        if cand not in self.synstate.children
                        and cand not in self.synstate.assms
                    ]
                    if unexplored_cands == []:
                        return None
                    else:
                        cand = unexplored_cands[0]
                        if steps < self.stepbudget:
                            self._dive(cand)
                        else:
                            return None
            else:
                unexplored_cands = [
                    cand
                    for cand in candidate_order
                    if cand not in self.synstate.children
                    and cand not in self.synstate.assms
                ]
                if unexplored_cands == [] or (self.synstate.fuel + self.fuelbudget < 0):
                    if not self._backtrack():
                        return None
                else:
                    cand = unexplored_cands[0]
                    if steps < self.stepbudget:
                        self._dive(cand)
                    else:
                        return None

    def synthesize(self, topmod: SpecModule, retries: int = 1) -> SpecModule:
        # Create a new SVA generator
        assert topmod.is_instantiated(), "Module not instantiated."

        setup_jasper(topmod, self.pyconfig.jgc, self.pyconfig.dc)
        self.svagen = SVAGen()
        self.svagen.create_pyc_specfile(
            topmod, filename=self.pyconfig.jgc.pycfile_abspath(), dc=self.pyconfig.dc
        )
        self.candidates = self.svagen.holes

        logger.info(f"Using strategy: {self.strategy.__class__.__name__}")
        self.state_invs = self.svagen.specs[self.svagen.topmod.path].state_spec_decl
        # Strip each line and concatenate
        self.state_invs = " ".join(
            [line.strip() for line in self.state_invs.split("\n")]
        )

        candidate_orders = []

        for i in range(retries):
            # Load the script
            loadscript(self.pyconfig.jgc.script)
            # Enable and disable the right assumptions
            set_assm_induction_2t(
                self.pyconfig.jgc.context, self.svagen.property_context
            )

            candidate_order = self.strategy.get_candidate_order(
                list(self.candidates.keys()),
                ctx=self.state_invs,
                prev_attempts=candidate_orders,
            )
            logger.debug(f"Got candidate order: {candidate_order}")
            candidate_orders.append(candidate_order)
            self.reset_state()
            invs = self._synthesize(candidate_order)

            if invs is None:
                # Synthesis failed
                logger.info(
                    f"Invariant synthesis failed at attempt: {i}. Retrying synthesis."
                )

            else:
                logger.info(
                    f"Synthesized invariants: {invs} at depth: {self.depth} and minimum fuel: {self.fuelbudget + self.minfuel}"
                )

                logger.info(self.strategy.get_stats())

                # Disable all eq holes
                for c in topmod._pycinternal__perholes:
                    c.deactivate()
                for inv in invs:
                    topmod._eq(self.candidates[inv], Context.STATE)

                break

        if invs is None:
            logger.info(
                f"Synthesis failed after {retries} attempts with {self.solvecalls} solve calls and {SynthesisTree.counter} steps."
            )
            with open("persyn.log", "a") as f:
                f.write(
                    f"{self.strategy.__class__.__name__},{self.solvecalls},{SynthesisTree.counter},fail\n"
                )
        else:
            logger.info(
                f"Synthesis complete. Synthesized invariants: {invs} in {i+1} attempts with {self.solvecalls} solve calls and {SynthesisTree.counter} steps."
            )
            with open("persyn.log", "a") as f:
                f.write(
                    f"{self.strategy.__class__.__name__},{self.solvecalls},{SynthesisTree.counter},success\n"
                )
        return topmod
