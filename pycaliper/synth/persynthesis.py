"""
    Synthesis for equality invariants using Jasper FV Interface
"""

import logging

from ..pycmanager import PYConfig

from ..per import SpecModule, PERHole, Context
from .iis_strategy import *

from pycaliper.svagen import SVAGen
from pycaliper.jginterface.jgoracle import (
    prove,
    prove_out_induction_2t,
    set_assm_induction_2t,
    is_pass,
    enable_assm,
    disable_assm,
    loadscript,
)


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


class PERSynthesizer:
    def __init__(
        self,
        psconf: PYConfig,
        strategy: IISStrategy = SeqStrategy(),
        fuelbudget: int = 3,
        stepbudget: int = 10,
    ) -> None:
        self.psc = psconf
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
                    if is_pass(prove(self.psc.context, cand)):
                        added = True
                        self.synstate.add_asrt(cand)
                        if self.synstate.add_secondary_assm(cand):
                            enable_assm(self.psc.context, cand)
                        logger.debug(f"Added assertion {cand} to synthesis node")
                        break

    def _dive(self, cand):
        self.synstate.add_child(cand)
        self.synstate = self.synstate.children[cand]
        logger.debug(f"Dived to new state: {self.synstate} on candidate: {cand}")
        self.depth += 1
        enable_assm(self.psc.context, cand)
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
                disable_assm(self.psc.context, c)
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
            return is_pass(prove_out_induction_2t(self.psc.context))
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

        self.svagen = SVAGen()
        self.svagen.create_pyc_specfile(topmod, filename=self.psc.pycfile)
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
            loadscript(self.psc.script)
            # Enable and disable the right assumptions
            set_assm_induction_2t(self.psc.context, self.svagen.property_context)

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
            logger.error(
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
