"""
File: pycaliper/synth/persynthesis.py
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

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
from pycaliper.jginterface.jgdesign import JGDesign
from pycaliper.synth.btorsynthesizer import BTORVerifier2TraceIncremental
from pycaliper.btorinterface.btordesign import BTORDesign


logger = logging.getLogger(__name__)


class SynthesisTree:
    """Represents a node in the synthesis tree."""

    counter = 0

    def __init__(self, asrts=[], assms=[], parent=None, inherits=None):
        """Initialize a SynthesisTree node.

        Args:
            asrts (list[str]): Assertions for the node.
            assms (list[str]): Assumptions for the node.
            parent (SynthesisTree, optional): Parent node.
            inherits (str, optional): Inherited assertion.
        """
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
        """Add a child node for a candidate.

        Args:
            cand (str): The candidate to add as a child.
        """
        self.children[cand] = SynthesisTree(self.asrts, self.assms, self, cand)

    def add_asrt(self, asrt):
        """Add an assertion to the node.

        Args:
            asrt (str): The assertion to add.
        """
        self.asrts.append(asrt)
        self.fuel += 1

    def add_secondary_assm(self, assm):
        """Add a secondary assumption to the node.

        Args:
            assm (str): The assumption to add.

        Returns:
            bool: True if the assumption was added, False otherwise.
        """
        if assm not in self.assms:
            self.assms.append(assm)
            self.fuel -= 1
            self.secondaries.append(assm)
            return True
        return False

    def is_self_inductive(self):
        """Check if the node is self-inductive.

        Returns:
            bool: True if the node is self-inductive, False otherwise.
        """
        return set(self.assms) == set(self.asrts)

    def __str__(self) -> str:
        return f"synnode::{self.id}(fuel={self.fuel})"


@dataclass
class HoudiniSynthesizerConfig:
    """Configuration for the Houdini synthesizer."""

    fuelbudget: int = 10
    stepbudget: int = 10
    retries: int = 1


@dataclass
class HoudiniSynthesizerStats:
    """Statistics for the Houdini synthesizer."""

    solvecalls: int
    steps: int
    minfuel: int
    success: bool


class HoudiniSynthesizer:
    """Houdini synthesizer for invariant synthesis."""

    def __init__(self):
        """Initialize the HoudiniSynthesizer."""
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
        """Set up the synthesis process."""
        raise NotImplementedError

    def _restart_synthesis(self):
        """Restart the synthesis process."""
        raise NotImplementedError

    def _can_add(self, cand) -> bool:
        """Check if a candidate can be added.

        Args:
            cand (str): The candidate to check.

        Returns:
            bool: True if the candidate can be added, False otherwise.
        """
        raise NotImplementedError

    def _enable_assm(self, assm):
        """Enable an assumption.

        Args:
            assm (str): The assumption to enable.
        """
        raise NotImplementedError

    def _disable_assm(self, assm):
        """Disable an assumption.

        Args:
            assm (str): The assumption to disable.
        """
        raise NotImplementedError

    def _check_safe(self) -> bool:
        """Check if the current state is safe.

        Returns:
            bool: True if the state is safe, False otherwise.
        """
        raise NotImplementedError

    def _saturate(self):
        """Saturate the current synthesis state."""
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
        """Dive into a new synthesis state.

        Args:
            cand (str): The candidate to dive into.
        """
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
        """Backtrack to the previous synthesis state.

        Returns:
            bool: True if backtracking was successful, False otherwise.
        """
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
        """Check if the current synthesis state is safe.

        Returns:
            bool: True if the state is safe, False otherwise.
        """
        if not self.synstate.checked:
            self.synstate.checked = True
            self.solvecalls += 1
            return self._check_safe()
        return False

    def _synthesize(self, candidate_order):
        """Perform the synthesis process.

        Args:
            candidate_order (list[str]): The order of candidates to synthesize.

        Returns:
            list[str]: The synthesized invariants.
        """
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
        """Synthesize invariants for a specification module.

        Args:
            topmod (SpecModule): The top-level specification module.
            des (Design): The design.
            dc (DesignConfig): The design configuration.
            strategy (IISStrategy, optional): The synthesis strategy.
            synconfig (HoudiniSynthesizerConfig, optional): The synthesis configuration.

        Returns:
            tuple[SpecModule, HoudiniSynthesizerStats]: The synthesized module and statistics.
        """
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
    """Houdini synthesizer for JasperGold."""

    def __init__(self):
        """Initialize the HoudiniSynthesizerJG."""
        super().__init__()
        self.jgc: JasperConfig = None
        self.svagen: SVAGen = None

    def _setup_synthesis(self):
        """Set up the synthesis process for JasperGold."""
        assert isinstance(self.des, JGDesign), "Design not of type JGDesign."
        self.jgc: JasperConfig = self.des.pyc.jgc
        setup_jasper(self.jgc, self.dc, self.specmodule)
        self.svagen = SVAGen()
        self.svagen.create_pyc_specfile(
            self.specmodule, filename=self.jgc.pycfile_abspath(), dc=self.dc
        )
        self.candidates = self.svagen.holes
        state_invs = self.svagen.specs[self.svagen.topmod.path].state_spec_decl
        # Strip each line and concatenate
        self.ctx = " ".join([line.strip() for line in state_invs.split("\n")])

    def _restart_synthesis(self):
        """Restart the synthesis process for JasperGold."""
        # Load the script
        loadscript(self.jgc.script)
        # Enable and disable the right assumptions
        set_assm_induction_2t(self.jgc.context, self.svagen.property_context)

        self.synstate = SynthesisTree()
        self.depth = 0
        self.minfuel = 0

    def _can_add(self, cand) -> bool:
        """Check if a candidate can be added for JasperGold.

        Args:
            cand (str): The candidate to check.

        Returns:
            bool: True if the candidate can be added, False otherwise.
        """
        return is_pass(prove(self.jgc.context, cand))

    def _enable_assm(self, cand):
        """Enable an assumption for JasperGold.

        Args:
            cand (str): The assumption to enable.
        """
        enable_assm(self.jgc.context, cand)

    def _disable_assm(self, cand):
        """Disable an assumption for JasperGold.

        Args:
            cand (str): The assumption to disable.
        """
        disable_assm(self.jgc.context, cand)

    def _check_safe(self) -> bool:
        """Check if the current state is safe for JasperGold.

        Returns:
            bool: True if the state is safe, False otherwise.
        """
        return is_pass(prove_out_induction_2t(self.jgc.context))


class HoudiniSynthesizerBTOR(HoudiniSynthesizer):
    """Houdini synthesizer for BTOR."""

    def __init__(self, gui=None):
        """Initialize the HoudiniSynthesizerBTOR.

        Args:
            gui: Optional GUI interface.
        """
        super().__init__()
        self.verifier = BTORVerifier2TraceIncremental(gui)

    def _setup_synthesis(self):
        """Set up the synthesis process for BTOR."""
        assert isinstance(
            self.des, BTORDesign
        ), f"Design not of type BTORDesign, but is {type(self.des)}"
        self.candidates = {
            h.per._get_id_for_per_hole(): h.per.logic
            for h in self.specmodule._pycinternal__perholes
        }

    def _restart_synthesis(self):
        """Restart the synthesis process for BTOR."""
        self.verifier = BTORVerifier2TraceIncremental()
        self.verifier.unroll(self.specmodule, self.des, self.dc)

        self.synstate = SynthesisTree()
        self.depth = 0
        self.minfuel = 0

    def _can_add(self, cand):
        """Check if a candidate can be added for BTOR.

        Args:
            cand (str): The candidate to check.

        Returns:
            bool: True if the candidate can be added, False otherwise.
        """
        # Add clause to engine and check if there is a CEX
        return self.verifier.can_add(cand)

    def _disable_assm(self, assm):
        """Disable an assumption for BTOR.

        Args:
            assm (str): The assumption to disable.
        """
        # Remove clause from engine
        self.verifier.disable_hole_assm(assm)

    def _enable_assm(self, assm):
        """Enable an assumption for BTOR.

        Args:
            assm (str): The assumption to enable.
        """
        # Add clause to engine
        self.verifier.enable_hole_assm(assm)

    def _check_safe(self):
        """Check if the current state is safe for BTOR.

        Returns:
            bool: True if the state is safe, False otherwise.
        """
        # Check if there is a CEX
        return self.verifier.verify().verified


class PERSynthesizer:
    """PER synthesizer for invariant synthesis."""

    def __init__(
        self,
        pyconfig: PYConfig,
        strategy: IISStrategy = SeqStrategy(),
        fuelbudget: int = 3,
        stepbudget: int = 10,
    ) -> None:
        """Initialize the PERSynthesizer.

        Args:
            pyconfig (PYConfig): The PyCaliper configuration.
            strategy (IISStrategy, optional): The synthesis strategy.
            fuelbudget (int, optional): The fuel budget for synthesis.
            stepbudget (int, optional): The step budget for synthesis.
        """
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
        """Reset the synthesis state."""
        self.synstate = SynthesisTree()
        self.depth = 0
        self.minfuel = 0

    def _saturate(self):
        """Saturate the current synthesis state."""
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
        """Dive into a new synthesis state.

        Args:
            cand (str): The candidate to dive into.
        """
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
        """Backtrack to the previous synthesis state.

        Returns:
            bool: True if backtracking was successful, False otherwise.
        """
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
        """Check if the current synthesis state is safe.

        Returns:
            bool: True if the state is safe, False otherwise.
        """
        if not self.synstate.checked:
            self.synstate.checked = True
            self.solvecalls += 1
            return is_pass(prove_out_induction_2t(self.pyconfig.jgc.context))
        return False

    def _synthesize(self, candidate_order):
        """Perform the synthesis process.

        Args:
            candidate_order (list[str]): The order of candidates to synthesize.

        Returns:
            list[str]: The synthesized invariants.
        """
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
        """Synthesize invariants for a specification module.

        Args:
            topmod (SpecModule): The top-level specification module.
            retries (int, optional): The number of retries for synthesis.

        Returns:
            SpecModule: The synthesized module.
        """
        # Create a new SVA generator
        assert topmod.is_instantiated(), "Module not instantiated."

        setup_jasper(self.pyconfig.jgc, self.pyconfig.dc, topmod)
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
