"""
File: pycaliper/verif/btorverifier.py
This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

"""
This module provides classes for verifying BTOR designs using PyCaliper's verification framework.
"""

import logging
import sys

from btoropt import program as prg
from btor2ex import BTOR2Ex, BoolectorSolver
from ..pycconfig import DesignConfig
from ..btorinterface.pycbtorinterface import PYCBTORInterface, BTORVerifResult
from ..btorinterface.btordesign import BTORDesign
from ..btorinterface.vcdgenerator import write_vcd
from ..per import SpecModule, SimulationSchedule

logger = logging.getLogger(__name__)


class BTORVerifier2Trace(PYCBTORInterface):
    """Verifier for BTOR designs using two-trace induction."""

    def __init__(self, gui=None) -> None:
        """Initialize the BTORVerifier2Trace.

        Args:
            gui: Optional GUI interface.
        """
        super().__init__(gui)

    def _setup_inductive_two_safety(
        self,
        prog: list[prg.Instruction],
        specmodule: SpecModule,
        des: DesignConfig,
        dc: DesignConfig,
    ):
        """Set up the inductive two-safety verification.

        Args:
            prog (list[prg.Instruction]): The program instructions.
            specmodule (SpecModule): The specification module.
            des (DesignConfig): The design configuration.
            dc (DesignConfig): The design configuration.

        Returns:
            tuple: Assumptions and assertions for verification.
        """
        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.cpy2 = dc.cpy2
        self.des = des
        self.specmodule = specmodule
        assert self.specmodule.is_instantiated(), "Module not instantiated."

        if (
            self.specmodule._pycinternal__perholes
            or self.specmodule._pycinternal__caholes
        ):
            logger.error(
                "Holes not supported in a verifier, please use a synthesizer. Exiting."
            )
            sys.exit(1)

        inv_input_assms = []
        inv_state_assms = []
        inv_assrts = []

        # Generate the assumptions and assertions
        for p in self.specmodule._pycinternal__input_invs:
            inv_input_assms.append(p.expr)
        for p in self.specmodule._pycinternal__state_invs:
            inv_state_assms.append(p.expr)
            inv_assrts.append(p.expr)
        for p in self.specmodule._pycinternal__output_invs:
            inv_assrts.append(p.expr)

        tt_input_assms = []
        tt_state_assms = []
        tt_assrts = []

        # Generate the assumptions and assertions
        for p in self.specmodule._pycinternal__input_tt:
            tt_input_assms.append(p)
        for p in self.specmodule._pycinternal__state_tt:
            tt_state_assms.append(p)
            tt_assrts.append(p)
        for p in self.specmodule._pycinternal__output_tt:
            tt_assrts.append(p)

        (kd, _) = self.specmodule.get_unroll_kind_depths()
        logger.debug(f"Performing verification with depth %s", kd + 1)
        return (
            inv_input_assms,
            inv_state_assms,
            inv_assrts,
            tt_input_assms,
            tt_state_assms,
            tt_assrts,
            kd + 1,
        )

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ) -> BTORVerifResult:
        """Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Args:
            specmodule (SpecModule): The specification module.
            des (BTORDesign): The BTOR design.
            dc (DesignConfig): The design configuration.

        Returns:
            BTORVerifResult: The verification result.
        """
        prog = des.prgm
        (
            inv_input_assms,
            inv_state_assms,
            inv_assrts,
            tt_input_assms,
            tt_state_assms,
            tt_assrts,
            k,
        ) = self._setup_inductive_two_safety(prog, specmodule, des, dc)

        all_assms = []
        for i in range(k):
            # Unroll 2k twice
            self._internal_execute()

            # Collect all assumptions
            all_assms.extend(
                self._get_prepost_tt_assm_constraints_at_cycle(tt_input_assms, i)
            )
            all_assms.extend(
                self._get_prepost_assm_constraints_at_cycle(
                    inv_input_assms, i, self.cpy1
                )
            )
            all_assms.extend(
                self._get_prepost_assm_constraints_at_cycle(
                    inv_input_assms, i, self.cpy2
                )
            )

            if i < k - 1:
                all_assms.extend(
                    self._get_prepost_tt_assm_constraints_at_cycle(tt_state_assms, i)
                )
                all_assms.extend(
                    self._get_prepost_assm_constraints_at_cycle(
                        inv_state_assms, i, self.cpy1
                    )
                )
                all_assms.extend(
                    self._get_prepost_assm_constraints_at_cycle(
                        inv_state_assms, i, self.cpy2
                    )
                )
            else:
                all_assms.extend(
                    self._get_pre_tt_assm_constraints_at_cycle(tt_state_assms, i)
                )
                all_assms.extend(
                    self._get_pre_assm_constraints_at_cycle(
                        inv_state_assms, i, self.cpy1
                    )
                )
                all_assms.extend(
                    self._get_pre_assm_constraints_at_cycle(
                        inv_state_assms, i, self.cpy2
                    )
                )

        # Check final state (note that unrolling has one extra state)
        all_assrts = []
        all_assrts.extend(self._get_tt_assrt_constraints_at_cycle(tt_assrts, k - 1))
        all_assrts.extend(
            self._get_assrt_constraints_at_cycle(inv_assrts, k - 1, self.cpy1)
        )
        all_assrts.extend(
            self._get_assrt_constraints_at_cycle(inv_assrts, k - 1, self.cpy2)
        )

        clk_assms = self._get_clock_constraints()

        for assrt in all_assrts:
            for assm in all_assms:
                self.symex.slv.mk_assume(assm)
            for clk_assm in clk_assms:
                self.symex.slv.mk_assume(clk_assm)
            # Apply all internal program assumptions
            for assmdict in self.symex.prgm_assms:
                for _, assmi in assmdict.items():
                    self.symex.slv.mk_assume(assmi)

            self.symex.slv.push()
            self.symex.slv.mk_assert(assrt)
            result = self.symex.slv.check_sat()
            logger.debug(
                "For assertion %s, result %s", assrt, "BUG" if result else "SAFE"
            )
            if result:
                logger.debug("Found a counterexample")
                btor_model = self.symex.get_model()
                logger.debug("Model:\n%s", btor_model)
                vcd_content = write_vcd(btor_model.signals, btor_model.assignments)
                res = BTORVerifResult(False, vcd_content)
                return res
            self.symex.slv.pop()

        logger.debug("No bug found, inductive proof complete")
        # Safe
        return BTORVerifResult(True, None)


class BTORVerifier1Trace(PYCBTORInterface):
    """Verifier for BTOR designs using one-trace induction."""

    def __init__(self, gui=None):
        """Initialize the BTORVerifier1Trace.

        Args:
            gui: Optional GUI interface.
        """
        super().__init__(gui)

    def _setup_inductive_one_safety(
        self,
        prog: list[prg.Instruction],
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ):
        """Set up the inductive one-safety verification.

        Args:
            prog (list[prg.Instruction]): The program instructions.
            specmodule (SpecModule): The specification module.
            des (BTORDesign): The BTOR design.
            dc (DesignConfig): The design configuration.

        Returns:
            tuple: Assumptions and assertions for verification.
        """
        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.cpy2 = dc.cpy2
        self.des = des
        self.specmodule = specmodule
        assert self.specmodule.is_instantiated(), "Module not instantiated."

        if (
            self.specmodule._pycinternal__perholes
            or self.specmodule._pycinternal__caholes
        ):
            logger.warn("Holes found in a verifier, ignoring them.")

        inv_input_assms = []
        inv_state_assms = []
        inv_assrts = []

        # Generate the assumptions and assertions
        for p in self.specmodule._pycinternal__input_invs:
            inv_input_assms.append(p.expr)
        for p in self.specmodule._pycinternal__state_invs:
            inv_state_assms.append(p.expr)
            inv_assrts.append(p.expr)
        for p in self.specmodule._pycinternal__output_invs:
            inv_assrts.append(p.expr)

        (kd, _) = self.specmodule.get_unroll_kind_depths()
        logger.debug(f"Performing verification with depth %s", kd + 1)
        return (inv_input_assms, inv_state_assms, inv_assrts, kd + 1)

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ) -> BTORVerifResult:
        """Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Args:
            specmodule (SpecModule): The specification module.
            des (BTORDesign): The BTOR design.
            dc (DesignConfig): The design configuration.

        Returns:
            BTORVerifResult: The verification result.
        """
        prog = des.prgm
        (
            inv_input_assms,
            inv_state_assms,
            inv_assrts,
            k,
        ) = self._setup_inductive_one_safety(prog, specmodule, des, dc)
        all_assms = []
        for i in range(k):
            # Unroll 2k times
            self._internal_execute()

            # Collect all assumptions
            all_assms.extend(
                self._get_prepost_assm_constraints_at_cycle(
                    inv_input_assms, i, self.cpy1
                )
            )

            if i < k - 1:
                all_assms.extend(
                    self._get_prepost_assm_constraints_at_cycle(
                        inv_state_assms, i, self.cpy1
                    )
                )
            else:
                all_assms.extend(
                    self._get_pre_assm_constraints_at_cycle(
                        inv_state_assms, i, self.cpy1
                    )
                )

        # Check final state (note that unrolling has one extra state)
        all_assrts = self._get_assrt_constraints_at_cycle(inv_assrts, k - 1, self.cpy1)

        # Clocking behaviour
        clk_assms = self._get_clock_constraints()

        for assrt_expr, assrt in zip(inv_assrts, all_assrts):
            for assm in all_assms:
                # self.dump_and_wait(assm)
                self.symex.slv.mk_assume(assm)
            for clk_assm in clk_assms:
                # self.dump_and_wait(clk_assm)
                self.symex.slv.mk_assume(clk_assm)
            for assmdict in self.symex.prgm_assms:
                for _, assmi in assmdict.items():
                    self.symex.slv.mk_assume(assmi)

            self.symex.slv.push()
            # self.dump_and_wait(assrt)
            self.symex.slv.mk_assert(assrt)
            # self.dump_and_wait(assrt)
            result = self.symex.slv.check_sat()
            logger.debug(
                "For assertion %s, result %s", assrt_expr, "BUG" if result else "SAFE"
            )
            if result:
                logger.debug("Found a counterexample")
                btor_model = self.symex.get_model()
                logger.debug("Model:\n%s", btor_model)
                vcd_content = write_vcd(btor_model.signals, btor_model.assignments)
                res = BTORVerifResult(False, vcd_content)
                return res
            self.symex.slv.pop()

        logger.debug("No bug found, inductive proof complete")
        # Safe
        return BTORVerifResult(True, None)

class BTORVerifierBMC(PYCBTORInterface):
    """Verifier for BTOR designs using one-trace BMC."""

    def __init__(self, gui=None):
        """Initialize the BTORVerifierBMC.

        Args:
            gui: Optional GUI interface.
        """
        super().__init__(gui)

    def _setup_bmc(
        self,
        prog: list[prg.Instruction],
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig
    ):
        """Set up the BMC verification.

        Args:
            prog (list[prg.Instruction]): The program instructions.
            specmodule (SpecModule): The specification module.
            des (BTORDesign): The BTOR design.
            dc (DesignConfig): The design configuration.

        Returns:
            tuple: Assumptions and assertions for verification.
        """
        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.des = des
        self.specmodule = specmodule
        assert self.specmodule.is_instantiated(), "Module not instantiated."

        if (
            self.specmodule._pycinternal__perholes
            or self.specmodule._pycinternal__caholes
        ):
            logger.warn("Holes found in a verifier, ignoring them.")

        return
        

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
        sched_str: str
    ) -> BTORVerifResult:
        """Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Args:
            specmodule (SpecModule): The specification module.
            des (BTORDesign): The BTOR design.
            dc (DesignConfig): The design configuration.

        Returns:
            BTORVerifResult: The verification result.
        """
        prog = des.prgm
        self._setup_bmc(prog, specmodule, des, dc)
        # Generate the assumptions and assertions
        simsched = self.specmodule._pycinternal__simsteps[sched_str]
        k = simsched.depth
        logger.debug(f"Performing verification with depth %s", simsched.depth)
    
        assms = []
        asrts = []
        for i in range(k):
            # Unroll 2k times
            self._internal_execute()

            # Collect all assumptions
            assms.append(
                self._get_prepost_assm_constraints_at_cycle(
                    simsched._pycinternal__steps[i]._pycinternal__assume, i, self.cpy1))

            # Check final state (note that unrolling has one extra state)
            asrts.append(
                self._get_assrt_constraints_at_cycle(
                    simsched._pycinternal__steps[i]._pycinternal__assert, i, self.cpy1))

        # Clocking behaviour
        clk_assms = self._get_clock_constraints()

        
        for i in range(k):
            for asrt_expr, asrt in zip(simsched._pycinternal__steps[i]._pycinternal__assert, asrts[i]):
                for j in range(i+1):
                    for assm in assms[j]:
                        # self.dump_and_wait(assm)
                        self.symex.slv.mk_assume(assm)
                for clk_assm in clk_assms:
                    # self.dump_and_wait(clk_assm)
                    self.symex.slv.mk_assume(clk_assm)
                for assmdict in self.symex.prgm_assms:
                    for _, assmi in assmdict.items():
                        self.symex.slv.mk_assume(assmi)

                self.symex.slv.push()
                # self.dump_and_wait(asrt)
                self.symex.slv.mk_assert(asrt)
                # self.dump_and_wait(asrt)
                result = self.symex.slv.check_sat()
                logger.debug(
                    "For assertion %s, at step %s: result %s", asrt_expr, i, "BUG" if result else "SAFE"
                )
                if result:
                    logger.debug("Found a counterexample")
                    btor_model = self.symex.get_model()
                    logger.debug("Model:\n%s", btor_model)
                    vcd_content = write_vcd(btor_model.signals, btor_model.assignments)
                    res = BTORVerifResult(False, vcd_content)
                    return res
                self.symex.slv.pop()

        logger.debug("No bug found, BMC proof complete")
        # Safe
        return BTORVerifResult(True, None)
