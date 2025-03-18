import logging
import sys

from btoropt import program as prg
from btor2ex import BTOR2Ex, BoolectorSolver
from ..btorinterface.pycbtorsymex import DesignConfig, PYCBTORInterface
from ..per import SpecModule, Eq, CondEq

import hashlib
import dill as pickle

logger = logging.getLogger(__name__)


class Design:
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self):
        raise NotImplementedError


class BTORDesign(Design):
    def __init__(self, name: str, prgm: list[prg.Instruction]) -> None:
        self.name = name
        self.prgm = prgm

    def __hash__(self):
        return hashlib.md5(pickle.dumps(self.prgm)).hexdigest()


class BTORVerifier2Trace(PYCBTORInterface):
    def __init__(self, gui=None) -> None:
        super().__init__(gui)

    def _setup_inductive_two_safety(
        self, prog: list[prg.Instruction], specmodule: SpecModule, dc: DesignConfig
    ):

        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.cpy2 = dc.cpy2
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

        eq_assms = []
        eq_assrts = []
        condeq_assms = []
        condeq_assrts = []

        # Generate the assumptions and assertions
        for p in self.specmodule._pycinternal__input_tt:
            match p:
                case Eq():
                    eq_assms.append(p.logic)
                case CondEq():
                    condeq_assms.append((p.cond, p.logic))
        for p in self.specmodule._pycinternal__state_tt:
            match p:
                case Eq():
                    eq_assms.append(p.logic)
                    eq_assrts.append(p.logic)
                case CondEq():
                    condeq_assms.append((p.cond, p.logic))
                    condeq_assrts.append((p.cond, p.logic))
        for p in self.specmodule._pycinternal__output_tt:
            match p:
                case Eq():
                    eq_assrts.append(p.logic)
                case CondEq():
                    condeq_assrts.append((p.cond, p.logic))

        (kd, _) = self.specmodule.get_unroll_kind_depths()
        logger.debug(f"Performing verification with depth %s", kd)
        return eq_assms, eq_assrts, condeq_assms, condeq_assrts, kd

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ) -> bool:
        """
        Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Returns:
            bool: is SAFE?
        """
        prog = des.prgm
        (
            eq_assms,
            eq_assrts,
            condeq_assms,
            condeq_assrts,
            k,
        ) = self._setup_inductive_two_safety(prog, specmodule, dc)

        logger.debug(
            f"Found eq_assms: {eq_assms}, eq_assrts: {eq_assrts}, condeq_assms: {condeq_assms}, cond_eq_assrts: {condeq_assrts}"
        )

        all_assms = []
        for i in range(k):
            # Unroll 2k twice
            self.symex.execute()
            self.symex.execute()

            ind_state = self.symex.state[2 * i]
            # Collect all assumptions
            all_assms.extend(
                self.get_tt_assm_constraints(eq_assms, condeq_assms, ind_state)
            )

        # Check final state (note that unrolling has one extra state)
        final_state = self.symex.state[2 * k - 1]
        all_assrts = self.get_tt_assrt_constraints(
            eq_assrts, condeq_assrts, final_state
        )

        clk_assms = self.get_clock_constraints(self.symex.state[:-1])

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
                logger.debug("Found a bug")
                model = self.symex.slv.get_model()
                logger.debug("Model:\n%s", model)
                return False
            self.symex.slv.pop()

        logger.debug("No bug found, inductive proof complete")
        # Safe
        return True


class BTORVerifier1Trace(PYCBTORInterface):
    def __init__(self, gui=None):
        super().__init__(gui)

    def _setup_inductive_one_safety(
        self, prog: list[prg.Instruction], specmodule: SpecModule, dc: DesignConfig
    ):
        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.cpy2 = dc.cpy2
        self.specmodule = specmodule
        assert self.specmodule.is_instantiated(), "Module not instantiated."

        if (
            self.specmodule._pycinternal__perholes
            or self.specmodule._pycinternal__caholes
        ):
            logger.warn("Holes found in a verifier, ignoring them.")

        inv_assms = []
        inv_assrts = []

        # Generate the assumptions and assertions
        for p in self.specmodule._pycinternal__input_invs:
            inv_assms.append(p.expr)
        for p in self.specmodule._pycinternal__state_invs:
            inv_assms.append(p.expr)
            inv_assrts.append(p.expr)
        for p in self.specmodule._pycinternal__output_invs:
            inv_assrts.append(p.expr)

        (kd, _) = self.specmodule.get_unroll_kind_depths()
        logger.debug(f"Performing verification with depth %s", kd)
        return inv_assms, inv_assrts, kd

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ) -> bool:
        """
        Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Returns:
            bool: is SAFE?
        """
        prog = des.prgm
        inv_assms, inv_assrts, k = self._setup_inductive_one_safety(
            prog, specmodule, dc
        )
        all_assms = []
        for i in range(k):
            # Unroll 2k times
            self.symex.execute()
            self.symex.execute()

            # Constrain on falling transitions (intra-steps)
            ind_state = self.symex.state[2 * i]
            # Collect all assumptions
            all_assms.extend(self.get_assm_constraints(inv_assms, ind_state))

        # Check final state (note that unrolling has one extra state)
        final_state = self.symex.state[2 * k - 1]
        all_assrts = self.get_assrt_constraints(inv_assrts, final_state)

        # Clocking behaviour
        clk_assms = self.get_clock_constraints(self.symex.state[:-1])

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
                logger.debug("Found a bug")
                model = self.symex.slv.get_model()
                logger.debug("Model:\n%s", model)
                return False
            self.symex.slv.pop()

        logger.debug("No bug found, inductive proof complete")
        # Safe
        return True
