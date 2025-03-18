import sys
import logging

from btoropt import program as prg
from btor2ex import BoolectorSolver, BTOR2Ex

from ..btorinterface.pycbtorsymex import PYCBTORInterface, DesignConfig
from ..verif.btorverifier import BTORDesign
from ..per import SpecModule, Eq, CondEq

logger = logging.getLogger(__name__)


class BTORVerifier2TraceIncremental(PYCBTORInterface):
    def __init__(self, gui=None) -> None:
        super().__init__(gui)

    def _setup_inductive_two_safety_syn(
        self, prog: list[prg.Instruction], specmodule: SpecModule, dc: DesignConfig
    ):

        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.cpy2 = dc.cpy2
        self.specmodule = specmodule
        assert self.specmodule.is_instantiated(), "Module not instantiated."

        if not self.specmodule._pycinternal__perholes:
            logger.error(
                "No holes to fill in a synthesizer, please use a verifier. Exiting."
            )
            sys.exit(1)

        if self.specmodule._pycinternal__caholes:
            logger.error(
                "Ctrl holes not supported in this (BTOR) synthesizer. Exiting."
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

        assert all(
            [isinstance(hol.per, Eq) for hol in self.specmodule._pycinternal__perholes]
        ), "Only Eq holes supported currently"

        (kd, _) = self.specmodule.get_unroll_kind_depths()
        return eq_assms, eq_assrts, condeq_assms, condeq_assrts, kd

    def unroll(self, specmodule: SpecModule, des: BTORDesign, dc: DesignConfig):
        """Synthesizer for inductive two-safety property

        Perform hole-synthesis for a single module w.r.t. the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Returns:
            bool: synthesis result
        """
        prog = des.prgm
        (
            eq_assms,
            eq_assrts,
            condeq_assms,
            condeq_assrts,
            k,
        ) = self._setup_inductive_two_safety_syn(prog, specmodule, dc)

        pers = [h.per for h in specmodule._pycinternal__perholes]

        all_assms = []
        hole_assms: dict[str, list] = {per._get_id_for_per_hole(): [] for per in pers}
        hole_assrts: dict = {per._get_id_for_per_hole(): None for per in pers}
        hole_logics = [per.logic for per in pers]

        for i in range(k):
            # Unroll twice
            self.symex.execute()
            self.symex.execute()

            ind_state = self.symex.state[2 * i]
            # Collect all assumptions
            all_assms.extend(
                self.get_tt_assm_constraints(eq_assms, condeq_assms, ind_state)
            )
            step_hole_assms = self.get_tt_assm_constraints(hole_logics, [], ind_state)
            for per, assm in zip(pers, step_hole_assms):
                hole_assms[per._get_id_for_per_hole()].append(assm)

        final_state = self.symex.state[2 * k - 1]
        all_assrts = self.get_tt_assrt_constraints(
            eq_assrts, condeq_assrts, final_state
        )

        final_hole_assrts = self.get_tt_assrt_constraints(hole_logics, [], final_state)
        for per, assrt in zip(pers, final_hole_assrts):
            hole_assrts[per._get_id_for_per_hole()] = assrt

        clk_assms = self.get_clock_constraints(self.symex.state[:-1])

        self.hole_assms = hole_assms
        self.hole_assrts = hole_assrts
        self.in_assms = all_assms + clk_assms
        self.curr_assms: set[str] = set()
        self.out_assrts = all_assrts
        self.assrt_exprs = eq_assrts + condeq_assrts

    def enable_hole_assm(self, hole_id: str):
        """Enable a hole by adding its assumptions and assertions"""
        self.curr_assms.add(hole_id)

    def disable_hole_assm(self, hole_id: str):
        """Disable a hole by removing its assumptions and assertions"""
        self.curr_assms.remove(hole_id)

    def can_add(self, hole_id: str) -> bool:
        """Check if the hole can be added"""
        for assm in self.in_assms:
            # self.dump_and_wait(assm)
            self.symex.slv.mk_assume(assm)
        for hid in self.curr_assms:
            for assm in self.hole_assms[hid]:
                # self.dump_and_wait(ha)
                self.symex.slv.mk_assume(assm)
        for assmdict in self.symex.prgm_assms:
            for _, assmi in assmdict.items():
                self.symex.slv.mk_assume(assmi)

        self.symex.slv.push()
        # self.dump_and_wait(assrt)
        self.symex.slv.mk_assert(self.hole_assrts[hole_id])
        # self.dump_and_wait(assrt)
        result = self.symex.slv.check_sat()
        logger.debug(
            "For hole assertion %s, result %s with curr_assms %s",
            hole_id,
            "UNSAFE (cannot add)" if result else "SAFE (can add)",
            ";".join(self.curr_assms),
        )
        self.symex.slv.pop()
        return not result

    def check_safe(self) -> bool:
        """Check if the program is safe"""
        for assrt_expr, assrt in zip(self.assrt_exprs, self.out_assrts):
            for assm in self.in_assms:
                # self.dump_and_wait(assm)
                self.symex.slv.mk_assume(assm)
            for hid in self.curr_assms:
                for assm in self.hole_assms[hid]:
                    self.symex.slv.mk_assume(assm)
            for assmdict in self.symex.prgm_assms:
                for _, assmi in assmdict.items():
                    self.symex.slv.mk_assume(assmi)

            self.symex.slv.push()
            # self.dump_and_wait(assrt)
            self.symex.slv.mk_assert(assrt)
            # self.dump_and_wait(assrt)
            result = self.symex.slv.check_sat()
            logger.debug(
                "For out assertion %s, result %s",
                assrt_expr,
                "UNSAFE" if result else "SAFE",
            )
            self.symex.slv.pop()
            if result:
                return False
        return True
