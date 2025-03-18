import sys
import logging

from btoropt import program as prg
from btor2ex import BoolectorSolver, BTOR2Ex

from ..btorinterface.pycbtorinterface import PYCBTORInterface
from ..btorinterface.btordesign import BTORDesign
from ..pycconfig import DesignConfig
from ..per import SpecModule, Eq, CondEq

logger = logging.getLogger(__name__)


class BTORVerifier2TraceIncremental(PYCBTORInterface):
    def __init__(self, gui=None) -> None:
        super().__init__(gui)

    def _setup_inductive_two_safety_syn(
        self,
        prog: list[prg.Instruction],
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ):

        self.symex = BTOR2Ex(BoolectorSolver("btor2"), prog, self.gui)
        self.cpy1 = dc.cpy1
        self.cpy2 = dc.cpy2
        self.des = des
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
        return (
            inv_input_assms,
            inv_state_assms,
            inv_assrts,
            tt_input_assms,
            tt_state_assms,
            tt_assrts,
            kd + 1,
        )

    def unroll(self, specmodule: SpecModule, des: BTORDesign, dc: DesignConfig):
        """Synthesizer for inductive two-safety property

        Perform hole-synthesis for a single module w.r.t. the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq

        Returns:
            bool: synthesis result
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
        ) = self._setup_inductive_two_safety_syn(prog, specmodule, des, dc)

        hole_pers = [h.per for h in specmodule._pycinternal__perholes]
        hole_assms: dict[str, list] = {
            per._get_id_for_per_hole(): [] for per in hole_pers
        }
        hole_assrts: dict = {per._get_id_for_per_hole(): None for per in hole_pers}

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

            if i < k - 1:
                step_hole_assms = self._get_prepost_tt_assm_constraints_at_cycle(
                    hole_pers, i
                )
                for per, assm in zip(hole_pers, step_hole_assms):
                    hole_assms[per._get_id_for_per_hole()].append(assm)
            else:
                step_hole_assms = self._get_pre_tt_assm_constraints_at_cycle(
                    hole_pers, i
                )
                for per, assm in zip(hole_pers, step_hole_assms):
                    hole_assms[per._get_id_for_per_hole()].append(assm)

        all_assrts = []
        all_assrts.extend(self._get_tt_assrt_constraints_at_cycle(tt_assrts, k - 1))
        all_assrts.extend(
            self._get_assrt_constraints_at_cycle(inv_assrts, k - 1, self.cpy1)
        )
        all_assrts.extend(
            self._get_assrt_constraints_at_cycle(inv_assrts, k - 1, self.cpy2)
        )

        final_hole_assrts = self._get_tt_assrt_constraints_at_cycle(hole_pers, k - 1)
        for per, assrt in zip(hole_pers, final_hole_assrts):
            hole_assrts[per._get_id_for_per_hole()] = assrt

        clk_assms = self._get_clock_constraints()

        self.hole_assms = hole_assms
        self.hole_assrts = hole_assrts
        self.in_assms = all_assms + clk_assms
        self.curr_assms: set[str] = set()
        self.out_assrts = all_assrts
        self.assrt_exprs = tt_assrts

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
