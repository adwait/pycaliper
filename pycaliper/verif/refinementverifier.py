"""
File: pycaliper/verif/refinementverifier.py
This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

# SpecModule to module refinement verification

import logging
import sys

from dataclasses import dataclass
from typing import Callable

from btor2ex import BTORSolver, BTORSort, BoolectorSolver
from pycaliper.per import Expr as PYCExpr
import pycaliper.per.expr as pycexpr
from pycaliper.per import Logic, SpecModule


logger = logging.getLogger(__name__)


@dataclass
class RefinementMap:
    """Data class for refinement mappings."""

    mappings: list[tuple[PYCExpr, PYCExpr]]


@dataclass
class FunctionalRefinementMap(RefinementMap):
    """Data class for functional refinement mappings."""

    mappings: list[tuple[Logic, PYCExpr]]


class RefinementVerifier:
    """Verifier for module-to-module and bounded simulation refinements."""

    def __init__(self, slv: BTORSolver = BoolectorSolver()):
        """Initialize the RefinementVerifier.

        Args:
            slv (BTORSolver): The BTOR solver to use.
        """
        self.slv = slv
        self.oplut = slv.oplut()
        # Variable map (dynamic and needs to be reset for each check)
        self.varmap = {}

    def reset(self):
        """Reset the variable map."""
        self.varmap = {}

    def convert_expr_to_btor2(self, expr: PYCExpr, bindinst: str, step=0):
        """Convert a PyCaliper expression to a BTOR2 expression.

        Args:
            expr (PYCExpr): The expression to convert.
            bindinst (str): The binding instance.
            step (int): The step index.

        Returns:
            BTOR2 expression.
        """
        logger.debug("Converting expression %s", expr)

        if isinstance(expr, Logic):
            if f"{bindinst}_{expr.name}_{step}" in self.varmap:
                return self.varmap[f"{bindinst}_{expr.name}_{step}"]

            logging.debug("Creating variable %s with width %d", expr.name, expr.width)
            var = self.slv.mk_var(
                f"{bindinst}_{expr.name}_{step}", BTORSort(expr.width)
            )
            self.varmap[f"{bindinst}_{expr.name}_{step}"] = var
            return var

        match expr:
            case pycexpr.OpApply(op=op, args=args):
                operands = [
                    self.convert_expr_to_btor2(arg, bindinst, step) for arg in args
                ]
                match op:
                    case pycexpr.LogicalAnd() | pycexpr.BinaryAnd():
                        return self.oplut["and"](*operands)
                    case pycexpr.LogicalOr() | pycexpr.BinaryOr():
                        return self.oplut["or"](*operands)
                    case pycexpr.BinaryXor():
                        return self.oplut["xor"](*operands)
                    case pycexpr.UnaryLogicalNot() | pycexpr.UnaryBitwiseNot():
                        return self.oplut["not"](*operands)
                    case pycexpr.LogicalShiftLeft():
                        return self.oplut["sll"](*operands)
                    case pycexpr.LogicalShiftRight():
                        return self.oplut["srl"](*operands)
                    case pycexpr.Add():
                        return self.oplut["add"](*operands)
                    case pycexpr.Sub():
                        return self.oplut["sub"](*operands)
                    case pycexpr.LessThan():
                        return self.oplut["ult"](*operands)
                    case pycexpr.LessThanEqual():
                        return self.oplut["ulte"](*operands)
                    case pycexpr.GreaterThan():
                        return self.oplut["ugt"](*operands)
                    case pycexpr.GreaterThanEqual():
                        return self.oplut["ugte"](*operands)
                    case pycexpr.Equality():
                        return self.oplut["eq"](*operands)
                    case pycexpr.Inequality():
                        return self.oplut["neq"](*operands)
            case pycexpr.Const(val=val, width=width):
                return self.slv.mk_const(val, BTORSort(width))
            case _:
                logger.error(
                    "Unsupported expression %s of type %s",
                    expr,
                    expr.__class__.__name__,
                )
                sys.exit(1)

    def dump_and_wait(self, assm):
        """Dump the assumption and wait for user input."""
        assm.Dump()
        assm.Dump("smt2")
        input("\nPRESS ENTER TO CONTINUE")

    def check_mm_refinement(
        self, mod1: SpecModule, mod2: SpecModule, rmap: RefinementMap
    ):
        """Check module-to-module refinement between two modules under a certain refinement map.

        Args:
            mod1 (SpecModule): The first module.
            mod2 (SpecModule): The second module.
            rmap (RefinementMap): The refinement map.
        """
        # Reset the global symbolic state
        self.reset()
        CPY1 = "cpy1"
        CPY2 = "cpy2"

        assert mod1.is_instantiated(), "Module 1 is not instantiated"
        assert mod2.is_instantiated(), "Module 2 is not instantiated"

        # Generate the assumptions and assertions
        # Input expressions
        input_a = [
            self.convert_expr_to_btor2(inv.expr, CPY1, 0)
            for inv in mod1._pycinternal__input_invs
        ]
        input_b = [
            self.convert_expr_to_btor2(inv.expr, CPY2, 0)
            for inv in mod2._pycinternal__input_invs
        ]

        # State expressions
        state_a_pre = []
        state_b_pre = []
        state_a_post = []
        state_b_post = []
        for inv in mod1._pycinternal__state_invs:
            state_a_pre.append(self.convert_expr_to_btor2(inv.expr, CPY1, 0))
            state_a_post.append(self.convert_expr_to_btor2(inv.expr, CPY1, 1))
        for inv in mod2._pycinternal__state_invs:
            state_b_pre.append(self.convert_expr_to_btor2(inv.expr, CPY2, 0))
            state_b_post.append(self.convert_expr_to_btor2(inv.expr, CPY2, 1))

        # Output expressions
        output_a = [
            self.convert_expr_to_btor2(inv.expr, CPY1, 1)
            for inv in mod1._pycinternal__output_invs
        ]
        output_b = [
            self.convert_expr_to_btor2(inv.expr, CPY2, 1)
            for inv in mod2._pycinternal__output_invs
        ]

        # Refinement assumptions
        ref_assms_pre = []
        ref_assms_post = []
        for a_expr, b_expr in rmap.mappings:
            ref_assms_pre.append(
                self.convert_expr_to_btor2(a_expr, CPY1, 0)
                == self.convert_expr_to_btor2(b_expr, CPY2, 0)
            )
            ref_assms_post.append(
                self.convert_expr_to_btor2(a_expr, CPY1, 1)
                == self.convert_expr_to_btor2(b_expr, CPY2, 1)
            )

        prev_a = []
        prev_b = []
        # Add the prev assumptions
        for signame, prev_sig in mod1._pycinternal__prev_signals.items():
            orig_sig = mod1._pycinternal__signals[signame]
            prev_a.append(
                self.convert_expr_to_btor2(prev_sig, CPY1, 1)
                == self.convert_expr_to_btor2(orig_sig, CPY1, 0)
            )
        for signame, prev_sig in mod2._pycinternal__prev_signals.items():
            orig_sig = mod2._pycinternal__signals[signame]
            prev_b.append(
                self.convert_expr_to_btor2(prev_sig, CPY2, 1)
                == self.convert_expr_to_btor2(orig_sig, CPY2, 0)
            )

        # Check input refinement
        logging.debug("Checking input refinement")
        for asrt, aexpr in zip(
            input_a + state_a_pre,
            mod1._pycinternal__input_invs + mod1._pycinternal__state_invs,
        ):

            for assm in input_b + state_b_pre + ref_assms_pre:
                # self.dump_and_wait(assm)
                self.slv.mk_assume(assm)

            self.slv.push()
            # self.dump_and_wait(self.slv.not_(asrt))
            self.slv.mk_assert(self.slv.not_(asrt))
            if self.slv.check_sat():
                logging.info("Input refinement failed for assertion %s", aexpr)
                logging.debug(self.slv.get_model())
                return False
            logging.debug("Input refinement passed for assertion %s", aexpr)
            self.slv.pop()
        logging.info("Input refinement passed")

        # Check inductive refinement
        logging.debug("Checking inductive refinement")
        for asrt, aexpr in zip(
            state_b_post + output_b,
            mod2._pycinternal__state_invs + mod2._pycinternal__output_invs,
        ):

            for assm in (
                state_a_post
                + output_a
                + ref_assms_pre
                + ref_assms_post
                + prev_a
                + prev_b
            ):
                # self.dump_and_wait(assm)
                self.slv.mk_assume(assm)

            self.slv.push()
            # self.dump_and_wait(self.slv.not_(asrt))
            self.slv.mk_assert(self.slv.not_(asrt))
            if self.slv.check_sat():
                logging.info("Inductive refinement failed for assertion %s", aexpr)
                logging.debug(self.slv.get_model())
                return False
            logging.debug("Inductive refinement passed for assertion %s", aexpr)
            self.slv.pop()

        logging.info("Inductive refinement passed")
        return True

    def check_ss_refinement(
        self, mod: SpecModule, bs1: str | Callable, bs2: str, flip_props: bool = False
    ):
        """Check bounded simulation refinement between two simulation schedules.

        Args:
            mod (SpecModule): The module to check refinement for.
            bs1 (str | Callable): The first simulation schedule.
            bs2 (str): The second simulation schedule.
            flip_props (bool): If True, turn assertions from the first schedule into assumptions.
        """
        # Reset the global symbolic state
        self.reset()
        CPY1 = "cpy1"

        # If bs1 is a method, get the name
        if not isinstance(bs1, str):
            bs1 = bs1.__name__
        if not isinstance(bs2, str):
            bs2 = bs2.__name__

        assert mod.is_instantiated(), "Module 1 is not instantiated"
        assert (
            bs1 in mod._pycinternal__simsteps
        ), f"Simulation schedule {bs1} not found in Module {mod.name}"
        assert (
            bs2 in mod._pycinternal__simsteps
        ), f"Simulation schedule {bs2} not found in Module {mod.name}"

        # Schedule 1
        sim1 = mod._pycinternal__simsteps[bs1]
        # Schedule 2
        sim2 = mod._pycinternal__simsteps[bs2]

        # Generate the assumptions and assertions
        # Input expressions
        assms_all = []
        if not flip_props:
            for i, step in enumerate(sim1.steps):
                assms_all.extend(
                    [
                        self.convert_expr_to_btor2(inv, CPY1, i)
                        for inv in step._pycinternal__assume
                    ]
                )
        else:
            for i, step in enumerate(sim1.steps):
                assms_all.extend(
                    [
                        self.convert_expr_to_btor2(inv, CPY1, i)
                        for inv in step._pycinternal__assume + step._pycinternal__assert
                    ]
                )

        assrts_all = []
        for i, step in enumerate(sim2.steps):
            assrts_all.extend(
                [
                    self.convert_expr_to_btor2(inv, CPY1, i)
                    for inv in step._pycinternal__assume
                ]
            )

        # TODO: Prev not handled
        assert len(mod._pycinternal__prev_signals) == 0, "Prev signals not handled"

        for assm in assms_all:
            # self.dump_and_wait(assm)
            self.slv.mk_assert(assm)

        # Check input refinement
        logging.debug("Checking input refinement")
        for asrt in assrts_all:

            self.slv.push()
            # self.dump_and_wait(self.slv.not_(asrt))
            self.slv.mk_assert(self.slv.not_(asrt))
            if self.slv.check_sat():
                logging.info(
                    "Bounded simulation refinement failed for assertion %s", asrt
                )
                logging.debug(self.slv.get_model())
                return False
            logging.debug("Bounded simulation refinement passed for assertion %s", asrt)
            self.slv.pop()

        logging.info("Bounded simulation refinement passed")
        return True
