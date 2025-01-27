# Module to module refinement verification

import logging
import sys

from btor2ex import BTORSolver, BTORSort
from pycaliper.per import Expr as PYCExpr
import pycaliper.per.expr as pycexpr
from pycaliper.per import Logic, Module, Path


logger = logging.getLogger(__name__)


class MMRVerifier:
    def __init__(self, slv: BTORSolver):
        self.slv = slv
        self.oplut = slv.oplut()

        self.varmap = {}

    def convert_expr_to_btor2(self, expr: PYCExpr, step=0):
        """Convert a PyCaliper expression to a BTOR2 expression"""

        logger.debug("Converting expression %s", expr)

        if isinstance(expr, Logic):
            if f"{expr.name}_{step}" in self.varmap:
                return self.varmap[f"{expr.name}_{step}"]

            logging.debug("Creating variable %s with width %d", expr.name, expr.width)
            var = self.slv.mk_var(f"{expr.name}_{step}", BTORSort(expr.width))
            self.varmap[f"{expr.name}_{step}"] = var
            return var

        match expr:
            case pycexpr.OpApply(op=op, args=args):
                operands = [self.convert_expr_to_btor2(arg, step) for arg in args]
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
        assm.Dump()
        assm.Dump("smt2")
        input("\nPRESS ENTER TO CONTINUE")

    def check_refinement(self, mod1: Module, mod2: Module, rmap: list, k=1):

        if k != 1:
            raise NotImplementedError("Only 1-step refinement is supported")

        # Clear the variable map
        self.varmap = {}

        # Instantiate the modules
        mod1.instantiate(Path([("a", [])]))
        mod2.instantiate(Path([("b", [])]))

        # Generate the assumptions and assertions
        # Input expressions
        input_a = []
        input_b = []
        for inv in mod1._pycinternal__input_invs:
            input_a.append(self.convert_expr_to_btor2(inv.expr, 0))
        for inv in mod2._pycinternal__input_invs:
            input_b.append(self.convert_expr_to_btor2(inv.expr, 0))

        # State expressions
        state_a_pre = []
        state_b_pre = []
        state_a_post = []
        state_b_post = []
        for inv in mod1._pycinternal__state_invs:
            state_a_pre.append(self.convert_expr_to_btor2(inv.expr, 0))
            state_a_post.append(self.convert_expr_to_btor2(inv.expr, 1))
        for inv in mod2._pycinternal__state_invs:
            state_b_pre.append(self.convert_expr_to_btor2(inv.expr, 0))
            state_b_post.append(self.convert_expr_to_btor2(inv.expr, 1))

        # Output expressions
        output_a = []
        output_b = []
        for inv in mod1._pycinternal__output_invs:
            output_a.append(self.convert_expr_to_btor2(inv.expr, 1))
        for inv in mod2._pycinternal__output_invs:
            output_b.append(self.convert_expr_to_btor2(inv.expr, 1))

        # Refinement assumptions
        ref_assms_pre = []
        ref_assms_post = []
        for (a_expr, b_expr) in rmap:
            ref_assms_pre.append(
                self.convert_expr_to_btor2(a_expr, 0)
                == self.convert_expr_to_btor2(b_expr, 0)
            )
            ref_assms_post.append(
                self.convert_expr_to_btor2(a_expr, 1)
                == self.convert_expr_to_btor2(b_expr, 1)
            )

        prev_a = []
        prev_b = []
        # Add the prev assumptions
        for signame, prev_sig in mod1._prev_signals.items():
            orig_sig = mod1._signals[signame]
            prev_a.append(
                self.convert_expr_to_btor2(prev_sig, 1)
                == self.convert_expr_to_btor2(orig_sig, 0)
            )
        for signame, prev_sig in mod2._prev_signals.items():
            orig_sig = mod2._signals[signame]
            prev_b.append(
                self.convert_expr_to_btor2(prev_sig, 1)
                == self.convert_expr_to_btor2(orig_sig, 0)
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
                logging.error("Input refinement failed for assertion %s", aexpr)
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
                input_a
                + input_b
                + state_a_pre
                + state_b_pre
                + state_a_post
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
                logging.error("Inductive refinement failed for assertion %s", aexpr)
                logging.debug(self.slv.get_model())
                return False
            logging.debug("Inductive refinement passed for assertion %s", aexpr)
            self.slv.pop()
        logging.info("Inductive refinement passed")
