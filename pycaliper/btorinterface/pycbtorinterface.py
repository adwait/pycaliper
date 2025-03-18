"""
    pycaliper:

    PyCaliperSymex:
        Symbolic execution engine performing PER verification on BTOR
"""

import sys
import logging
from dataclasses import dataclass

from btor2ex import BTORSort, BTOR2Ex

from pycaliper.per import Logic, Path, SpecModule, PER, Eq, CondEq
from pycaliper.per import Expr as PYCExpr
import pycaliper.per.expr as pycexpr
from pycaliper.pycgui import PYCGUI

from .btordesign import BTORDesign, ClkEdge

logger = logging.getLogger(__name__)


@dataclass
class BTORVerifResult:
    verified: bool
    model: str


class PYCBTORInterface:
    """
    Symbolically execute a BTOR program: the barebones
    """

    def __init__(self, gui: PYCGUI = None):
        self.symex: BTOR2Ex = None
        self.cpy1: str = ""
        self.cpy2: str = ""
        self.specmodule: SpecModule = None
        self.des: BTORDesign = None
        self.gui = gui

    def get_lid(self, pth_, pref: str) -> int:
        """Get the LID (label-id) for a given path"""
        if isinstance(pth_, Path):
            pth = pth_
            path1 = f"{pref}.{pth.get_hier_path()}"
        elif isinstance(pth_, Logic):
            path1 = pth_.get_sva(pref)
        else:
            logger.error(
                "Element %s is neither a Path nor a Logic, but is %s", pth, type(pth)
            )
            sys.exit(1)
        if path1 not in self.symex.names:
            logger.error("Signal path %s not found in BTOR program", path1)
            sys.exit(1)
        logging.debug(
            "Logic of interest: path: %s, lid: %s", path1, self.symex.names[path1]
        )
        return self.symex.names[path1]

    def convert_expr_to_btor2(self, expr: PYCExpr, frame, pref: str):
        """Convert a PyCaliper expression to a BTOR2 expression"""

        logger.debug("Converting expression %s.%s", pref, expr)

        if (
            isinstance(expr, Logic)
            and expr.path.slicelow == -1
            and expr.path.slicehigh == -1
        ):
            lid = self.get_lid(expr, pref)
            btor_expr = frame[lid]
            logging.debug(
                "Logic of interest: %s, lid: %s, btor_expr: %s", expr, lid, btor_expr
            )
            # self.dump_and_wait(btor_expr)
            return btor_expr

        match expr:
            case pycexpr.OpApply(op=op, args=args):
                operands = [
                    self.convert_expr_to_btor2(arg, frame, pref) for arg in args
                ]
                match op:
                    case pycexpr.LogicalAnd():
                        redop = [self.symex.oplut["redor"](op) for op in operands]
                        assert len(redop) == 2, "LogicalAnd must have 2 operands"
                        return self.symex.oplut["and"](redop[0], redop[1])
                    case pycexpr.BinaryAnd():
                        return self.symex.oplut["and"](*operands)
                    case pycexpr.LogicalOr():
                        return self.symex.oplut["redor"](
                            self.symex.oplut["or"](*operands)
                        )
                    case pycexpr.BinaryOr():
                        return self.symex.oplut["or"](*operands)
                    case pycexpr.BinaryXor():
                        return self.symex.oplut["xor"](*operands)
                    case pycexpr.UnaryLogicalNot():
                        return self.symex.oplut["not"](
                            self.symex.oplut["redor"](*operands)
                        )
                    case pycexpr.UnaryBitwiseNot():
                        return self.symex.oplut["not"](*operands)
                    case pycexpr.LogicalShiftLeft():
                        return self.symex.oplut["sll"](*operands)
                    case pycexpr.LogicalShiftRight():
                        return self.symex.oplut["srl"](*operands)
                    case pycexpr.Add():
                        return self.symex.oplut["add"](*operands)
                    case pycexpr.Sub():
                        return self.symex.oplut["sub"](*operands)
                    case pycexpr.LessThan():
                        return self.symex.oplut["ult"](*operands)
                    case pycexpr.LessThanEqual():
                        return self.symex.oplut["ulte"](*operands)
                    case pycexpr.GreaterThan():
                        return self.symex.oplut["ugt"](*operands)
                    case pycexpr.GreaterThanEqual():
                        return self.symex.oplut["ugte"](*operands)
                    case pycexpr.Equality():
                        return self.symex.oplut["eq"](*operands)
                    case pycexpr.Inequality():
                        return self.symex.oplut["neq"](*operands)
                    case pycexpr.UnaryBitwiseAnd():
                        return self.symex.oplut["redand"](*operands)
                    case pycexpr.UnaryBitwiseOr():
                        return self.symex.oplut["redor"](*operands)
                    case _:
                        logger.error("Unsupported operator %s", op)
                        sys.exit(1)
            case pycexpr.Const(val=val, width=width):
                return self.symex.slv.mk_const(val, BTORSort(width))
            case Logic(width=width, path=pth, root=root):
                high = pth.slicehigh
                low = pth.slicelow
                assert high != -1, "Slice must be set, found high=%s, low=%s" % (
                    high,
                    low,
                )
                # New logic with a slice-free path
                new_logic = Logic(root=root)
                new_logic.path = Path(path=pth.path)
                if low == -1:
                    btor_expr = self.symex.oplut["slice"](
                        frame[self.get_lid(new_logic, pref)], 1, high, high
                    )
                else:
                    btor_expr = self.symex.oplut["slice"](
                        frame[self.get_lid(new_logic, pref)], high - low + 1, high, low
                    )
                return btor_expr
            case _:
                logger.error(
                    "Unsupported expression %s of class %s",
                    expr,
                    expr.__class__.__name__,
                )
                sys.exit(1)

    def get_tt_assm_constraints(self, assms: list[PER], frame):
        """Get the constraints for the assumptions"""
        cons = []
        for assm in assms:
            # Equality assumptions
            if isinstance(assm, Eq):
                lid1 = self.get_lid(assm.logic, self.cpy1)
                lid2 = self.get_lid(assm.logic, self.cpy2)
                cons.append(self.symex.slv.eq_(frame[lid1], frame[lid2]))
            # Conditional equality assumptions
            elif isinstance(assm, CondEq):
                # assert isinstance(assm.cond, Logic), "Currently, condition must be a Logic in CondEq."
                ant1 = self.convert_expr_to_btor2(assm.cond, frame, self.cpy1)
                ant2 = self.convert_expr_to_btor2(assm.cond, frame, self.cpy2)
                con1 = self.convert_expr_to_btor2(assm.logic, frame, self.cpy1)
                con2 = self.convert_expr_to_btor2(assm.logic, frame, self.cpy2)
                cons.append(
                    self.symex.slv.implies_(
                        self.symex.slv.and_(ant1, ant2),
                        self.symex.slv.eq_(con1, con2),
                    )
                )
        return cons

    def get_tt_assrt_constraints(self, assrts: list[PER], frame):
        """Get the constraints for the assumptions"""
        # TODO: this will panic if constraint is on output
        cons = []
        for assrt in assrts:
            # Equality assumptions
            if isinstance(assrt, Eq):
                lid1 = self.get_lid(assrt.logic, self.cpy1)
                lid2 = self.get_lid(assrt.logic, self.cpy2)
                # Add negation of constraint
                cons.append(self.symex.slv.neq_(frame[lid1], frame[lid2]))
            # Conditional equality assumptions
            elif isinstance(assrt, CondEq):
                # assert isinstance(assrt.cond, Logic), "Currently, condition must be a Logic in CondEq."
                ant1 = self.convert_expr_to_btor2(assrt.cond, frame, self.cpy1)
                ant2 = self.convert_expr_to_btor2(assrt.cond, frame, self.cpy2)
                con1 = self.convert_expr_to_btor2(assrt.logic, frame, self.cpy1)
                con2 = self.convert_expr_to_btor2(assrt.logic, frame, self.cpy2)
                cons.append(
                    self.symex.slv.and_(
                        self.symex.slv.and_(ant1, ant2), self.symex.slv.neq_(con1, con2)
                    )
                )
        return cons

    def get_assm_constraints(self, assms, frame, pref) -> list:
        """Get the constraints for the assumptions"""
        cons = []
        for assm in assms:
            logger.debug("Assm: %s", assm)
            assm_expr = self.convert_expr_to_btor2(assm, frame, pref)
            cons.append(assm_expr)
        return cons

    def get_assrt_constraints(self, assrts, frame, pref):
        """Get the constraints for the assertions"""
        cons = []
        for assrt in assrts:
            logger.debug("Assrt: %s", assrt)
            assrt_expr = self.convert_expr_to_btor2(~assrt, frame, pref)
            cons.append(assrt_expr)
        return cons

    def _get_clock_constraints(self):
        """Get the constraints for the clock"""
        clk_lid = self.get_lid(self.specmodule.get_clk(), self.cpy1)
        cons = []
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            if self.des.clkedge == ClkEdge.POSEDGE:
                start_value = 0
            else:
                start_value = 1
            for state in self.symex.state[:-1]:
                cons.append(
                    self.symex.slv.eq_(
                        state[clk_lid],
                        self.symex.slv.mk_const(start_value, BTORSort(1)),
                    )
                )
                start_value = 1 - start_value
            return cons
        else:
            # No clock constraints needed as clock has not been converted to fflogic
            return []

    def _internal_execute(self):
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            # Execute twice to get a single cycle
            self.symex.execute()
            self.symex.execute()
        else:
            # Execute once
            self.symex.execute()

    def _get_pre_state(self, i):
        # State at which assumes need to be impressed
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            return self.symex.state[2 * i]
        else:
            return self.symex.state[i]

    def _get_post_state(self, i):
        # State at which asserts need to be checked
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            return self.symex.state[2 * i + 1]
        else:
            return self.symex.state[i]

    def _get_prepost_assm_constraints_at_cycle(self, assms, i, pref):
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            return self.get_assm_constraints(
                assms, self._get_pre_state(i), pref
            ) + self.get_assm_constraints(assms, self._get_post_state(i), pref)
        else:
            return self.get_assm_constraints(assms, self._get_pre_state(i), pref)

    def _get_prepost_tt_assm_constraints_at_cycle(self, assms, i):
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            return self.get_tt_assm_constraints(
                assms, self._get_pre_state(i)
            ) + self.get_tt_assm_constraints(assms, self._get_post_state(i))
        else:
            return self.get_tt_assm_constraints(assms, self._get_pre_state(i))

    def _get_pre_assm_constraints_at_cycle(self, assms, i, pref):
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            return self.get_assm_constraints(assms, self._get_pre_state(i), pref)
        else:
            return []

    def _get_pre_tt_assm_constraints_at_cycle(self, assms, i):
        if self.des.clkedge == ClkEdge.POSEDGE or self.des.clkedge == ClkEdge.NEGEDGE:
            return self.get_tt_assm_constraints(assms, self._get_pre_state(i))
        else:
            return []

    def _get_tt_assrt_constraints_at_cycle(self, assrts, i):
        return self.get_tt_assrt_constraints(assrts, self._get_post_state(i))

    def _get_assrt_constraints_at_cycle(self, assrts, i, pref):
        return self.get_assrt_constraints(assrts, self._get_post_state(i), pref)

    def dump_and_wait(self, assm):
        assm.Dump()
        assm.Dump("smt2")
        input("\nPRESS ENTER TO CONTINUE")
