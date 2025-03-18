"""
    pycaliper:

    PyCaliperSymex:
        Symbolic execution engine performing PER verification on BTOR
"""

import sys
import logging

from btoropt import program as prg
from btor2ex import BTORSolver, BTORSort, BTOR2Ex

from pycaliper.per import Logic, Path, SpecModule
from pycaliper.per import Expr as PYCExpr
import pycaliper.per.expr as pycexpr
from pycaliper.pycmanager import DesignConfig
from pycaliper.pycgui import PYCGUI

logger = logging.getLogger(__name__)


class PYCBTORInterface:
    """
    Symbolically execute a BTOR program: the barebones
    """

    def __init__(self, gui: PYCGUI = None):
        self.symex: BTOR2Ex = None
        self.cpy1: str = ""
        self.cpy2: str = ""
        self.specmodule: SpecModule = None
        self.gui = gui

    def get_lid(self, pth_) -> int:
        """Get the LID (label-id) for a given path"""
        if isinstance(pth_, Path):
            pth = pth_
        elif isinstance(pth_, Logic):
            pth = pth_.path
        else:
            logger.error(
                "Element %s is neither a Path nor a Logic, but is %s", pth, type(pth)
            )
            sys.exit(1)
        path1 = f"{self.cpy1}.{pth.get_hier_path()}"
        if path1 not in self.symex.names:
            logger.error("Signal path %s not found in BTOR program", path1)
            sys.exit(1)
        return self.symex.names[path1]

    def convert_expr_to_btor2(self, expr: PYCExpr, frame):
        """Convert a PyCaliper expression to a BTOR2 expression"""

        logger.debug("Converting expression %s", expr)

        if isinstance(expr, Logic):
            lid = self.get_lid(expr)
            btor_expr = frame[lid]
            logging.debug(
                "Logic of interest: %s, lid: %s, btor_expr: %s", expr, lid, btor_expr
            )
            # self.dump_and_wait(btor_expr)
            return btor_expr

        match expr:
            case pycexpr.OpApply(op=op, args=args):
                operands = [self.convert_expr_to_btor2(arg, frame) for arg in args]
                match op:
                    case pycexpr.LogicalAnd() | pycexpr.BinaryAnd():
                        return self.symex.oplut["and"](*operands)
                    case pycexpr.LogicalOr() | pycexpr.BinaryOr():
                        return self.symex.oplut["or"](*operands)
                    case pycexpr.BinaryXor():
                        return self.symex.oplut["xor"](*operands)
                    case pycexpr.UnaryLogicalNot() | pycexpr.UnaryBitwiseNot():
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
            case pycexpr.Const(val=val, width=width):
                return self.symex.slv.mk_const(val, BTORSort(width))
            case _:
                logger.error("Unsupported expression %s", expr)
                # Print class/type of expr
                print(expr.__class__.__name__)
                sys.exit(1)

    def get_lid_pair(self, pth_) -> tuple[int, int]:
        """Get the LID (label-id) pair for a given path"""
        if isinstance(pth_, Path):
            pth = pth_
        elif isinstance(pth_, Logic):
            pth = pth_.path
        else:
            logger.error(
                "Element %s is neither a Path nor a Logic, but is %s", pth, type(pth)
            )
            sys.exit(1)
        path1 = f"{self.cpy1}.{pth.get_hier_path()}"
        path2 = f"{self.cpy2}.{pth.get_hier_path()}"
        if path1 not in self.symex.names:
            logger.error("Signal path %s not found in BTOR program", path1)
            sys.exit(1)
        if path2 not in self.symex.names:
            logger.error("Signal path %s not found in BTOR program", path2)
            sys.exit(1)
        lid1 = self.symex.names[path1]
        lid2 = self.symex.names[path2]
        return (lid1, lid2)

    def get_tt_assm_constraints(self, eq_assms, condeq_assms, frame):
        """Get the constraints for the assumptions"""
        cons = []
        # Equality assumptions
        for assm_pth in eq_assms:
            lid1, lid2 = self.get_lid_pair(assm_pth)
            cons.append(self.symex.slv.eq_(frame[lid1], frame[lid2]))

        # Conditional equality assumptions
        for cond_assm in condeq_assms:
            pre_lid1, pre_lid2 = self.get_lid_pair(cond_assm[0])
            post_lid1, post_lid2 = self.get_lid_pair(cond_assm[1])
            cons.append(
                self.symex.slv.implies_(
                    self.symex.slv.and_(frame[pre_lid1], frame[pre_lid2]),
                    self.symex.slv.eq_(frame[post_lid1], frame[post_lid2]),
                )
            )
        return cons

    def get_tt_assrt_constraints(self, eq_assrts, condeq_assrts, frame):
        """Get the constraints for the assumptions"""
        # TODO: this will panic if constraint is on output
        cons = []
        for assrt_pth in eq_assrts:
            lid1, lid2 = self.get_lid_pair(assrt_pth)
            # Add negation of constraint
            cons.append(self.symex.slv.neq_(frame[lid1], frame[lid2]))
        # Conditional equality assumptions
        for cond_assrt in condeq_assrts:
            pre_lid1, pre_lid2 = self.get_lid_pair(cond_assrt[0])
            post_lid1, post_lid2 = self.get_lid_pair(cond_assrt[1])
            cons.append(
                self.symex.slv.and_(
                    self.symex.slv.and_(frame[pre_lid1], frame[pre_lid2]),
                    self.symex.slv.neq_(frame[post_lid1], frame[post_lid2]),
                )
            )
        return cons

    def get_assm_constraints(self, assms, frame) -> list:
        """Get the constraints for the assumptions"""
        cons = []
        for assm in assms:
            logger.debug("Assm: %s", assm)
            assm_expr = self.convert_expr_to_btor2(assm, frame)
            cons.append(assm_expr)
        return cons

    def get_assrt_constraints(self, assrts, frame):
        """Get the constraints for the assertions"""
        cons = []
        for assrt in assrts:
            logger.debug("Assrt: %s", assrt)
            assrt_expr = self.convert_expr_to_btor2(~assrt, frame)
            cons.append(assrt_expr)
        return cons

    def get_clock_constraints(self, states):
        clk_lid = self.get_lid(self.specmodule.get_clk())
        cons = []
        start_value = 0
        for state in states:
            cons.append(
                self.symex.slv.eq_(
                    state[clk_lid], self.symex.slv.mk_const(start_value, BTORSort(1))
                )
            )
            start_value = 1 - start_value
        return cons

    def dump_and_wait(self, assm):
        assm.Dump()
        assm.Dump("smt2")
        # input("\nPRESS ENTER TO CONTINUE")
