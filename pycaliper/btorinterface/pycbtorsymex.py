"""
    pycaliper:

    PyCaliperSymex:
        Symbolic execution engine performing PER verification on BTOR
"""

import sys
import logging

from btoropt import program as prg
from btor2ex import BTORSolver, BTORSort, BTOR2Ex

from pycaliper.per import Logic
from pycaliper.per import Expr as PYCExpr
import pycaliper.per.expr as pycexpr
from pycaliper.pycmanager import PYConfig

logger = logging.getLogger(__name__)


class PYCBTORSymex(BTOR2Ex):
    """
    Symbolically execute a BTOR program: the barebones
    """

    def __init__(
        self,
        pyconfig: PYConfig,
        solver: BTORSolver,
        prog: list[prg.Instruction],
        cpy1: str = "a",
        cpy2: str = "b",
    ):
        super().__init__(solver, prog)

        self.pyconfig = pyconfig
        self.cpy1 = cpy1
        self.cpy2 = cpy2

        # Two trace invariants
        self.eq_assms: list = []
        self.condeq_assms: list[tuple[PYCExpr, PYCExpr]] = []
        self.eq_assrts: list = []
        self.condeq_assrts: list[tuple[PYCExpr, PYCExpr]] = []

        self.holes = []

        # One trace invariants
        self.inv_assms: list[PYCExpr] = []
        self.inv_assrts: list[PYCExpr] = []

        self.pycexpr_to_btor2_opmap = {
            pycexpr.LogicalAnd: "and",
            pycexpr.LogicalOr: "or",
            pycexpr.BinaryXor: "xor",
            pycexpr.UnaryLogicalNot: "not",
            pycexpr.LogicalShiftLeft: "sll",
            pycexpr.LogicalShiftRight: "srl",
            pycexpr.Add: "add",
            pycexpr.Sub: "sub",
            pycexpr.LessThan: "slt",
            pycexpr.LessThanEqual: "slte",
            pycexpr.GreaterThan: "sgt",
            pycexpr.GreaterThanEqual: "sgte",
            pycexpr.Equality: "eq",
            pycexpr.Inequality: "neq",
            pycexpr.BinaryAnd: "and",
            pycexpr.BinaryOr: "or",
            pycexpr.UnaryBitwiseNot: "not",
        }

    def add_eq_assms(self, assms: list[PYCExpr]):
        """Add two trace equality assumptions"""
        # TODO: ignore slices for now
        self.eq_assms.extend(assms)

    def add_condeq_assms(self, condeq_assms: list[tuple[PYCExpr, PYCExpr]]):
        """Add two trace conditional equality assumptions"""
        self.condeq_assms.extend(condeq_assms)

    def add_eq_assrts(self, assrts: list[PYCExpr]):
        """Add two trace equality assertions"""
        # TODO: ignore slices for now
        self.eq_assrts.extend(assrts)

    def add_condeq_assrts(self, condeq_assrts: list[tuple[PYCExpr, PYCExpr]]):
        """Add two trace conditional equality assertions"""
        self.condeq_assrts.extend(condeq_assrts)

    def add_hole_constraints(self, holes: list[PYCExpr]):
        """Add constraints for holes"""
        self.holes.extend(holes)

    def add_assms(self, assms: list[PYCExpr]):
        """Add one-trace assumptions"""
        self.inv_assms.extend(assms)

    def add_assrts(self, assrts: list[PYCExpr]):
        """Add one-trace assertions"""
        self.inv_assrts.extend(assrts)

    def get_lid(self, pth: PYCExpr) -> int:
        """Get the LID (label-id) for a given path"""
        path1 = f"{self.cpy1}.{pth}"
        if path1 not in self.names:
            logger.error("Signal path %s not found in BTOR program", path1)
            sys.exit(1)
        return self.names[path1]

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
                logger.error("Unsupported expression %s", expr)
                # Print class/type of expr
                print(expr.__class__.__name__)
                sys.exit(1)

    def get_lid_pair(self, pth: PYCExpr) -> tuple[int, int]:
        """Get the LID (label-id) pair for a given path"""
        path1 = f"{self.cpy1}.{pth}"
        path2 = f"{self.cpy2}.{pth}"
        if path1 not in self.names:
            logger.error("Signal path %s not found in BTOR program", path1)
            sys.exit(1)
        if path2 not in self.names:
            logger.error("Signal path %s not found in BTOR program", path2)
            sys.exit(1)
        lid1 = self.names[path1]
        lid2 = self.names[path2]
        return (lid1, lid2)

    def get_tt_assm_constraints(self, frame):
        """Get the constraints for the assumptions"""
        cons = []
        # Equality assumptions
        for assm_pth in self.eq_assms:
            lid1, lid2 = self.get_lid_pair(assm_pth)
            cons.append(self.slv.eq_(frame[lid1], frame[lid2]))

        # Conditional equality assumptions
        for cond_assm in self.condeq_assms:
            pre_lid1, pre_lid2 = self.get_lid_pair(cond_assm[0])
            post_lid1, post_lid2 = self.get_lid_pair(cond_assm[1])
            cons.append(
                self.slv.implies_(
                    self.slv.and_(frame[pre_lid1], frame[pre_lid2]),
                    self.slv.eq_(frame[post_lid1], frame[post_lid2]),
                )
            )
        return cons

    def get_tt_assrt_constraints(self, frame):
        """Get the constraints for the assumptions"""
        # TODO: this will panic if constraint is on output
        cons = []
        for assrt_pth in self.eq_assrts:
            lid1, lid2 = self.get_lid_pair(assrt_pth)
            # Add negation of constraint
            cons.append(self.slv.neq_(frame[lid1], frame[lid2]))
        # Conditional equality assumptions
        for cond_assrt in self.condeq_assrts:
            pre_lid1, pre_lid2 = self.get_lid_pair(cond_assrt[0])
            post_lid1, post_lid2 = self.get_lid_pair(cond_assrt[1])
            cons.append(
                self.slv.and_(
                    self.slv.and_(frame[pre_lid1], frame[pre_lid2]),
                    self.slv.neq_(frame[post_lid1], frame[post_lid2]),
                )
            )
        return cons

    def get_assm_constraints(self, frame) -> list:
        """Get the constraints for the assumptions"""
        cons = []
        for assm in self.inv_assms:
            logger.debug("Assm: %s", assm)
            assm_expr = self.convert_expr_to_btor2(assm, frame)
            cons.append(assm_expr)
        return cons

    def get_assrt_constraints(self, frame):
        """Get the constraints for the assertions"""
        cons = []
        for assrt in self.inv_assrts:
            logger.debug("Assrt: %s", assrt)
            assrt_expr = self.convert_expr_to_btor2(~assrt, frame)
            cons.append(assrt_expr)
        return cons

    def get_hole_constraints(self, preframe, postframe):
        """Get constraints for holes"""
        precons = []
        postcons = []
        for hole in self.holes:
            lid1, lid2 = self.get_lid_pair(hole)
            precons.append(self.slv.eq_(preframe[lid1], postframe[lid2]))
            postcons.append(self.slv.neq_(postframe[lid1], preframe[lid2]))
        return precons, postcons

    def inductive_two_safety(self) -> bool:
        """Verifier for inductive two-safety property

        Returns:
            bool: is SAFE?
        """
        # Unroll twice
        self.execute()
        # Check
        self.execute()

        pre_state = self.state[0]
        post_state = self.state[1]

        logger.debug("Pre state: %s", pre_state)
        logger.debug("Post state: %s", post_state)

        assms = self.get_tt_assm_constraints(pre_state)
        assrts = self.get_tt_assrt_constraints(post_state)

        logger.debug("Assms: %s", assms)
        logger.debug("Assrts: %s", assrts)

        for assrt in assrts:
            # ! This is buggy since assertions are not removed
            for assm in assms:
                self.slv.mk_assume(assm)
            # Apply all internal program assumptions
            for assmdict in self.prgm_assms:
                for _, assmi in assmdict.items():
                    self.slv.mk_assume(assmi)
            self.slv.mk_assert(assrt)
            result = self.slv.check_sat()
            logger.debug(
                "For assertion %s, result %s", assrt, "BUG" if result else "SAFE"
            )
            if result:
                logger.debug("Found a bug")
                model = self.slv.get_model()
                logger.debug("Model:\n%s", model)
                return False

        logger.debug("No bug found, inductive proof complete")
        # Safe
        return True

    def inductive_two_safety_syn(self) -> bool:
        """Synthesizer for inductive two-safety property

        Returns:
            bool: synthesis result
        """
        # Unroll twice
        self.execute()
        # Check
        self.execute()

        pre_state = self.state[0]
        post_state = self.state[1]

        logger.debug("Pre state: %s", pre_state)
        logger.debug("Post state: %s", post_state)

        assms = self.get_tt_assm_constraints(pre_state)
        assrts = self.get_tt_assrt_constraints(post_state)

        hole_assms, hole_assrts = self.get_hole_constraints(pre_state, post_state)

        logger.debug("Assms: %s", assms)
        logger.debug("Assrts: %s", assrts)
        logger.debug("Hole Assms: %s", hole_assms)
        logger.debug("Hole Assrts: %s", hole_assrts)

        while hole_assms:
            # ! This is buggy since assertions are not removed
            failed = False
            for assrt in assrts + hole_assrts:
                for assm in assms + hole_assms:
                    self.slv.mk_assume(assm)
                for assmdict in self.prgm_assms:
                    for _, assmi in assmdict.items():
                        self.slv.mk_assume(assmi)
                self.slv.mk_assert(assrt)
                result = self.slv.check_sat()
                logger.debug(
                    "For assertion %s, result %s", assrt, "BUG" if result else "SAFE"
                )
                if result:
                    logger.debug("Found a bug")
                    model = self.slv.get_model()
                    logger.debug("Model:\n%s", model)
                    failed = True
                    break
            if failed:
                hole_assms = hole_assms[:-1]
                hole_assrts = hole_assrts[:-1]
            else:
                logger.debug("Self-inductive fp found, synthesis complete.")
                return self.holes[: len(hole_assms)]

        logger.debug("No synthesis solution found, synthesis failed.")
        return []

    def get_clock_constraints(self, states):
        clk_lid = self.get_lid(self.pyconfig.clk)
        cons = []
        start_value = 0
        for state in states:
            cons.append(
                self.slv.eq_(
                    state[clk_lid], self.slv.mk_const(start_value, BTORSort(1))
                )
            )
            start_value = 1 - start_value
        return cons

    def dump_and_wait(self, assm):
        assm.Dump()
        assm.Dump("smt2")
        # input("\nPRESS ENTER TO CONTINUE")

    def inductive_one_safety(self) -> bool:
        """Verifier for inductive one-trace property

        Returns:
            bool: is SAFE?
        """
        k = self.pyconfig.k
        all_assms = []
        for i in range(k):
            # Unroll 2k times
            self.execute()
            self.execute()

            # Constrain on falling transitions (intra-steps)
            ind_state = self.state[2 * i]
            # Collect all assumptions
            all_assms.extend(self.get_assm_constraints(ind_state))

        # Check final state (note that unrolling has one extra state)
        final_state = self.state[2 * k - 1]
        all_assrts = self.get_assrt_constraints(final_state)

        # Clocking behaviour
        clk_assms = self.get_clock_constraints(self.state[:-1])

        for assrt_expr, assrt in zip(self.inv_assrts, all_assrts):
            for assm in all_assms:
                # self.dump_and_wait(assm)
                self.slv.mk_assume(assm)
            for clk_assm in clk_assms:
                # self.dump_and_wait(clk_assm)
                self.slv.mk_assume(clk_assm)
            for assmdict in self.prgm_assms:
                for _, assmi in assmdict.items():
                    self.slv.mk_assume(assmi)

            self.slv.push()
            # self.dump_and_wait(assrt)
            self.slv.mk_assert(assrt)
            # self.dump_and_wait(assrt)
            result = self.slv.check_sat()
            logger.debug(
                "For assertion %s, result %s", assrt_expr, "BUG" if result else "SAFE"
            )
            if result:
                logger.debug("Found a bug")
                model = self.slv.get_model()
                logger.debug("Model:\n%s", model)
                return False
            self.slv.pop()

        logger.debug("No bug found, inductive proof complete")
        # Safe
        return True
