import logging
import sys

from btoropt import program as prg
from ..btorinterface.pycbtorsymex import PYCBTORSymex, DesignConfig
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


class BTORVerifier2Trace:
    def __init__(self, gui=None) -> None:
        self.gui = gui

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ) -> bool:
        """
        Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq
        """
        assert specmodule.is_instantiated(), "Module not instantiated."

        slv = PYCBTORSymex(des.prgm, dc, specmodule, gui=self.gui)

        if specmodule._pycinternal__perholes or specmodule._pycinternal__caholes:
            logger.error(
                "Holes not supported in a verifier, please use a synthesizer. Exiting."
            )
            sys.exit(1)

        eq_assms = []
        eq_assrts = []
        condeq_assms = []
        condeq_assrts = []

        # Generate the assumptions and assertions
        for p in specmodule._pycinternal__input_tt:
            match p:
                case Eq():
                    eq_assms.append(p.logic)
                case CondEq():
                    condeq_assms.append((p.cond, p.logic))
        for p in specmodule._pycinternal__state_tt:
            match p:
                case Eq():
                    eq_assms.append(p.logic)
                    eq_assrts.append(p.logic)
                case CondEq():
                    condeq_assms.append((p.cond, p.logic))
                    condeq_assrts.append((p.cond, p.logic))
        for p in specmodule._pycinternal__output_tt:
            match p:
                case Eq():
                    eq_assrts.append(p.logic)
                case CondEq():
                    condeq_assrts.append((p.cond, p.logic))

        slv.add_eq_assms(eq_assms)
        slv.add_condeq_assms(condeq_assms)
        slv.add_eq_assrts(eq_assrts)
        slv.add_condeq_assrts(condeq_assrts)

        logger.debug(f"eq_assms: %s, eq_assrts: %s", eq_assms, eq_assrts)

        # Perform verification
        return slv.inductive_two_safety()


class BTORVerifier1Trace:
    def __init__(self, gui=None):
        self.gui = gui

    def verify(
        self,
        specmodule: SpecModule,
        des: BTORDesign,
        dc: DesignConfig,
    ) -> bool:
        """
        Perform verification for a single module of the following property:
            input_eq && state_eq |-> ##1 output_eq && state_eq
        """
        assert specmodule.is_instantiated(), "Module not instantiated."

        slv = PYCBTORSymex(des.prgm, dc, specmodule, gui=self.gui)

        if specmodule._pycinternal__perholes or specmodule._pycinternal__caholes:
            logger.warn("Holes found in a verifier, ignoring them.")

        assms = []
        assrts = []

        # Generate the assumptions and assertions
        for p in specmodule._pycinternal__input_invs:
            assms.append(p.expr)
        for p in specmodule._pycinternal__state_invs:
            assms.append(p.expr)
            assrts.append(p.expr)
        for p in specmodule._pycinternal__output_invs:
            assrts.append(p.expr)

        slv.add_assms(assms)
        slv.add_assrts(assrts)

        (kd, _) = specmodule.get_unroll_kind_depths()
        logger.debug(
            f"Performing verification with assms: %s, assrts: %s with depth %s",
            assms,
            assrts,
            kd,
        )

        # Perform verification
        return slv.inductive_one_safety(k=kd)
