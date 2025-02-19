import logging

from ..pycmanager import PYConfig

from .. import svagen
from ..jginterface.jgoracle import (
    prove_out_induction_1t,
    prove_out_induction_2t,
    prove_out_bmc,
    loadscript,
    is_pass,
    set_assm_induction_1t,
    set_assm_induction_2t,
    set_assm_bmc,
)

from .invverifier import InvVerifier

logger = logging.getLogger(__name__)


class JGVerifier1Trace(InvVerifier):
    """One trace property verifier"""

    def __init__(self, pyconfig: PYConfig) -> None:
        super().__init__(pyconfig)
        self.svagen = None

    def verify(self, module) -> bool:
        """Verify one trace properties for the given module

        Args:
            module (SpecModule): SpecModule to verify

        Returns:
            bool: True if the module is safe, False otherwise
        """

        self.svagen = svagen.SVAGen(module)
        self.svagen.create_pyc_specfile(filename=self.psc.pycfile, onetrace=True)
        self.candidates = self.svagen.holes

        loadscript(self.psc.script)
        # Enable the assumptions for 1 trace verification
        set_assm_induction_1t(self.psc.context, self.svagen.property_context)

        res = is_pass(prove_out_induction_1t(self.psc.context))
        res_str = "SAFE" if res else "UNSAFE"
        logger.info(f"One trace verification result: {res_str}")
        return res


class JGVerifier2Trace(InvVerifier):
    """Two trace property verifier"""

    def __init__(self, pyconfig: PYConfig) -> None:
        super().__init__(pyconfig)
        self.svagen = None
        self.candidates = None

    def verify(self, module):
        """Verify two trace properties for the given module

        Args:
            module (SpecModule): SpecModule to verify

        Returns:
            bool: True if the module is safe, False otherwise
        """
        self.svagen = svagen.SVAGen(module)
        self.svagen.create_pyc_specfile(filename=self.psc.pycfile)
        self.candidates = self.svagen.holes

        loadscript(self.psc.script)
        # Enable the assumptions for 2 trace verification
        set_assm_induction_2t(self.psc.context, self.svagen.property_context)

        res = is_pass(prove_out_induction_2t(self.psc.context))
        res_str = "SAFE" if res else "UNSAFE"
        logger.info("Two trace verification result: %s", res_str)
        return res


class JGVerifier1TraceBMC(InvVerifier):
    """One trace property verifier with BMC"""

    def __init__(self, pyconfig: PYConfig) -> None:
        super().__init__(pyconfig)
        self.svagen = None
        self.candidates = None

    def verify(self, module, schedule: str):
        """Verify one trace properties for the given module

        Args:
            module (SpecModule): SpecModule to verify
            schedule (str): Simulation constraints

        Returns:
            bool: True if the module is safe, False otherwise
        """

        self.svagen = svagen.SVAGen(module)
        self.svagen.create_pyc_specfile(filename=self.psc.pycfile)
        self.candidates = self.svagen.holes

        loadscript(self.psc.script)
        # Enable the assumptions for 1 trace verification
        set_assm_bmc(self.psc.context, self.svagen.property_context, schedule)

        results = [
            is_pass(r)
            for r in prove_out_bmc(
                self.psc.context, self.svagen.property_context, schedule
            )
        ]
        results_str = "\n\t".join(
            [
                f"Step {i}: SAFE" if res else f"Step {i}: UNSAFE"
                for (i, res) in enumerate(results)
            ]
        )
        logger.info("One trace verification result:\n\t%s", results_str)
        return results
