import logging

from ..pycconfig import PYConfig

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

from pycaliper.per import SpecModule

logger = logging.getLogger(__name__)


class JGVerifier1Trace:
    """One trace property verifier"""

    def __init__(self) -> None:
        pass

    def verify(self, specmodule: SpecModule, pyconfig: PYConfig) -> bool:
        """Verify one trace properties for the given module

        Args:
            specmodule (SpecModule): SpecModule to verify

        Returns:
            bool: True if the module is safe, False otherwise
        """

        svageni = svagen.SVAGen()
        svageni.create_pyc_specfile(
            specmodule, filename=pyconfig.jgc.pycfile, onetrace=True, dc=pyconfig.dc
        )
        self.candidates = svageni.holes

        loadscript(pyconfig.jgc.script)
        # Enable the assumptions for 1 trace verification
        set_assm_induction_1t(pyconfig.jgc.context, svageni.property_context)

        res = is_pass(prove_out_induction_1t(pyconfig.jgc.context))
        res_str = "SAFE" if res else "UNSAFE"
        logger.info(f"One trace verification result: {res_str}")
        return res


class JGVerifier2Trace:
    """Two trace property verifier"""

    def __init__(self) -> None:
        pass

    def verify(self, specmodule, pyconfig: PYConfig) -> bool:
        """Verify two trace properties for the given module

        Args:
            specmodule (SpecModule): SpecModule to verify

        Returns:
            bool: True if the module is safe, False otherwise
        """
        svageni = svagen.SVAGen()
        svageni.create_pyc_specfile(
            specmodule, filename=pyconfig.jgc.pycfile, dc=pyconfig.dc
        )

        loadscript(pyconfig.jgc.script)
        # Enable the assumptions for 2 trace verification
        set_assm_induction_2t(pyconfig.jgc.context, svageni.property_context)

        res = is_pass(prove_out_induction_2t(pyconfig.jgc.context))
        res_str = "SAFE" if res else "UNSAFE"
        logger.info("Two trace verification result: %s", res_str)
        return res


class JGVerifier1TraceBMC:
    """One trace property verifier with BMC"""

    def __init__(self) -> None:
        pass

    def verify(self, specmodule: SpecModule, pyconfig: PYConfig, schedule: str):
        """Verify one trace properties for the given module

        Args:
            specmodule (SpecModule): SpecModule to verify
            schedule (str): Simulation constraints

        Returns:
            bool: True if the module is safe, False otherwise
        """

        svageni = svagen.SVAGen()
        svageni.create_pyc_specfile(
            specmodule, filename=pyconfig.jgc.pycfile, dc=pyconfig.dc
        )
        self.candidates = svageni.holes

        loadscript(pyconfig.jgc.script)
        # Enable the assumptions for 1 trace verification
        set_assm_bmc(pyconfig.jgc.context, svageni.property_context, schedule)

        results = [
            is_pass(r)
            for r in prove_out_bmc(
                pyconfig.jgc.context, svageni.property_context, schedule
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
