"""
File: pycaliper/synth/alignsynthesis.py
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

import logging
import sys
from vcdvcd import VCDVCD

from ..per import SpecModule, CtrAlignHole, Logic, Context

from ..vcdutils import get_subtrace
from ..pycmanager import PYCManager, DesignConfig

from ..jginterface.jgoracle import prove, is_pass, create_vcd_trace

from .synthprog import ZDDLUTSynthProgram

logger = logging.getLogger(__name__)


class AlignSynthesizer:
    def __init__(self, tmgr: PYCManager) -> None:
        """Initialize the AlignSynthesizer with a PYCManager.

        Args:
            tmgr (PYCManager): The PyCaliper manager for handling tasks.
        """
        self.tmgr = tmgr

    def _sample_tracepath(self) -> str:
        """Generate a simulation VCD trace from the design.

        Returns:
            str: Path to the generated VCD trace.
        """
        vcd_path = self.tmgr.get_vcd_path_random()
        if vcd_path is None:
            logger.error("No traces found.")
            sys.exit(1)
        return vcd_path

    def _inspect_module(self, specmodule: SpecModule) -> bool:
        """Inspect the module for well-formedness.

        Checks whether:
            (a) all signals in holes are single bit logic signals and
            (b) no two holes have common signals.

        Args:
            specmodule (SpecModule): The specification module to inspect.

        Returns:
            bool: True if module is well-formed, False otherwise.
        """
        caholes = specmodule._pycinternal__caholes
        holesigs = [set(h.sigs) for h in caholes]

        for i in range(len(caholes)):
            ch = caholes[i]
            if not all([isinstance(s, Logic) and s.width == 1 for s in ch.sigs]):
                logger.error(f"Non single bit-logic signal found in chole: {ch}.")
                return False

            for j in range(i + 1, len(caholes)):
                if holesigs[i].intersection(holesigs[j]):
                    logger.error(
                        f"Signals in holes {i} and {j} are not disjoint, found \
                        intersection {holesigs[i].intersection(holesigs[j])}"
                    )
                    return False

        return True

    def _synthesize_cahole(
        self, specmodule: SpecModule, cahole: CtrAlignHole, dc: DesignConfig
    ):
        """Attempt synthesis for a control alignment hole.

        Args:
            specmodule (SpecModule): The specification module containing the hole.
            cahole (CtrAlignHole): The control alignment hole to synthesize.
            dc (DesignConfig): The design configuration.
        """
        logger.debug(f"Attempting synthesis for hole {cahole}")

        vcdfile = self._sample_tracepath()
        vcdobj: VCDVCD = VCDVCD(vcdfile)

        intsigs = [cahole.ctr] + cahole.sigs

        clk = specmodule.get_clk().get_hier_path()
        # TODO: range should be a parameter
        trace = get_subtrace(vcdobj, intsigs, range(0, 16), clk, dc.cpy1)

        for s in cahole.sigs:
            zddsp = ZDDLUTSynthProgram(cahole.ctr, s)

            # Filter out assignments that have X
            filtered = [
                (assn[cahole.ctr].val, assn[s].val)
                for assn in trace
                if not (assn[cahole.ctr].isx or assn[s].isx)
            ]

            logger.debug(
                f"Filtered assignments for ctr {cahole.ctr} and s {s}: {filtered}"
            )

            ctr_vals = [f[0] for f in filtered]
            out_vals = [f[1] for f in filtered]
            zddsp.add_entries(ctr_vals, out_vals)

            if zddsp.solve():
                logger.info(
                    f"Solution found for hole {cahole}. Adding as an invariant."
                )

                # Disable the hole
                cahole.deactivate()

                inv = zddsp.get_inv()
                specmodule._inv(inv, Context.STATE)
            else:
                logger.info(f"No solution found for hole {cahole}.")

    def synthesize(self, specmodule: SpecModule, dc: DesignConfig) -> SpecModule:
        """Synthesize control alignment holes in the specification module.

        Args:
            specmodule (SpecModule): The specification module to synthesize.
            dc (DesignConfig): The design configuration.

        Returns:
            SpecModule: The synthesized specification module.
        """
        if not self._inspect_module(specmodule):
            logger.error(
                "SpecModule holes are not well-formed for AlignSynth, please check log. Exiting."
            )
            sys.exit(1)

        for h in specmodule._pycinternal__caholes:
            self._synthesize_cahole(specmodule, h, dc)

        return specmodule
