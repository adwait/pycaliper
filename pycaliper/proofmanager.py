"""
File: pycaliper/proofmanager.py

This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

from typing import Callable

from pycaliper.pycmanager import mock_or_connect
from pycaliper.pycgui import GUIPacket, WebGUI, RichGUI

import btoropt
from pycaliper.per import SpecModule

from pycaliper.pycconfig import DesignConfig, PYConfig, Design
from pycaliper.jginterface.jgdesign import JGDesign
from pycaliper.verif.jgverifier import JGVerifier1TraceBMC
from pycaliper.btorinterface.btordesign import BTORDesign
from pycaliper.verif.btorverifier import BTORVerifier1Trace
from pycaliper.verif.refinementverifier import RefinementMap, RefinementVerifier

from btor2ex.utils import parsewrapper

from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    result: bool


@dataclass
class OneTraceBndPR(ProofResult):
    spec: str
    sched: str
    design: str

    def __str__(self) -> str:
        return "OneTraceBnd(%s, %s, %s)" % (self.spec, self.sched, self.design)


@dataclass
class OneTraceIndPR(ProofResult):
    spec: str
    design: str
    dc: DesignConfig

    def __str__(self) -> str:
        return "OneTraceInd(%s, %s, %s)" % (self.spec, self.design, self.dc)


@dataclass
class TwoTraceIndPR(ProofResult):
    spec: str
    design: str
    dc: DesignConfig

    def __str__(self) -> str:
        return "TwoTraceInd(%s, %s, %s)" % (self.spec, self.design, self.dc)


@dataclass
class MMRefinementPR(ProofResult):
    spec1: str
    spec2: str
    rmap: RefinementMap

    def __str__(self) -> str:
        return "MMRefinement(%s, %s, %s)" % (self.spec1, self.spec2, self.rmap)


@dataclass
class SSRefinementPR(ProofResult):
    spec: str
    sched1: str
    sched2: str
    flip: bool

    def __str__(self) -> str:
        return "SSRefinement(%s, %s, %s, %s)" % (
            self.spec,
            self.sched1,
            self.sched2,
            self.flip,
        )


def mk_btordesign(name: str, filename: str) -> BTORDesign:
    """Creates a BTORDesign from a file.

    Args:
        name (str): The name of the design.
        filename (str): The filename of the BTOR file.

    Returns:
        BTORDesign: The created BTORDesign object.
    """
    prgm = btoropt.parse(parsewrapper(filename))
    return BTORDesign(name, prgm)


class ProofManager:
    def __init__(self, webgui=False, cligui=False) -> None:
        """Initializes the ProofManager.

        Args:
            webgui (bool, optional): Whether to use the web GUI. Defaults to False.
            cligui (bool, optional): Whether to use the CLI GUI. Defaults to False.
        """
        self.proofs: list[ProofResult] = []
        self.designs: dict[str, Design] = {}
        self.specs: dict[str, SpecModule] = {}
        self.pyconfigs: dict[str, PYConfig] = {}
        if webgui:
            self.gui = WebGUI()
            self.gui.run()
        elif cligui:
            self.gui = RichGUI()
            self.gui.run()
        else:
            self.gui = None

    def _push_update(self, data: GUIPacket) -> None:
        """Pushes an update to the GUI.

        Args:
            data (GUIPacket): The data to push.
        """
        if self.gui:
            self.gui.push_update(data)

    def mk_spec(self, spec: SpecModule.__class__, name: str, **kwargs) -> SpecModule:
        """Creates a specification module.

        Args:
            spec (SpecModule.__class__): The specification module class.
            name (str): The name of the specification.
            **kwargs: Additional keyword arguments for the specification.

        Returns:
            SpecModule: The created specification module.
        """
        if name in self.specs:
            logger.warning(f"Spec {name} already exists.")

        new_spec: SpecModule = spec(name, **kwargs)
        new_spec.instantiate()
        self.specs[name] = new_spec
        new_spec.name = name
        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_SPEC,
                sname=name,
                file=f"{spec.__module__}.{spec.__name__}",
                params=str(kwargs),
            )
        )
        return new_spec

    def mk_btor_design_from_file(self, file: str, name: str) -> BTORDesign:
        """Creates a BTORDesign from a file.

        Args:
            file (str): The filename of the BTOR file.
            name (str): The name of the design.

        Returns:
            BTORDesign: The created BTORDesign object.
        """
        if name in self.designs:
            logger.warning(f"Design {name} already exists.")
        prgm = btoropt.parse(parsewrapper(file))
        des = BTORDesign(name, prgm)
        self.designs[name] = des

        self._push_update(
            GUIPacket(t=GUIPacket.T.NEW_DESIGN, dname=name, file=file, params=None)
        )
        return des

    def mk_jg_design_from_pyc(self, name: str, pyc: PYConfig) -> Design:
        """Creates a JGDesign from a PYConfig.

        Args:
            name (str): The name of the design.
            pyc (PYConfig): The PYConfig object.

        Returns:
            Design: The created JGDesign object.
        """
        if name in self.designs:
            logger.warning(f"Design {name} already exists.")
        des = JGDesign(name, pyc)
        self.designs[name] = des

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_DESIGN, dname=name, file=pyc.jgc.pycfile, params=None
            )
        )
        return des

    def mk_btor_proof_one_trace(
        self,
        spec: SpecModule | str,
        design: Design | str,
        dc: DesignConfig = DesignConfig(),
    ) -> ProofResult:
        """Creates a BTOR proof for one trace.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.
            dc (DesignConfig, optional): The design configuration. Defaults to DesignConfig().

        Returns:
            ProofResult: The result of the proof.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]
        if isinstance(design, str):
            if design not in self.designs:
                raise ValueError(f"Design {design} not found.")
            design = self.designs[design]

        res = BTORVerifier1Trace(self.gui).verify(spec, design, dc)

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                sname=spec.name,
                dname=design.name,
                result=("PASS" if res else "FAIL"),
            )
        )
        pr = OneTraceIndPR(spec=spec.name, design=design.name, dc=dc, result=res)
        self.proofs.append(pr)
        return pr

    def mk_jg_proof_bounded_spec(
        self, spec: SpecModule | str, design: Design | str, sched: Callable | str
    ) -> ProofResult:
        """Creates a JG proof for a bounded specification.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.
            sched (Callable | str): The schedule or name.

        Returns:
            ProofResult: The result of the proof.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]
        if isinstance(design, str):
            if design not in self.designs:
                raise ValueError(f"Design {design} not found.")
            design = self.designs[design]

        sched_name = sched.__name__ if callable(sched) else sched

        assert isinstance(
            design, JGDesign
        ), "Design must be a JGDesign for Jasper verification."

        mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
        verifier = JGVerifier1TraceBMC()

        res = verifier.verify(spec, design.pyc, sched_name)
        result_str = ",".join(
            [f"step<{i}>:{'PASS' if r else 'FAIL'}" for i, r in enumerate(res)]
        )
        pr = OneTraceBndPR(
            spec=spec.name, sched=sched_name, design=design.name, result=all(res)
        )

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                proofterm=str(pr),
                result=("PASS" if all(res) else "FAIL"),
            )
        )
        self.proofs.append(pr)
        return pr

    def check_mm_refinement(
        self, spec1: SpecModule | str, spec2: SpecModule | str, rmap: RefinementMap
    ) -> ProofResult:
        """Checks MM refinement between two specifications.

        Args:
            spec1 (SpecModule | str): The first specification module or name.
            spec2 (SpecModule | str): The second specification module or name.
            rmap (RefinementMap): The refinement map.

        Returns:
            ProofResult: The result of the refinement check.
        """
        if isinstance(spec1, str):
            if spec1 not in self.specs:
                raise ValueError(f"Spec {spec1} not found.")
            spec1 = self.specs[spec1]
        if isinstance(spec2, str):
            if spec2 not in self.specs:
                raise ValueError(f"Spec {spec2} not found.")
            spec2 = self.specs[spec2]

        res = RefinementVerifier().check_mm_refinement(spec1, spec2, rmap)
        pr = MMRefinementPR(spec1=spec1.name, spec2=spec2.name, rmap=rmap, result=res)
        self.proofs.append(pr)

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                proofterm=str(pr),
                result=("PASS" if res else "FAIL"),
            )
        )

        return pr

    def check_ss_refinement(
        self,
        spec: SpecModule | str,
        sched1: Callable | str,
        sched2: Callable | str,
        flip: bool = False,
    ) -> ProofResult:
        """Checks SS refinement between two schedules.

        Args:
            spec (SpecModule | str): The specification module or name.
            sched1 (Callable | str): The first schedule or name.
            sched2 (Callable | str): The second schedule or name.
            flip (bool, optional): Whether to flip the assertions in sched1. Defaults to False.

        Returns:
            ProofResult: The result of the refinement check.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]

        sched1_name = sched1.__name__ if callable(sched1) else sched1
        sched2_name = sched2.__name__ if callable(sched2) else sched2

        res = RefinementVerifier().check_ss_refinement(spec, sched1, sched2, flip)
        pr = SSRefinementPR(
            spec=spec.name,
            sched1=sched1_name,
            sched2=sched2_name,
            result=res,
            flip=flip,
        )
        self.proofs.append(pr)

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                proofterm=str(pr),
                result=("PASS" if res else "FAIL"),
            )
        )

        return pr
