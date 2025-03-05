from pycaliper.pycgui import GUIPacket, WebGUI, RichGUI

import btoropt
from pycaliper.per import SpecModule

from pycaliper.pycconfig import DesignConfig
from pycaliper.verif.btorverifier import BTORVerifier1Trace, BTORDesign, Design

from btor2ex.btor2ex.utils import parsewrapper

from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    spec: str
    design: str
    dc: str
    result: bool


def mk_btordesign(name: str, filename: str):
    prgm = btoropt.parse(parsewrapper(filename))
    return BTORDesign(name, prgm)


class ProofManager:
    def __init__(self, webgui=False, cligui=False) -> None:
        self.proofs: list[ProofResult] = []
        self.designs: dict[str, Design] = {}
        self.specs: dict[str, SpecModule] = {}
        if webgui:
            self.gui = WebGUI()
            self.gui.run()
        elif cligui:
            self.gui = RichGUI()
            self.gui.run()
        else:
            self.gui = None

    def _push_update(self, data: GUIPacket):
        if self.gui:
            self.gui.push_update(data)

    def mk_spec(self, spec: SpecModule.__class__, name: str, **kwargs):
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
                file=spec.__module__,
                params=str(kwargs),
            )
        )
        return new_spec

    def mk_btor_design_from_file(self, file: str, name: str) -> BTORDesign:
        if name in self.designs:
            logger.warning(f"Design {name} already exists.")
        prgm = btoropt.parse(parsewrapper(file))
        des = BTORDesign(name, prgm)
        self.designs[name] = des

        self._push_update(
            GUIPacket(t=GUIPacket.T.NEW_DESIGN, dname=name, file=file, params=None)
        )
        return des

    def mk_btor_proof_one_trace(
        self,
        spec: SpecModule | str,
        design: Design | str,
        dc: DesignConfig = DesignConfig(),
    ) -> ProofResult:
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
        pr = ProofResult(spec.name, design.name, dc, res)
        self.proofs.append(pr)
        return pr
