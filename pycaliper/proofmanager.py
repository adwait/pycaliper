"""
File: pycaliper/proofmanager.py

This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

from typing import Callable

from pycaliper.pycgui import GUIPacket, WebGUI, RichGUI

import btoropt
from pycaliper.per import SpecModule

from pycaliper.pycconfig import DesignConfig, PYConfig, Design, JasperConfig
from pycaliper.jginterface.jgdesign import JGDesign
from pycaliper.verif.jgverifier import (
    JGVerifier1TraceBMC,
    JGVerifier1Trace,
    JGVerifier2Trace,
    JGVerifier1TraceInvariant,
)
from pycaliper.btorinterface.btordesign import BTORDesign
from pycaliper.verif.btorverifier import BTORVerifier1Trace, BTORVerifier2Trace
from pycaliper.verif.refinementverifier import RefinementMap, RefinementVerifier
from pycaliper.synth.persynthesis import (
    PERSynthesizer,
    HoudiniSynthesizerJG,
    HoudiniSynthesizerBTOR,
    HoudiniSynthesizerConfig,
    HoudiniSynthesizerStats,
)
from pycaliper.synth.alignsynthesis import AlignSynthesizer
from pycaliper.synth.iis_strategy import SeqStrategy, RandomStrategy, LLMStrategy, IISStrategy
from pycaliper.svagen import SVAGen


from btor2ex.utils import parsewrapper

from dataclasses import dataclass
import json
import sys
import logging
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from pycaliper.jginterface import jasperclient as jgc

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


# Configuration schemas
JG_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "jasper": {
            "type": "object",
            "properties": {
                # The Jasper working directory relative to the pycaliper directory
                "jdir": {"type": "string"},
                # The TCL script relative to the Jasper working directory
                "script": {"type": "string"},
                # Location of the generated SVA file relative to the Jasper working directory
                "pycfile": {"type": "string"},
                # Proof node context
                "context": {"type": "string"},
                # Design list file
                "design_list": {"type": "string"},
                # Port number to connect to Jasper server
                "port": {"type": "integer"},
            },
            "required": ["jdir", "script", "pycfile", "context"],
        }
    },
    "required": ["jasper"],
    "additionalProperties": False,
}


D_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "cpy1": {"type": "string"},
        "cpy2": {"type": "string"},
        "topmod": {"type": "string"},
        "lang": {"type": "string"},
        "clk": {"type": "string"},
    },
    "required": ["cpy1", "topmod"],
    "additionalProperties": False,
}


def mock_or_connect(mock: bool, port: int) -> bool:
    """Connect to Jasper or run in mock mode.

    Args:
        mock (bool): True if running in mock mode, False if connected to Jasper.
        port (int): Port number to connect to Jasper.

    Returns:
        bool: True if connected to Jasper, False if running in mock mode.
    """
    if mock:
        logger.info("Running in mock mode.")
        return False
    else:
        jgc.connect_tcp("localhost", port)
        return True


def get_jgconfig(jgcpath: str) -> JasperConfig:
    """Load and validate Jasper configuration from file.

    Args:
        jgcpath (str): Path to the Jasper configuration JSON file.

    Returns:
        JasperConfig: The loaded and validated Jasper configuration.
    """
    with open(jgcpath, "r") as f:
        jgconfig = json.load(f)
    # And validate it
    try:
        validate(instance=jgconfig, schema=JG_CONFIG_SCHEMA)
    except ValidationError as e:
        logger.error(f"Jasper config schema validation failed: {e.message}")
        logger.error(
            f"Please check schema:\n{json.dumps(JG_CONFIG_SCHEMA, indent=4, sort_keys=True, separators=(',', ': '))}"
        )
        sys.exit(1)
    return JasperConfig(
        jdir=jgconfig["jasper"]["jdir"],
        script=jgconfig["jasper"]["script"],
        pycfile=jgconfig["jasper"]["pycfile"],
        context=jgconfig["jasper"]["context"],
        design_list=jgconfig["jasper"].get("design_list", "design.lst"),
        port=jgconfig["jasper"].get("port", 8080),
    )


def get_designconfig(dcpath: str) -> DesignConfig:
    """Load and validate design configuration from file.

    Args:
        dcpath (str): Path to the design configuration JSON file.

    Returns:
        DesignConfig: The loaded and validated design configuration.
    """
    with open(dcpath, "r") as f:
        dconfig = json.load(f)
    try:
        validate(instance=dconfig, schema=D_CONFIG_SCHEMA)
    except ValidationError as e:
        logger.error(f"Design config schema validation failed: {e.message}")
        logger.error(
            f"Please check schema:\n{json.dumps(D_CONFIG_SCHEMA, indent=4, sort_keys=True, separators=(',', ': '))}"
        )
        sys.exit(1)
    return DesignConfig(
        cpy1=dconfig["cpy1"],
        cpy2=dconfig.get("cpy2", "b"),
        lang=dconfig.get("lang", "sv12"),
        topmod=dconfig["topmod"],
        clk=dconfig.get("clk", "clk"),
    )


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

    def save_spec(self, module: SpecModule, filepath: str) -> None:
        """Save a specification module to a file.

        Args:
            module (SpecModule): The specification module to save.
            filepath (str): The path where to save the specification file.
        """
        with open(filepath, "w") as f:
            f.write(module.full_repr())
        logger.info(f"Specification written to {filepath}.")

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

    def mk_jg_design_from_pyc(
        self, name: str, jasper_config_path: str, design_config_path: str
    ) -> Design:
        """Creates a JGDesign from Jasper and Design config paths.

        Args:
            name (str): The name of the design.
            jasper_config_path (str): Path to the Jasper config JSON file.
            design_config_path (str): Path to the design config JSON file.

        Returns:
            Design: The created JGDesign object.
        """
        if name in self.designs:
            logger.warning(f"Design {name} already exists.")
        jgc = get_jgconfig(jasper_config_path)
        dc = get_designconfig(design_config_path)
        pyc = PYConfig(jgc=jgc, dc=dc)
        des = JGDesign(name, pyc)
        self.designs[name] = des

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_DESIGN,
                dname=name,
                file=f"JG: {jasper_config_path}, DC: {design_config_path}",
                params=None,
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

    def mk_per_synthesis(
        self,
        spec: SpecModule | str,
        strategy: str = "seq",
        fuel: int = 3,
        steps: int = 10,
        retries: int = 1,
    ) -> SpecModule:
        """Performs PER synthesis on a specification.

        Args:
            spec (SpecModule | str): The specification module or name.
            strategy (str, optional): The synthesis strategy ('seq', 'rand', 'llm'). Defaults to "seq".
            fuel (int, optional): Fuel budget for synthesis. Defaults to 3.
            steps (int, optional): Step budget for synthesis. Defaults to 10.
            retries (int, optional): Number of retries. Defaults to 1.

        Returns:
            SpecModule: The synthesized specification module.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]

        # Get strategy
        strat = self.get_strategy(strategy)
        
        # Create a PYConfig - this is a simplified version for PER synthesis
        # In practice, this would need proper Jasper configuration
        jgc = JasperConfig(
            jdir="temp", script="temp.tcl", pycfile="temp.sv", context="temp"
        )
        dc = DesignConfig(cpy1="a", topmod="top")
        pyconfig = PYConfig(jgc=jgc, dc=dc)

        synthesizer = PERSynthesizer(pyconfig, strat, fuel, steps)
        result_spec = synthesizer.synthesize(spec, retries)
        
        logger.info(f"PER synthesis completed for spec {spec.name}")
        return result_spec

    def mk_houdini_synthesis_jg(
        self,
        spec: SpecModule | str,
        design: Design | str,
        strategy: str = "seq",
        fuel: int = 10,
        steps: int = 10,
    ) -> tuple[SpecModule, HoudiniSynthesizerStats]:
        """Performs Houdini synthesis using JasperGold.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.
            strategy (str, optional): The synthesis strategy. Defaults to "seq".
            fuel (int, optional): Fuel budget for synthesis. Defaults to 10.
            steps (int, optional): Step budget for synthesis. Defaults to 10.

        Returns:
            tuple[SpecModule, HoudiniSynthesizerStats]: The synthesized spec and statistics.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]
        if isinstance(design, str):
            if design not in self.designs:
                raise ValueError(f"Design {design} not found.")
            design = self.designs[design]

        assert isinstance(
            design, JGDesign
        ), "Design must be a JGDesign for Houdini JG synthesis."

        strat = self.get_strategy(strategy)
        config = HoudiniSynthesizerConfig(fuelbudget=fuel, stepbudget=steps)
        
        synthesizer = HoudiniSynthesizerJG()
        result_spec, stats = synthesizer.synthesize(spec, design, design.pyc.dc, strat, config)
        
        logger.info(f"Houdini JG synthesis completed for spec {spec.name}")
        return result_spec, stats

    def mk_houdini_synthesis_btor(
        self,
        spec: SpecModule | str,
        design: Design | str,
        strategy: str = "seq",
        fuel: int = 10,
        steps: int = 10,
    ) -> tuple[SpecModule, HoudiniSynthesizerStats]:
        """Performs Houdini synthesis using BTOR.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.
            strategy (str, optional): The synthesis strategy. Defaults to "seq".
            fuel (int, optional): Fuel budget for synthesis. Defaults to 10.
            steps (int, optional): Step budget for synthesis. Defaults to 10.

        Returns:
            tuple[SpecModule, HoudiniSynthesizerStats]: The synthesized spec and statistics.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]
        if isinstance(design, str):
            if design not in self.designs:
                raise ValueError(f"Design {design} not found.")
            design = self.designs[design]

        assert isinstance(
            design, BTORDesign
        ), "Design must be a BTORDesign for Houdini BTOR synthesis."

        strat = self.get_strategy(strategy)
        config = HoudiniSynthesizerConfig(fuelbudget=fuel, stepbudget=steps)
        dc = DesignConfig(cpy1="a", topmod="top")  # Default config
        
        synthesizer = HoudiniSynthesizerBTOR(self.gui)
        result_spec, stats = synthesizer.synthesize(spec, design, dc, strat, config)
        
        logger.info(f"Houdini BTOR synthesis completed for spec {spec.name}")
        return result_spec, stats

    def mk_align_synthesis(
        self,
        spec: SpecModule | str,
        dc: DesignConfig = DesignConfig(),
        trace_dir: str = None,
    ) -> SpecModule:
        """Performs alignment synthesis on a specification.

        Args:
            spec (SpecModule | str): The specification module or name.
            dc (DesignConfig, optional): The design configuration. Defaults to DesignConfig().
            trace_dir (str, optional): Directory containing VCD trace files. Defaults to None.

        Returns:
            SpecModule: The synthesized specification module.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]

        synthesizer = AlignSynthesizer(trace_dir)
        result_spec = synthesizer.synthesize(spec, dc)
        
        logger.info(f"Alignment synthesis completed for spec {spec.name}")
        return result_spec


    def generate_sva(
        self,
        spec: SpecModule | str,
        output_file: str,
        dc: DesignConfig = None,
        onetrace: bool = False,
    ) -> None:
        """Generates SystemVerilog Assertions (SVA) from a specification.

        Args:
            spec (SpecModule | str): The specification module or name.
            output_file (str): Path to the output SVA file.
            dc (DesignConfig, optional): Design configuration. Defaults to None.
            onetrace (bool, optional): Generate one-trace properties only. Defaults to False.
        """
        if isinstance(spec, str):
            if spec not in self.specs:
                raise ValueError(f"Spec {spec} not found.")
            spec = self.specs[spec]

        if dc is None:
            dc = DesignConfig(cpy1="a", topmod="top")

        svagen = SVAGen()
        svagen.create_pyc_specfile(spec, dc, output_file, onetrace)
        
        logger.info(f"SVA generated for spec {spec.name} and saved to {output_file}")


    def get_verifier(self, engine_type: str, **kwargs):
        """Gets a verifier instance of the specified type.

        Args:
            engine_type (str): The type of verifier ('btor_one_trace', 'btor_two_trace', 
                              'jg_one_trace', 'jg_two_trace', 'jg_bmc', 'jg_invariant').
            **kwargs: Additional arguments for the verifier.

        Returns:
            Verifier: The requested verifier instance.
        """
        match engine_type.lower():
            case "btor_one_trace":
                return BTORVerifier1Trace(kwargs.get("gui", self.gui))
            case "btor_two_trace":
                return BTORVerifier2Trace(kwargs.get("gui", self.gui))
            case "jg_one_trace":
                return JGVerifier1Trace()
            case "jg_two_trace":
                return JGVerifier2Trace()
            case "jg_bmc":
                return JGVerifier1TraceBMC()
            case "jg_invariant":
                return JGVerifier1TraceInvariant()
            case _:
                raise ValueError(f"Unknown verifier type: {engine_type}")

    def get_synthesizer(self, engine_type: str, **kwargs):
        """Gets a synthesizer instance of the specified type.

        Args:
            engine_type (str): The type of synthesizer ('per', 'houdini_jg', 'houdini_btor', 'align').
            **kwargs: Additional arguments for the synthesizer.

        Returns:
            Synthesizer: The requested synthesizer instance.
        """
        match engine_type.lower():
            case "per":
                jgc = kwargs.get("jgc", JasperConfig(jdir="temp", script="temp.tcl", pycfile="temp.sv", context="temp"))
                dc = kwargs.get("dc", DesignConfig(cpy1="a", topmod="top"))
                pyconfig = PYConfig(jgc=jgc, dc=dc)
                strategy = kwargs.get("strategy", SeqStrategy())
                fuel = kwargs.get("fuel", 3)
                steps = kwargs.get("steps", 10)
                return PERSynthesizer(pyconfig, strategy, fuel, steps)
            case "houdini_jg":
                return HoudiniSynthesizerJG()
            case "houdini_btor":
                return HoudiniSynthesizerBTOR(kwargs.get("gui", self.gui))
            case "align":
                trace_dir = kwargs.get("trace_dir", None)
                return AlignSynthesizer(trace_dir)
            case _:
                raise ValueError(f"Unknown synthesizer type: {engine_type}")

    def get_strategy(self, strategy_name: str) -> IISStrategy:
        """Gets a synthesis strategy instance.

        Args:
            strategy_name (str): The strategy name ('seq', 'rand', 'llm').

        Returns:
            IISStrategy: The requested strategy instance.
        """
        match strategy_name.lower():
            case "seq":
                return SeqStrategy()
            case "rand" | "random":
                return RandomStrategy()
            case "llm":
                return LLMStrategy()
            case _:
                raise ValueError(f"Unknown strategy: {strategy_name}")


    def load_spec_from_path(self, specpath: str, name: str = "", params: dict = {}) -> SpecModule:
        """Loads a specification module from a file path.

        Args:
            specpath (str): Path to the specification module file.
            name (str, optional): Name for the spec. If empty, uses the class name.
            params (dict, optional): Parameters to pass to the specification. Defaults to {}.

        Returns:
            SpecModule: The loaded and instantiated specification module.
        """
        import importlib.util
        import os
        
        # Load the module from file
        spec_name = os.path.splitext(os.path.basename(specpath))[0]
        spec = importlib.util.spec_from_file_location(spec_name, specpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the SpecModule class in the module
        spec_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, SpecModule) and 
                attr != SpecModule):
                spec_class = attr
                break
        
        if spec_class is None:
            raise ValueError(f"No SpecModule class found in {specpath}")
        
        # Use class name if no name provided
        if not name:
            name = spec_class.__name__
        
        # Create and return the spec
        return self.mk_spec(spec_class, name, **params)

    def mk_design_from_config(
        self, 
        name: str, 
        jgc_path: str = "", 
        dc_path: str = "",
        btor_path: str = ""
    ) -> Design:
        """Creates a design from configuration files or BTOR file.

        Args:
            name (str): Name for the design.
            jgc_path (str, optional): Path to Jasper config file.
            dc_path (str, optional): Path to design config file.
            btor_path (str, optional): Path to BTOR file.

        Returns:
            Design: The created design object.
        """
        if btor_path:
            return self.mk_btor_design_from_file(btor_path, name)
        elif jgc_path and dc_path:
            return self.mk_jg_design_from_pyc(name, jgc_path, dc_path)
        else:
            raise ValueError("Must provide either btor_path or both jgc_path and dc_path")

    def mk_btor_proof_two_trace(
        self,
        spec: SpecModule | str,
        design: Design | str,
        dc: DesignConfig = DesignConfig(),
    ) -> ProofResult:
        """Creates a BTOR proof for two trace verification.

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

        res = BTORVerifier2Trace(self.gui).verify(spec, design, dc)

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                sname=spec.name,
                dname=design.name,
                result=("PASS" if res.verified else "FAIL"),
            )
        )
        pr = TwoTraceIndPR(spec=spec.name, design=design.name, dc=dc, result=res.verified)
        self.proofs.append(pr)
        return pr

    def mk_jg_proof_one_trace(
        self, spec: SpecModule | str, design: Design | str
    ) -> ProofResult:
        """Creates a JG proof for one trace property verification.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.

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

        assert isinstance(
            design, JGDesign
        ), "Design must be a JGDesign for Jasper verification."

        mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
        verifier = JGVerifier1Trace()

        res = verifier.verify(spec, design.pyc)
        pr = OneTraceIndPR(
            spec=spec.name, design=design.name, dc=design.pyc.dc, result=res
        )

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                proofterm=str(pr),
                result=("PASS" if res else "FAIL"),
            )
        )
        self.proofs.append(pr)
        return pr

    def mk_jg_proof_two_trace(
        self, spec: SpecModule | str, design: Design | str
    ) -> ProofResult:
        """Creates a JG proof for two trace property verification.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.

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

        assert isinstance(
            design, JGDesign
        ), "Design must be a JGDesign for Jasper verification."

        mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
        verifier = JGVerifier2Trace()

        res = verifier.verify(spec, design.pyc)
        pr = TwoTraceIndPR(
            spec=spec.name, design=design.name, result=res, dc=design.pyc.dc
        )

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                proofterm=str(pr),
                result=("PASS" if res else "FAIL"),
            )
        )
        self.proofs.append(pr)
        return pr

    def mk_jg_proof_invariant(
        self, spec: SpecModule | str, design: Design | str, schedule: str
    ) -> ProofResult:
        """Creates a JG proof for invariant verification using sequences.

        Args:
            spec (SpecModule | str): The specification module or name.
            design (Design | str): The design or name.
            schedule (str): The sequence schedule name.

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

        assert isinstance(
            design, JGDesign
        ), "Design must be a JGDesign for Jasper verification."

        mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
        verifier = JGVerifier1TraceInvariant()

        res = verifier.verify(spec, design.pyc, schedule)
        pr = OneTraceIndPR(
            spec=spec.name, design=design.name, dc=design.pyc.dc, result=res
        )

        self._push_update(
            GUIPacket(
                t=GUIPacket.T.NEW_PROOF,
                iden=str(len(self.proofs)),
                proofterm=str(pr),
                result=("PASS" if res else "FAIL"),
            )
        )
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
