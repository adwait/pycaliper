"""
File: pycaliper/proofmanager.py

This file is a part of the PyCaliper tool.
See LICENSE.md for licensing information.

Author: Adwait Godbole, UC Berkeley
"""

import os
import sys
import logging
import json
from dataclasses import dataclass
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from typing import Callable
from pydantic import BaseModel
import importlib

import btoropt

from btor2ex.utils import parsewrapper
from pycaliper.jginterface import jasperclient as jgc
from pycaliper.pycgui import GUIPacket, WebGUI, RichGUI
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
    HoudiniSynthesizerJG,
    HoudiniSynthesizerBTOR,
    HoudiniSynthesizerConfig,
    HoudiniSynthesizerStats,
)
from pycaliper.synth.alignsynthesis import AlignSynthesizer
from pycaliper.synth.iis_strategy import SeqStrategy, RandomStrategy, LLMStrategy, IISStrategy
from pycaliper.svagen import SVAGen


logger = logging.getLogger(__name__)


class PYCArgs(BaseModel):
    """Arguments for PyCaliper tasks.

    This class defines the arguments that can be passed to PyCaliper tasks.
    It uses Pydantic for validation and default values.

    Attributes:
        specpath (str): Path to the specification module.
        jgcpath (str): Path to the Jasper configuration file.
        dcpath (str): Path to the design configuration file.
        params (str): Parameters for the specification module.
        sdir (str): Directory to save results to.
        tdir (str): Directory containing trace files.
        onetrace (bool): Whether to verify only one-trace properties.
        bmc (str): Bounded model checking configuration.
        requires_jasper (bool): Whether this task requires Jasper connection.
    """

    specpath: str = ""  #: Path to the specification module
    jgcpath: str = ""  #: Path to the Jasper configuration file
    dcpath: str = ""  #: Path to the design configuration file
    params: str = ""  #: Parameters for the specification module
    sdir: str = ""  #: Directory to save results to
    tdir: str = ""  #: Directory containing trace files
    onetrace: bool = False  #: Whether to verify only one-trace properties
    bmc: str = ""  #: Bounded model checking configuration
    requires_jasper: bool = True  #: Whether this task requires Jasper connection


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


def get_pyconfig(args: PYCArgs) -> PYConfig:
    """Create a PyCaliper configuration from arguments.

    This function creates a PyCaliper configuration from the provided arguments,
    including loading and validating Jasper and design configurations.

    Args:
        args (PYCArgs): PyCaliper arguments.

    Returns:
        PYConfig: PyCaliper configuration.
    """

    if args.jgcpath != "":
        jasperc = get_jgconfig(args.jgcpath)
    else:
        jasperc = JasperConfig()

    if args.dcpath != "":
        designc = get_designconfig(args.dcpath)
    else:
        designc = DesignConfig()

    return PYConfig(
        # Working directory
        sdir=args.sdir,
        # Is this a mock run
        mock=(args.jgcpath == ""),
        # Jasper configuration
        jgc=jasperc,
        # Trace directory
        tdir=args.tdir,
        onetrace=args.onetrace,
        dc=designc,
    )


class NameGenerator:
    """A simple name generator for PyCaliper tasks."""
    def __init__(self):
        self.namecounts = {}

    def get_name(self, base: str) -> str:
        """Generate a unique name based on the base name."""
        if base not in self.namecounts:
            self.namecounts[base] = 0
        self.namecounts[base] += 1
        return f"{base}_{self.namecounts[base]}"

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
        self.connected = False
        self.ng = NameGenerator()
        if webgui:
            self.gui = WebGUI()
            self.gui.run()
        elif cligui:
            self.gui = RichGUI()
            self.gui.run()
        else:
            self.gui = None

    def mock_or_connect(self, mock: bool, port: int) -> bool:
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
        elif not self.connected:
            jgc.connect_tcp("localhost", port)
            self.connected = True
            return True
            

    def _get_specmodname(specpath: str) -> str:
        """Extract the specification module name from a path.

        Args:
            specpath (str): Path to the specification module.

        Returns:
            str: Name of the specification module.
        """
        if "/" in specpath:
            module_name = specpath.rsplit("/", 1)[1]
            if "." in module_name:
                module_name = module_name.rsplit(".", 1)[0]
            return module_name
        return specpath

    def mk_spec_from_path(self, specpath: str, args: PYCArgs) -> SpecModule:
        """Dynamically import the spec module and create an instance of it."""

        params = {}
        if args.params:
            for pair in args.params.split(","):
                key, value = pair.split("=")
                params[key] = int(value)

        # Split the module name into the module name and the parent package
        module_path, module_name = specpath.rsplit("/", 1)

        # Check if the path exists
        if not os.path.isdir(module_path):
            logger.error(f"Path '{module_path}' does not exist.")
            exit(1)
        # Add the module path to sys.path
        sys.path.append(module_path)

        try:
            name = None
            if "." in module_name:
                module_name, class_name = module_name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                logger.debug(
                    f"Successfully imported module: {module_name} from {module_path}"
                )
                name = self.ng.get_name(class_name)
                module_class = getattr(module, class_name)
            else:
                # Import the module using importlib
                module = importlib.import_module(module_name)
                logger.debug(
                    f"Successfully imported module: {module_name} from {module_path}"
                )
                name = self.ng.get_name(module_name)
                module_class = getattr(module, module_name)

            specmodule = module_class(**params)
            if not isinstance(specmodule, SpecModule):
                logger.error(f"Class {specmodule} is not a SpecModule.")
                return None
            logger.debug(f"Created instance {name} of {module_name} with params: {params}")

            specmodule.instantiate()
            self.specs[name] = specmodule
            specmodule.name = name
            self._push_update(
                GUIPacket(
                    t=GUIPacket.T.NEW_SPEC,
                    sname=name,
                    file=f"{module_class.__module__}.{module_class.__name__}",
                    params=str(params),
                )
            )
            return specmodule

        except ImportError as e:
            logger.error(f"Error importing module {module_name} from {module_path}: {e}")
            return None
        finally:
            # Clean up: remove the path from sys.path to avoid potential side effects
            sys.path.remove(module_path)

    def start(self, args: PYCArgs) -> tuple[PYConfig, SpecModule]:
        """Start a PyCaliper task.

        This function sets up all components for a PyCaliper task and checks that
        the task can be run in the current mode.

        Args:
            args (PYCArgs): PyCaliper arguments.

        Returns:
            tuple: Tuple of (pyconfig, module).
        """
        pyconfig = get_pyconfig(args)
        is_connected = self.mock_or_connect(pyconfig.mock, pyconfig.jgc.port)

        module = self.mk_spec_from_path(args.specpath, args)
        assert module is not None, f"SpecModule {args.specpath} not found."

        if args.requires_jasper and not is_connected:
            logger.error(
                "This task requires Jasper sockets, cannot be run in mock mode."
            )
            sys.exit(1)

        return pyconfig, module

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

        self.mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
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

        self.mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
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

        self.mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
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

        self.mock_or_connect(design.pyc.mock, design.pyc.jgc.port)
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
