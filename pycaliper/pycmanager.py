import os
import sys
import logging
import random
from enum import Enum

import tempfile
import importlib

import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from pycaliper.jginterface import jasperclient as jgc

from pydantic import BaseModel
from .pycconfig import PYConfig, DesignConfig, JasperConfig
from .per.per import SpecModule

logger = logging.getLogger(__name__)


class PYCTask(Enum):
    SVAGEN = 0
    VERIF1T = 1
    VERIF2T = 2
    VERIFBMC = 3
    CTRLSYNTH = 4
    PERSYNTH = 5
    FULLSYNTH = 6


class PYCArgs(BaseModel):
    specpath: str = ""
    jgcpath: str = ""
    dcpath: str = ""
    params: str = ""
    sdir: str = ""
    tdir: str = ""
    onetrace: bool = False
    bmc: str = ""


class PYCManager:
    def __init__(self, pyconfig: PYConfig):
        self.pycspec: str = pyconfig.pycspec
        # Previous VCD traces directory

        self.pyconfig = pyconfig
        self.sdir = pyconfig.sdir

        # Create a temporary directory for the run, grab the name, and clean it up
        wdir = tempfile.TemporaryDirectory(prefix="pyc_wdir_")
        self.wdir = wdir.name
        wdir.cleanup()

        logger.info(f"Working directory: {self.wdir}")
        self.tracedir = f"{self.wdir}/traces"
        self.specdir = f"{self.wdir}/specs"

        # Create the directories
        os.makedirs(self.tracedir, exist_ok=True)
        os.makedirs(self.specdir, exist_ok=True)

        self.num_vcd_files = 0
        self.traces = {}

        self.num_spec_files = 0
        self.specs = {}

        if pyconfig.tdir != "":
            self.gather_all_traces(pyconfig.tdir)

    def gather_all_traces(self, tdir):
        # Collect all vcd files in the directory (non-subdirs) at wdir
        for f in os.listdir(tdir):
            if f.endswith(".vcd"):
                # Copy the trace to tracedir
                os.system(f"cp {tdir}/{f} {self.tracedir}")
                self.traces[self.num_vcd_files] = f"{self.tracedir}/{f}"
                self.num_vcd_files += 1

    def create_vcd_path(self):
        path = f"{self.tracedir}/trace{self.num_vcd_files}.vcd"
        self.traces[self.num_vcd_files] = path
        self.num_vcd_files += 1
        return path

    def get_vcd_path(self, idx):
        return self.traces[idx]

    def get_vcd_path_random(self):
        if self.num_vcd_files == 0:
            logger.warn(f"No VCD files found in directory {self.tracedir}.")
            return None
        return self.traces[random.randint(0, self.num_vcd_files - 1)]

    def save_spec(self, module: SpecModule):
        # Create path
        path = f"{self.specdir}/{self.pycspec}.spec{self.num_spec_files}.py"
        self.specs[self.num_spec_files] = path
        self.num_spec_files += 1

        with open(path, "x") as f:
            f.write(module.full_repr())

        logger.info(f"Specification written to {path}.")

    def save(self):
        if self.sdir != "":
            logging.info(f"Saving to {self.sdir}")
            # Copy wdir to sdir
            os.system(f"cp -r {self.wdir}/. {self.sdir}/")

    def close(self):
        # Close the socket
        self.save()
        if not self.pyconfig.mock:
            jgc.close_tcp()
        logger.info("PyCaliper run completed, socket closed.")


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


def get_specmodname(specmod):
    if "/" in specmod:
        module_name = specmod.rsplit("/", 1)[1]
        if "." in module_name:
            module_name = module_name.rsplit(".", 1)[0]
        return module_name
    return specmod


def create_module(specpath, args):
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
        if "." in module_name:
            module_name, class_name = module_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            logger.debug(
                f"Successfully imported module: {module_name} from {module_path}"
            )
            return getattr(module, class_name)(**params)
        else:
            # Import the module using importlib
            module = importlib.import_module(module_name)
            logger.debug(
                f"Successfully imported module: {module_name} from {module_path}"
            )
            return getattr(module, module_name)(**params)
    except ImportError as e:
        logger.error(f"Error importing module {module_name} from {module_path}: {e}")
        return None
    finally:
        # Clean up: remove the path from sys.path to avoid potential side effects
        sys.path.remove(module_path)


def mock_or_connect(pyconfig: PYConfig) -> bool:
    if pyconfig.mock:
        logger.info("Running in mock mode.")
        return False
    else:
        jgc.connect_tcp("localhost", pyconfig.jgc.port)
        return True


def get_pyconfig(args: PYCArgs) -> PYConfig:

    if args.jgcpath != "":
        # Create a Jasper configuration
        with open(args.jgcpath, "r") as f:
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
        jasperc = JasperConfig(
            jdir=jgconfig["jasper"]["jdir"],
            script=jgconfig["jasper"]["script"],
            pycfile=jgconfig["jasper"]["pycfile"],
            context=jgconfig["jasper"]["context"],
            design_list=jgconfig["jasper"].get("design_list", "design.lst"),
            port=jgconfig["jasper"].get("port", 8080),
        )
    else:
        jasperc = JasperConfig()

    if args.dcpath != "":
        with open(args.dcpath, "r") as f:
            dconfig = json.load(f)
        try:
            validate(instance=dconfig, schema=D_CONFIG_SCHEMA)
        except ValidationError as e:
            logger.error(f"Design config schema validation failed: {e.message}")
            logger.error(
                f"Please check schema:\n{json.dumps(D_CONFIG_SCHEMA, indent=4, sort_keys=True, separators=(',', ': '))}"
            )
            sys.exit(1)
        designc = DesignConfig(
            cpy1=dconfig["cpy1"],
            cpy2=dconfig.get("cpy2", "b"),
            lang=dconfig.get("lang", "sv12"),
            topmod=dconfig["topmod"],
            clk=dconfig.get("clk", "clk"),
        )
    else:
        designc = DesignConfig()

    return PYConfig(
        # Working directory
        sdir=args.sdir,
        # Is this a mock run
        mock=(args.jgcpath == ""),
        # Jasper configuration
        jgc=jasperc,
        # Spec config
        pycspec=get_specmodname(args.specpath),
        # Trace directory
        tdir=args.tdir,
        onetrace=args.onetrace,
        dc=designc,
    )


def setup_all(args: PYCArgs) -> tuple[bool, PYConfig, PYCManager]:
    pyconfig = get_pyconfig(args)
    tmgr = PYCManager(pyconfig)
    is_connected = mock_or_connect(pyconfig)

    return is_connected, pyconfig, tmgr


def start(task: PYCTask, args: PYCArgs) -> tuple[PYConfig, PYCManager, SpecModule]:

    is_connected, pyconfig, tmgr = setup_all(args)

    module = create_module(args.specpath, args)
    assert module is not None, f"SpecModule {args.specpath} not found."

    match task:
        case PYCTask.VERIF1T | PYCTask.VERIF2T | PYCTask.PERSYNTH | PYCTask.CTRLSYNTH:
            if not is_connected:
                logger.error(
                    f"Task {task} requires Jasper sockets, cannot be run in mock mode."
                )
                sys.exit(1)
        case _:
            pass

    return pyconfig, tmgr, module
