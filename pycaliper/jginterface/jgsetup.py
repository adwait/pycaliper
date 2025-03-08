import pathlib
import logging

from ..pycconfig import DesignConfig, JasperConfig

logger = logging.getLogger(__name__)

HARNESS_CLK = "clk"
HARNESS_RST = "rst"
FVRESET = "fvreset"


def _create_tcl_script(
    lang: str = "sv12",
    design_lst_file: str = "design.lst",
    harnessmod: str = "miter",
    max_trace_len: int = 3,
) -> str:
    return f"""clear -all

# Disable some info messages and warning messages
# set_message -disable VERI-9033 ; # array automatically black-boxed
# set_message -disable WNL008 ; # module is undefined. All instances will be black-boxed
# set_message -disable VERI-1002 ; # Can't disable this error message (net does not have a driver)
# set_message -disable VERI-1407 ; # attribute target identifier not found in this scope
set_message -disable VERI-1018 ; # info
set_message -disable VERI-1328 ; # info
set_message -disable VERI-2571 ; # info
# set_message -disable VERI-2571 ; # info: disabling old hierarchical reference handler
set_message -disable INL011 ; # info: processing file
# set_message -disable VERI-1482 ; # analyzing verilog file
set_message -disable VERI-1141 ; # system task is not supported
set_message -disable VERI-1060 ; # 'initial' construct is ignored
set_message -disable VERI-1142 ; # system task is ignored for synthesis
# set_message -disable ISW003 ; # top module name
# set_message -disable HIER-8002 ; # disabling old hierarchical reference handler
set_message -disable WNL046 ; # renaming embedded assertions due to name conflicts
set_message -disable VERI-1995 ; # unique/priority if/case is not full
                                 # (we check these conditions with the elaborate
                                 #  option -extract_case_assertions)

set JASPER_FILES {{
    {harnessmod}.sv
}}

set env(DESIGN_HOME) [pwd]
set err_status [catch {{analyze -{lang} +define+INVARIANTS +define+JASPER +define+SYNTHESIS +libext+.v+.sv+.vh+.svh+ -f {design_lst_file} {{*}}$JASPER_FILES}} err_msg]
if $err_status {{error $err_msg}}

elaborate \
    -top {harnessmod} \
    -no_preconditions \
    -extract_case_assertions \
    -disable_auto_bbox

proc write_reset_seq {{file}} {{
    puts $file "{FVRESET} 1'b1"
    puts $file 1
    puts $file "{FVRESET} 1'b0"
    puts $file {{$}}
    close $file
}}

proc reset_formal {{}} {{
    write_reset_seq  [open "reset.rseq" w]
    # reset -expression {FVRESET}
    reset -sequence "reset.rseq"
}}


clock clk

# Constrain primary inputs to only change on @(posedge eph1)
clock -rate -default clk

reset_formal

# Set default Jasper proof engines (overrides use_nb engine settings)
set_engine_mode {{Ht}}

set_max_trace_length {max_trace_len}

"""


def _create_design_instance(dc: DesignConfig, inst_name: str) -> str:
    return f"""
{dc.topmod} {inst_name} (
    .{dc.clk}({HARNESS_CLK})
);
"""


def _create_proof_harness_miter(pycfile: str, harnessmod: str, dc: DesignConfig) -> str:
    return f"""
// Parent module with a miter with different inputs
module {harnessmod} (
    input wire {HARNESS_CLK}
    , input wire {HARNESS_RST}
);

    {_create_design_instance(dc, dc.cpy1)}

    {_create_design_instance(dc, dc.cpy2)}


    default clocking cb @(posedge {HARNESS_CLK});
    endclocking // cb

    logic {FVRESET};

    `include "{pycfile}"

endmodule
"""


def _create_proof_harness_single(
    pycfile: str, harnessmod: str, dc: DesignConfig
) -> str:
    return f"""
// Parent module with a single design
module {harnessmod} (
    input wire {HARNESS_CLK}
    , input wire {HARNESS_RST}
);

    {_create_design_instance(dc, dc.cpy1)}


    default clocking cb @(posedge {HARNESS_CLK});
    endclocking // cb

    logic {FVRESET};

    `include "{pycfile}"

endmodule
"""


def setup_jasper(dc: DesignConfig, jgc: JasperConfig):

    if dc.topmod == "":
        logger.warning("Top module name is not set, skipping Jasper setup.")
        return False
    elif dc.cpy1 == "":
        logger.warning("First instance name is not set, skipping Jasper setup.")
        return False
    elif dc.cpy1 == dc.cpy2:
        logger.warning(
            "First and second instance names are the same, skipping Jasper setup."
        )
        return False

    jgdir = pathlib.Path(jgc.jdir)

    # Design file is relative to the Jasper working directory
    lstfile = (jgdir / jgc.design_list).resolve()
    # Make sure that the design list file exists
    if not lstfile.exists():
        raise FileNotFoundError(f"Design list file '{lstfile}' does not exist.")

    if jgc.context.startswith("<embedded>::"):
        # Truncate the context name
        harnessmod = jgc.context[12:]
    else:
        # Complain
        logger.error(
            f"Context name '{jgc.context}' is not supported, must start with '<embedded>::'."
        )

    # If TCL file already exists, overwrite
    # Create the TCL script
    tclfile = (jgdir / jgc.script).resolve()
    with open(tclfile, "w") as f:
        f.write(
            _create_tcl_script(
                design_lst_file=lstfile.as_posix(), harnessmod=harnessmod, lang=dc.lang
            )
        )

    # Write the harness module to a file (if already exists, overwrite)
    harnessfile = (jgdir / f"{harnessmod}.sv").resolve()
    # Create the harness module
    if dc.cpy2 != "":
        harness = _create_proof_harness_miter(jgc.pycfile, harnessmod, dc)
    else:
        harness = _create_proof_harness_single(jgc.pycfile, harnessmod, dc)
    with open(harnessfile, "w") as f:
        f.write(harness)

    return True
