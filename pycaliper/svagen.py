"""
Generate SVA (wires, assumes, asserts) for specifications in PER
"""

import sys
import logging
from pydantic import BaseModel

from .per import SpecModule, Eq, CondEq, Path, Context, PER, Inv, PERHole, AuxModule
from .propns import *
from .pycconfig import DesignConfig

logger = logging.getLogger(__name__)


# Internal signals
COUNTER = "_pycinternal__counter"
STEP_SIGNAL = "_pycinternal__step"


def step_signal(k: int):
    return f"{STEP_SIGNAL}_{k}"


# def step_property(k: int):
#     return f"step_{k}"


def per_sva(mod: SpecModule, spec_ctx: Context):
    if spec_ctx == Context.INPUT:
        return f"{mod.get_hier_path('_')}_input"
    elif spec_ctx == Context.STATE:
        return f"{mod.get_hier_path('_')}_state"
    elif spec_ctx == Context.OUTPUT:
        return f"{mod.get_hier_path('_')}_output"
    else:
        logger.error(f"Invalid context {spec_ctx}")
        sys.exit(1)


def inv_sva(mod: SpecModule, spec_ctx: Context):
    if spec_ctx == Context.INPUT:
        return f"{mod.get_hier_path('_')}_input_inv"
    elif spec_ctx == Context.STATE:
        return f"{mod.get_hier_path('_')}_state_inv"
    elif spec_ctx == Context.OUTPUT:
        return f"{mod.get_hier_path('_')}_output_inv"
    else:
        logger.error(f"Invalid context {spec_ctx}")
        sys.exit(1)


class ModuleSpec(BaseModel):
    # SpecModule path
    path: Path
    input_spec_decl: str
    state_spec_decl: str
    output_spec_decl: str
    input_inv_spec_decl_comp: str
    state_inv_spec_decl_comp: str
    output_inv_spec_decl_comp: str
    input_inv_spec_decl_single: str
    state_inv_spec_decl_single: str
    output_inv_spec_decl_single: str


class SVAContext(BaseModel):
    holes: list[str] = []
    assms_2trace: list[str] = []
    asrts_2trace: list[str] = []
    assms_1trace: list[str] = []
    asrts_1trace: list[str] = []
    assms_bmc: dict[str, list[str]] = {}
    asrts_bmc: dict[str, list[str]] = {}
    seq_props: list[str] = []


class SVAGen:
    def __init__(self) -> None:
        self.topmod = None
        self.dc: DesignConfig = DesignConfig()
        self.specs: dict[Path, ModuleSpec] = {}
        self.holes: dict[str, PERHole] = {}
        self.property_context = SVAContext()

    def _reset(self):
        self.specs = {}
        self.holes = {}
        self.property_context = SVAContext()

    def _generate_decls_for_per(self, per: PER):
        declfull = per.logic.get_hier_path("_")
        if per.logic.is_arr_elem():
            declbase = per.logic.get_hier_path_nonindex()
            declsize = f"[0:{per.logic.parent.size-1}]"
        else:
            declbase = per.logic.get_hier_path("_")
            declsize = ""
        if isinstance(per, Eq):
            wirename = eq_sva(declfull)
            declname = eq_sva(declbase)
        else:
            # isinstance(per, CondEq):
            wirename = condeq_sva(declfull)
            declname = condeq_sva(declbase)
        return (wirename, declname, declsize)

    def _generate_decls_for_per_hole(self, per: PER):
        if per.logic.is_arr_elem():
            logger.error(
                f"Array elements not supported in holes: {per.logic.get_hier_path()}"
            )
            sys.exit(1)
        wirename = per._get_id_for_per_hole()
        declsize = ""
        return (wirename, wirename, declsize)

    def _gen_1t_single(self, mod: SpecModule, invs: list[Inv], spec_ctx: Context):
        inv_exprs = []
        for inv in invs:
            inv_exprs.append(inv.get_sva(self.dc.cpy1))
        inv_spec = "(\n\t" + " && \n\t".join(inv_exprs + ["1'b1"]) + ")"
        return f"wire {inv_sva(mod, spec_ctx)} = {inv_spec};"

    def _gen_1t_comp(self, mod: SpecModule, invs: list[Inv], spec_ctx: Context):
        inv_exprs = []
        for inv in invs:
            inv_exprs.append(inv.get_sva(self.dc.cpy1))
            inv_exprs.append(inv.get_sva(self.dc.cpy2))
        inv_spec = "(\n\t" + " && \n\t".join(inv_exprs + ["1'b1"]) + ")"
        return f"wire {inv_sva(mod, spec_ctx)} = {inv_spec};"

    def _gen_2t_comp(self, mod: SpecModule, pers: list[PER], spec_ctx: Context):
        assigns_ = {}
        decls_ = {}
        declwires_ = []
        for per in pers:
            (wirename, declname, declsize) = self._generate_decls_for_per(per)
            exprname = per.get_sva(self.dc.cpy1, self.dc.cpy2)
            assigns_[wirename] = f"assign {wirename} = ({exprname});"
            decls_[declname] = f"logic {declname} {declsize};"
            declwires_.append(wirename)
        svaspec = "(\n\t" + " && \n\t".join(declwires_ + ["1'b1"]) + ")"
        topdecl = f"wire {per_sva(mod, spec_ctx)} = {svaspec};"
        return (assigns_, decls_, topdecl)

    def _generate_decls_inner(self, mod: SpecModule):

        # Holes are not currently supported in submodules
        if mod != self.topmod and len(mod._pycinternal__perholes) != 0:
            logger.error(
                f"Holes not supported yet in sub-modules: found one in {mod.path}"
            )
            sys.exit(1)

        # Wire declarations for invariants
        decls = {}
        # Assignments to invariant wires
        assigns = {}
        # Generate recursively for submodules
        for _, submod in mod._pycinternal__submodules.items():
            (inner_decls, inner_assigns) = self._generate_decls_inner(submod)
            decls.update(inner_decls)
            assigns.update(inner_assigns)

        # Generate wires for current modules
        (assigns_, decls_, input_decl) = self._gen_2t_comp(
            mod, mod._pycinternal__input_tt, Context.INPUT
        )
        assigns.update(assigns_)
        decls.update(decls_)

        (assigns_, decls_, state_decl) = self._gen_2t_comp(
            mod, mod._pycinternal__state_tt, Context.STATE
        )
        assigns.update(assigns_)
        decls.update(decls_)

        (assigns_, decls_, output_decl) = self._gen_2t_comp(
            mod, mod._pycinternal__output_tt, Context.OUTPUT
        )
        assigns.update(assigns_)
        decls.update(decls_)

        for hole in mod._pycinternal__perholes:
            if hole.active:
                (wirename, declname, declsize) = self._generate_decls_for_per_hole(
                    hole.per
                )
                exprname = hole.per.get_sva(self.dc.cpy1, self.dc.cpy2)
                assigns[wirename] = f"assign {wirename} = ({exprname});"
                decls[declname] = f"logic {declname} {declsize};"

        input_inv_decl_comp = self._gen_1t_comp(
            mod, mod._pycinternal__input_invs, Context.INPUT
        )
        state_inv_decl_comp = self._gen_1t_comp(
            mod, mod._pycinternal__state_invs, Context.STATE
        )
        output_inv_decl_comp = self._gen_1t_comp(
            mod, mod._pycinternal__output_invs, Context.OUTPUT
        )

        input_inv_decl_single = self._gen_1t_single(
            mod, mod._pycinternal__input_invs, Context.INPUT
        )
        state_inv_decl_single = self._gen_1t_single(
            mod, mod._pycinternal__state_invs, Context.STATE
        )
        output_inv_decl_single = self._gen_1t_single(
            mod, mod._pycinternal__output_invs, Context.OUTPUT
        )

        self.specs[mod.path] = ModuleSpec(
            path=mod.path,
            input_spec_decl=input_decl,
            state_spec_decl=state_decl,
            output_spec_decl=output_decl,
            input_inv_spec_decl_comp=input_inv_decl_comp,
            state_inv_spec_decl_comp=state_inv_decl_comp,
            output_inv_spec_decl_comp=output_inv_decl_comp,
            input_inv_spec_decl_single=input_inv_decl_single,
            state_inv_spec_decl_single=state_inv_decl_single,
            output_inv_spec_decl_single=output_inv_decl_single,
        )

        return (decls, assigns)

    def _generate_decls(self):

        properties = []

        input_props_1t = f"{inv_sva(self.topmod, Context.INPUT)}"
        state_props_1t = f"{inv_sva(self.topmod, Context.STATE)}"
        output_props_1t = f"{inv_sva(self.topmod, Context.OUTPUT)}"

        input_props_2t = f"{per_sva(self.topmod, Context.INPUT)} && {input_props_1t}"
        state_props_2t = f"{per_sva(self.topmod, Context.STATE)} && {state_props_1t}"
        output_props_2t = f"{per_sva(self.topmod, Context.OUTPUT)} && {output_props_1t}"

        properties.append(
            f"{get_as_assm(TOP_INPUT_1T_PROP)} : assume property\n"
            + f"\t({input_props_1t});"
        )
        self.property_context.assms_1trace.append(TOP_INPUT_1T_PROP)
        properties.append(
            f"{get_as_assm(TOP_STATE_1T_PROP)} : assume property\n"
            + f"\t(!({STEP_SIGNAL}) |-> ({state_props_1t}));"
        )
        self.property_context.assms_1trace.append(TOP_STATE_1T_PROP)
        properties.append(
            f"{get_as_prop(TOP_OUTPUT_1T_PROP)} : assert property\n"
            + f"\t({STEP_SIGNAL} |-> ({state_props_1t} && {output_props_1t}));"
        )
        self.property_context.asrts_1trace.append(TOP_INPUT_1T_PROP)

        properties.append(
            f"{get_as_assm(TOP_INPUT_2T_PROP)} : assume property\n"
            + f"\t({input_props_2t});"
        )
        self.property_context.assms_2trace.append(TOP_INPUT_2T_PROP)
        properties.append(
            f"{get_as_assm(TOP_STATE_2T_PROP)} : assume property\n"
            + f"\t(!({STEP_SIGNAL}) |-> ({state_props_2t}));"
        )
        self.property_context.assms_2trace.append(TOP_STATE_2T_PROP)
        properties.append(
            f"{get_as_prop(TOP_OUTPUT_2T_PROP)} : assert property\n"
            + f"\t({STEP_SIGNAL} |-> ({state_props_2t} && {output_props_2t}));"
        )
        self.property_context.asrts_2trace.append(TOP_OUTPUT_2T_PROP)

        for hole in self.topmod._pycinternal__perholes:
            if hole.active:
                if isinstance(hole.per, Eq):
                    assm_prop = (
                        f"A_{hole.per._get_id_for_per_hole()} : assume property\n"
                        + f"\t(!({STEP_SIGNAL}) |-> {hole.per._get_id_for_per_hole()});"
                    )
                    asrt_prop = (
                        f"P_{hole.per._get_id_for_per_hole()} : assert property\n"
                        + f"\t(({STEP_SIGNAL}) |-> {hole.per._get_id_for_per_hole()});"
                    )
                    self.holes[hole.per._get_id_for_per_hole()] = hole.per.logic
                    properties.extend([assm_prop, asrt_prop])
                    self.property_context.holes.append(hole.per._get_id_for_per_hole())

                elif isinstance(hole.per, CondEq):
                    assm_prop = (
                        f"A_{hole.per._get_id_for_per_hole()} : assume property\n"
                        + f"\t(!({STEP_SIGNAL}) |-> {hole.per._get_id_for_per_hole()});"
                    )
                    asrt_prop = (
                        f"P_{hole.per._get_id_for_per_hole()} : assert property\n"
                        + f"\t(({STEP_SIGNAL}) |-> {hole.per._get_id_for_per_hole()});"
                    )
                    self.holes[hole.per._get_id_for_per_hole()] = hole.per.logic
                    properties.extend([assm_prop, asrt_prop])
                    self.property_context.holes.append(hole.per._get_id_for_per_hole())

        return properties, self._generate_decls_inner(self.topmod)

    def _generate_step_decls(self) -> list[str]:
        """
        Generate properties for each step in the simulation

        Returns:
            list[str]: List of properties for each step
        """
        properties = []
        for sched_name, sched in self.topmod._pycinternal__simsteps.items():
            steps = sched._pycinternal__steps
            self.property_context.assms_bmc[sched_name] = []
            self.property_context.asrts_bmc[sched_name] = []
            for i, step in enumerate(steps):
                assumes = [
                    expr.get_sva(self.dc.cpy1) for expr in step._pycinternal__assume
                ]
                assume_spec = "(\n\t" + " && \n\t".join(assumes + ["1'b1"]) + ")"
                asserts = [
                    expr.get_sva(self.dc.cpy1) for expr in step._pycinternal__assert
                ]
                assert_spec = "(\n\t" + " && \n\t".join(asserts + ["1'b1"]) + ")"

                # TODO: increase counter bound appropriately
                properties.append(
                    f"{get_as_assm(TOP_STEP_PROP(sched_name, i))} : assume property\n"
                    + f"\t({step_signal(i)} |-> {assume_spec});"
                )
                self.property_context.assms_bmc[sched_name].append(
                    TOP_STEP_PROP(sched_name, i)
                )
                properties.append(
                    f"{get_as_prop(TOP_STEP_PROP(sched_name, i))} : assert property\n"
                    + f"\t({step_signal(i)} |-> {assert_spec});"
                )
                self.property_context.asrts_bmc[sched_name].append(
                    TOP_STEP_PROP(sched_name, i)
                )

        return properties

    def _generate_seq_decls(self) -> list[str]:
        """
        Generate properties for each sequential implication property

        Returns:
            list[str]: List of properties for each step
        """
        properties = []
        sequences = []
        for sched_name, sched in self.topmod._pycinternal__seqprops.items():
            steps = sched._pycinternal__steps
            sched_assumes = []
            sched_asserts = []
            # self.property_context.assms_bmc[sched_name] = []
            # self.property_context.asrts_bmc[sched_name] = []
            for i, step in enumerate(steps):
                assumes = [
                    expr.get_sva(self.dc.cpy1) for expr in step._pycinternal__assume
                ]
                assume_spec = "(\n\t" + " && \n\t".join(assumes + ["1'b1"]) + ")"
                asserts = [
                    expr.get_sva(self.dc.cpy1) for expr in step._pycinternal__assert
                ]
                assert_spec = "(\n\t" + " && \n\t".join(asserts + ["1'b1"]) + ")"
                sched_assumes.append(assume_spec)
                sched_asserts.append(assert_spec)

            assm_sequence_ = " ##1 ".join(sched_assumes)
            assm_sequence = (
                f"sequence {get_assm_sequence(sched_name)};\n"
                + f"\t({assm_sequence_});\n"
                + "endsequence\n"
            )
            assert_sequence_ = " ##1 ".join(sched_asserts)
            assert_sequence = (
                f"sequence {get_assrt_sequence(sched_name)};\n"
                + f"\t({assert_sequence_});\n"
                + "endsequence\n"
            )

            sequences.append(assm_sequence)
            sequences.append(assert_sequence)

            properties.append(
                f"{get_as_prop(TOP_SEQ_PROP(sched_name))} : assert property\n"
                + f"\t({get_assm_sequence(sched_name)} |-> {get_assrt_sequence(sched_name)});"
            )
            self.property_context.seq_props.append(TOP_SEQ_PROP(sched_name))

        return properties, sequences

    def counter_step(self, kd: int, td: int):
        """
        Generate counter and step signals for the simulation.

        Args:
            kd (int): Number of steps for inductions (generate induction trigger)
            td (int): Number of steps across all simulations schedules + induction
        """
        counter_width = len(bin(td)) - 2
        vtype = f"logic [{counter_width-1}:0]" if counter_width > 1 else "logic"
        vlog = f"""
\t{vtype} {COUNTER};
\talways @(posedge clk) begin
\t    if (fvreset) begin
\t        {COUNTER} <= 0;
\t    end else begin
\t        if ({COUNTER} < {counter_width}'d{td}) begin
\t            {COUNTER} <= ({COUNTER} + {counter_width}'b1);
\t        end
\t    end
\tend
\tlogic {STEP_SIGNAL} = ({COUNTER} == {counter_width}'d{kd});
"""
        for i in range(td):
            vlog += f"\tlogic {step_signal(i)} = ({COUNTER} == {counter_width}'d{i});\n"
        return vlog

    def create_pyc_specfile(
        self,
        topmod: SpecModule,
        dc: DesignConfig,
        filename="temp.pyc.sv",
        onetrace=False,
    ):
        assert topmod.is_instantiated(), "Top module must be instantiated"
        self._reset()
        self.topmod = topmod
        self.dc = dc

        kd, td = self.topmod.get_unroll_kind_depths()
        vlog = self.counter_step(kd, td)

        properties, all_decls = self._generate_decls()
        properties.extend(self._generate_step_decls())

        seq_props, seq_decls = self._generate_seq_decls()
        properties.extend(seq_props)

        aux_modules = []
        # Get auxiliary modules if any
        for _, aux_mod in self.topmod._pycinternal__auxmodules.items():
            if aux_mod.is_leftcopy():
                aux_modules.append(aux_mod.get_instance_str(dc.cpy1))
            else:
                aux_modules.append(aux_mod.get_instance_str(dc.cpy2))

        with open(filename, "w") as f:
            f.write(vlog + "\n")

            for assign in all_decls[0].values():
                f.write(assign + "\n")
            for decl in all_decls[1].values():
                f.write(decl + "\n")

            # Write auxiliary modules
            f.write("\n")
            f.write("/////////////////////////////////////\n")
            f.write("// Auxiliary modules\n")
            for aux_mod in aux_modules:
                f.write(aux_mod + "\n")

            for mod, spec in self.specs.items():
                f.write("\n")
                f.write(f"/////////////////////////////////////\n")
                f.write(f"// SpecModule {mod.get_hier_path()}\n")
                f.write("\n")
                f.write(spec.input_spec_decl + "\n")
                f.write(spec.state_spec_decl + "\n")
                f.write(spec.output_spec_decl + "\n")
                if not onetrace:
                    f.write(spec.input_inv_spec_decl_comp + "\n")
                    f.write(spec.state_inv_spec_decl_comp + "\n")
                    f.write(spec.output_inv_spec_decl_comp + "\n")
                else:
                    f.write(spec.input_inv_spec_decl_single + "\n")
                    f.write(spec.state_inv_spec_decl_single + "\n")
                    f.write(spec.output_inv_spec_decl_single + "\n")

            f.write("\n")
            f.write("/////////////////////////////////////\n")
            f.write("// Sequences for top module\n")
            f.write("\n\n".join(seq_decls))
            f.write("\n")
            f.write("/////////////////////////////////////\n")
            f.write("// Assumptions and Assertions for top module\n")
            f.write("\n\n".join(properties))
            f.write("\n")

        logger.info(f"Generated spec file: {filename}")
