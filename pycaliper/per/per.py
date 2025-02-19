"""
    PyCaliper

    Author: Adwait Godbole, UC Berkeley

    File: per/per.py

    Internal representation classes for PyCaliper:
        PER (Partial Equivalence Relations):
            A partial equivalence relation defined through equality and
                conditional equality assertions.
        Logic:
            Single bitvectors
        Struct:
            Support for SystemVerilog structs
        SpecModule:
            A module in the specification hierarchy, what else can it be?
"""

import logging
import sys

from enum import Enum
from typing import Callable
import copy
from dataclasses import dataclass

import inspect
from functools import wraps
from textwrap import indent

from pycaliper.per.expr import Expr, Const

logger = logging.getLogger(__name__)

UNKNOWN_WIDTH = -1

RESERVED = ["except", "try", "else", "if", "elif"]


def nonreserved_or_fresh(name: str):
    if name in RESERVED:
        return f"{name}_"
    return name


@dataclass
class Path:
    """Path: representing a hierarchical path in the specification.

    path: lis[tuple[str, list]]: at each hierarchical level, string represents identifier and
        the list represents the indexing (if unindexed, then the index is and empty list [])
    slicehigh: int: high index of the slice (default 0)
    slicelow: int: low index of the slice (default 0)
    """

    path: list[tuple[str, list]]
    # Slice of bitvector: low and high indices
    slicelow: int = -1
    slicehigh: int = -1

    def get_hier_path(self, sep=".") -> str:
        """Get the hierarchical path as a string

        Args:
            sep (str, optional): Separator in generated path string. Defaults to '.'.

        Returns:
            str: Path string.
        """
        # No slicing
        if self.slicelow == -1 and self.slicehigh == -1:
            slicestr = ""
        # Single bit
        elif self.slicelow == -1:
            slicestr = f"[{self.slicehigh}]" if sep == "." else f"_{self.slicehigh}"
        # Proper slice
        else:
            slicestr = (
                f"[{self.slicehigh}:{self.slicelow}]"
                if sep == "."
                else f"_{self.slicehigh}_{self.slicelow}"
            )
        # Base signal string
        if len(self.path) == 0:
            basepath = []
        elif sep == ".":
            basepath = [
                f"{s}{''.join([f'[{i}]' for i in inds])}" for (s, inds) in self.path
            ]
        else:
            basepath = [
                f"{s}{''.join([f'_{i}' for i in inds])}" for (s, inds) in self.path[:-1]
            ]
            basepath += [
                f"{self.path[-1][0]}{''.join([f'[{i}]' for i in self.path[-1][1]])}"
            ]
        return f"{sep.join(basepath)}{slicestr}"

    def get_hier_path_nonindex(self) -> str:
        """Get the hierarchical path string without last level index. Uses '_' as separator.
            For example: a.b[0].c[1] -> a_b_0_c

        Returns:
            str: Path string.
        """
        basepath = [
            f"{s}{''.join([f'_{i}' for i in inds])}" for (s, inds) in self.path[:-1]
        ]
        return f"{'_'.join(basepath + [self.path[-1][0]])}"

    def get_hier_path_flatindex(self) -> str:
        """Get the hierarchical path string with all indices flattened. Uses '_' as separator.
            For example: a.b[0].c[1] -> a_b_0_c_1

        Returns:
            str: Path string.
        """
        basepath = [f"{s}{''.join([f'_{i}' for i in inds])}" for (s, inds) in self.path]
        return f"{'_'.join(basepath)}"

    def add_level_index(self, i: int) -> "Path":
        """Add an index to the last level of the path. For example, a.b[0].c -> a.b[0].c[1]

        Args:
            i (int): index to be added

        Returns:
            Path: new path with the index added.
        """
        lastlevel = (self.path[-1][0], self.path[-1][1] + [i])
        return Path(self.path[:-1] + [lastlevel], self.slicelow, self.slicehigh)

    def add_level(self, name: str) -> "Path":
        """Add a new level to the path. For example, a.b[0] -> a.b[0].c

        Args:
            name (str): name of the new level

        Returns:
            Path: new path with the level added.
        """
        return Path(self.path + [(name, [])], self.slicelow, self.slicehigh)

    def __hash__(self) -> int:
        # Hash (required for dataclasses) based on the path string
        return hash(self.get_hier_path())


def get_path_from_hierarchical_str(pathstr: str) -> Path:
    """Get a path with a single level
    Args: name (str): name of the singleton path
    Returns: Path object
    """
    return Path([(name, []) for name in pathstr.split(".")])


class TypedElem:
    """An element in the design hierarchy (Logic, LogicArray, Struct, ...) with a type."""

    def __init__(self, root: str = None):
        self.name = ""
        self.root = root
        pass

    def instantiate(self, path: Path):
        logger.error(f"Not implemented instantiate in class: {self.__class__.__name__}")
        sys.exit(1)

    def _typ(self):
        # This is only used for pprinting the type of the element
        logger.error(f"Not implemented _typ in class: {self.__class__.__name__}")
        sys.exit(1)


class Logic(Expr, TypedElem):
    """Class for single bitvectors/signals"""

    def __init__(self, width: int = 1, name: str = "", root: str = None) -> None:
        """
        Args:
            width (int, optional): width of logic signal. Defaults to 1.
            name (str, optional): signal name. Defaults to ''; this is overwritten
                at instantiation time by using the introsepcted attribute name in the parent class.
        """
        # Width of this element, default is 1, UNKNOWN_WIDTH is used for unknown width
        self.width: int = width
        self.name = name
        # Hierarchical path
        self.path: Path = Path([])
        # If member of a logic array, this is the parent array
        self.parent = None
        # Root
        self.root = root

    def instantiate(self, path: Path, parent: "LogicArray" = None) -> "Logic":
        """Instantiate the logic signal with a path and parent array

        Args:
            path (Path): Path object representing the hierarchical path
            parent (LogicArray, optional): Parent array. Defaults to None.

        Returns:
            Logic: return self
        """
        self.path = path
        self.parent = parent
        return self

    def _typ(self):
        if self.width == 1:
            return "logic"
        else:
            return f"logic [{self.width-1}:0]"

    # Call corresponding functions in Path
    def get_hier_path(self, sep: str = "."):
        return self.path.get_hier_path(sep)

    def get_hier_path_nonindex(self):
        return self.path.get_hier_path_nonindex()

    def get_hier_path_flatindex(self):
        return self.path.get_hier_path_flatindex()

    def get_sva(self, pref: str = "a") -> str:
        """
        Args:
            pref (str, optional): Top-level module prefix string. Defaults to 'a'.

        Returns:
            str: SVA representation of the signal.
        """
        if self.root is not None:
            return f"{self.root}.{self.get_hier_path()}"
        return f"{pref}.{self.get_hier_path()}"

    def is_arr_elem(self) -> bool:
        """Check if this signal is an element of an array (inspects the path to see if last level is indexed).

        Returns:
            bool: True if the signal is an array element.
        """
        return len(self.path.path[-1][1]) > 0

    def __str__(self) -> str:
        return self.get_hier_path("_")

    def __repr__(self):
        return f"self.{self.get_hier_path()}"

    def __call__(self, hi: int, lo: int = -1) -> "Logic":
        """Slice the signal

        Args:
            hi (int): high index of the slice
            lo (int, optional): low index. Defaults to -1 (which means that low index is unsliced).

        Returns:
            Logic: a new signal object representing the slice
        """
        if hi >= self.width or lo < -1 or hi < lo:
            logger.error("Out of bounds: hi=%d, lo=%d, width=%d", hi, lo, self.width)
            sys.exit(1)
        else:
            slicedsig = copy.deepcopy(self)
            slicedsig.path.slicehigh = hi
            slicedsig.path.slicelow = lo
            return slicedsig

    def __hash__(self) -> int:
        # Hash based on the path string.
        return hash(self.get_hier_path())


class LogicArray(TypedElem):
    """An array of logic signals"""

    def __init__(
        self,
        typ_const: Callable[[], TypedElem],
        size: int,
        base: int = 0,
        name: str = "",
        root: str = None,
    ):
        """An array of logic signals

        Args:
            typ_const (Callable[[], TypedElem]): function that returns a TypedElem object
            size (int): size of the array
            base (int, optional): base index. Defaults to 0.
            name (str, optional): array basename. Defaults to ''.
        """
        self.typ: Callable[[], TypedElem] = typ_const
        self.name: str = name
        self.path: Path = Path([])
        self.size: int = size
        self.base: int = base
        self.logic = [typ_const() for _ in range(size)]

    def instantiate(self, path: Path):
        self.path = path
        for i, o in enumerate(self.logic):
            o.name = f"{self.name}[{i+self.base}]"
            o.instantiate(path.add_level_index(i + self.base), self)
        return self

    def _typ(self):
        return f"{self.typ()._typ()} [{self.base}:{self.base+self.size-1}]"

    def get_hier_path(self, sep: str = "."):
        return self.path.get_hier_path(sep)

    def __getitem__(self, key: int):
        """
        Args:
            key (int): index of the signal, offset by the base index

        Returns:
            TypedElem: signal at the given index
        """
        return self.logic[key - self.base]

    def __str__(self):
        return self.path.get_hier_path()

    def __repr__(self):
        return f"self.{self.name}"

    def __call__(self, index: int) -> TypedElem:
        """
        Args:
            index (int): index of the signal, offset by the base index

        Returns:
            TypedElem: signal at the given index
        """
        return self.logic[index - self.base]


class Prop:
    """Property class"""

    def __init__(self):
        pass

    def get_sva(self, pref: str = "a"):
        raise NotImplementedError(
            "Method not implemented for abstract base Prop class."
        )

    def __repr__(self):
        raise NotImplementedError(
            "Method not implemented for abstract base Prop class."
        )


# Partial equivalence relation
class PER(Prop):
    """Partial Equivalence Relation (PER) base class"""

    def __init__(self) -> None:
        self.logic: Logic = None

    def __str__(self) -> str:
        raise NotImplementedError("Method not implemented for abstract base PER class.")

    def get_sva(self, cpy1: str, cpy2: str):
        raise NotImplementedError("Method not implemented for abstract base PER class.")


class Eq(PER):
    """Relational equality assertion."""

    def __init__(self, logic: TypedElem) -> None:
        """
        Args:
            logic (TypedElem): the element to be equated
        """
        super().__init__()
        # TODO: add support for non-Logic (struct) types
        if not isinstance(logic, Logic):
            logger.error(f"Invalid PER type: {logic}, currently only Logic supported.")
            sys.exit(1)
        self.logic = logic

    def __str__(self) -> str:
        return f"eq({self.logic})"

    def get_sva(self, cpy1: str = "a", cpy2: str = "b") -> str:
        """
        Args:
            cpy1 (str, optional): Hierarchy prefix of left copy. Defaults to 'a'.
            cpy2 (str, optional): Hierarchy prefix of right copy. Defaults to 'b'.

        Returns:
            str: SVA representation of the equality assertion.
        """
        return f"{self.logic.get_sva(cpy1)} == {self.logic.get_sva(cpy2)}"

    def __repr__(self):
        return f"self.eq({repr(self.logic)})"


class CondEq(PER):
    """Conditional equality assertion"""

    def __init__(self, cond: Expr, logic: Logic) -> None:
        super().__init__()
        self.cond = cond
        self.logic = logic

    def __str__(self) -> str:
        return f"condeq({self.cond}, {self.logic})"

    def get_sva(self, cpy1: str = "a", cpy2: str = "b") -> str:
        """Get the SVA representation of the conditional equality assertion."""
        return (
            f"!({self.cond.get_sva(cpy1)} && {self.cond.get_sva(cpy2)}) | "
            + f"({self.logic.get_sva(cpy1)} == {self.logic.get_sva(cpy2)})"
        )

    def __repr__(self):
        return f"self.when({repr(self.cond)})({repr(self.logic)})"


class Inv(Prop):
    """Invariant class"""

    def __init__(self, expr: Expr):
        self.expr = expr

    def get_sva(self, pref: str = "a"):
        return self.expr.get_sva(pref)

    def __repr__(self):
        return f"self.inv({repr(self.expr)})"


class Struct(TypedElem):
    """Struct as seen in SV"""

    def __init__(self, name="", root: str = None, **kwargs) -> None:
        self.name = name
        self.params = kwargs
        self.path = Path([])
        self._pycinternal__signals: dict[str, TypedElem] = {}
        self._pycinternal__state_tt: list[PER] = []
        self.root = root

    def _typ(self):
        return self.__class__.__name__

    def get_sva(self, pref: str = "a") -> str:
        if self.root is not None:
            return f"{self.root}.{self.name}"
        return f"{pref}.{self.name}"

    def __str__(self) -> str:
        return self.name

    def state(self):
        # Equivalence class definition for this struct
        pass

    def eq(self, expr: Expr) -> None:
        ceq = Eq(expr)
        self._pycinternal__state_tt.append(ceq)

    def when(self, cond: Expr):
        def _lambda(*pers: PER):
            for per in pers:
                if isinstance(per, Logic):
                    ceqs = [CondEq(cond, per)]
                elif isinstance(per, CondEq):
                    ceqs = [CondEq(cond & per.cond, per.logic)]
                elif isinstance(per, LogicArray):
                    ceqs = [CondEq(cond, p) for p in per.logic]
                else:
                    logger.error(f"Invalid PER type: {per}")
                    sys.exit(1)
            self._pycinternal__state_tt.extend(ceqs)

        return _lambda

    def instantiate(self, path: Path):
        self.path = path
        sigattrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if (
                isinstance(obj, Logic)
                or isinstance(obj, LogicArray)
                or isinstance(obj, Struct)
            ):
                # Assign name if not provided during declaration
                if obj.name == "":
                    obj.name = attr
                sigattrs[obj.name] = obj.instantiate(path.add_level(obj.name))
        self._pycinternal__signals = sigattrs
        # INFO: This is not yet supported.
        # TODO: support state equality definitions for structs
        # self.state()
        return self

    def _typ(self) -> str:
        return f"{self.__class__.__name__}"

    def sprint(self) -> str:
        """Pretty string for the struct definition"""
        s = ""
        s += f"struct {self.name}({self.__class__.__name__})"
        s += "signals:\n"
        for k, v in self._pycinternal__signals.items():
            s += f"\t{k} : {v._typ()}\n"
        s += "state:\n"
        for i in self._pycinternal__state_tt:
            s += f"\t{i}\n"
        return s

    def pprint(self):
        """Pretty print the struct definition"""
        print(self.sprint())

    def get_repr(self, reprs):
        reprs[self.__class__.__name__] = repr(self)
        for s, t in self._pycinternal__signals.items():
            if isinstance(t, Struct):
                if t.__class__.__name__ not in reprs:
                    reprs = t.get_repr(reprs)
        return reprs

    def __repr__(self):
        """Generate Python code for the struct definition"""
        inits = ["\tdef __init__(self, name = ''):", f"\t\tsuper().__init__(name)"]
        for s, t in self._pycinternal__signals.items():
            if isinstance(t, Logic):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = Logic({t.width}, "{t.name}")'
                )
            elif isinstance(t, LogicArray):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = LogicArray(lambda: Logic({t.typ().width}), {t.size}, "{t.name}")'
                )
            elif isinstance(t, Struct):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = {t.__class__.__name__}("{t.name}")'
                )
            else:
                logger.error(f"Invalid signal type: {t}")
                sys.exit(1)
        initstring = "\n".join(inits)

        return f"""
class {self.__class__.__name__}(Struct):

{initstring}
        """


class Group:
    """A hierarchical group; does not have any hierarchical position associated with itself."""

    def __init__(self, name: str = ""):
        self.name = name
        self._elems: dict[str, TypedElem] = {}

    def instantiate(self, path: Path):
        for attr in dir(self):
            obj = getattr(self, attr)
            if (
                isinstance(obj, Logic)
                or isinstance(obj, LogicArray)
                or isinstance(obj, Struct)
                or isinstance(obj, SpecModule)
            ):
                if obj.name == "":
                    obj.name = attr
                self._elems[obj.name] = obj.instantiate(path.add_level(obj.name))
        return self

    def get_repr(self, reprs):
        reprs[self.__class__.__name__] = repr(self)
        for s, t in self._elems.items():
            if isinstance(t, Struct) or isinstance(t, SpecModule):
                if t.__class__.__name__ not in reprs:
                    reprs = t.get_repr(reprs)
        return reprs

    def __repr__(self):

        inits = ["\tdef __init__(self, name = ''):", f"\t\tsuper().__init__(name)"]
        for s, t in self._elems.items():
            if isinstance(t, Logic):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = Logic({t.width}, "{t.name}")'
                )
            elif isinstance(t, LogicArray):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = LogicArray(lambda: Logic({t.typ().width}), {t.size}, "{t.name}")'
                )
            elif isinstance(t, Struct):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = {t.__class__.__name__}("{t.name}")'
                )
            elif isinstance(t, SpecModule):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = {t.__class__.__name__}("{t.name}", {t.params})'
                )
            else:
                logger.error(f"Invalid signal type: {t}")
                sys.exit(1)
        initstring = "\n".join(inits)

        return f"""
class {self.__class__.__name__}(Group):

{initstring}
        """


class SVFuncApply(Expr):
    """Apply a SystemVerilog function to a list of arguments"""

    def __init__(self, func: "SVFunc", args: tuple[Expr]) -> None:
        self.func = func
        self.args = args

    def get_sva(self, pref: str = "a") -> str:
        """Get the SVA representation of the function application."""
        return f"{self.func}({', '.join([a.get_sva(pref) for a in self.args])})"


class SVFunc:
    def __init__(self, name=""):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __call__(self, *args):
        return SVFuncApply(self, args)


class Context(Enum):
    """Context of specification declarations made within a module."""

    INPUT = 0
    STATE = 1
    OUTPUT = 2
    UNROLL = 3


class Hole:
    """A synthesis hole"""

    def __init__(self):
        self.active = True
        pass

    def deactivate(self):
        self.active = False


class PERHole(Hole):
    """A synthesis hole for a PER"""

    def __init__(self, per: PER, ctx: Context):
        super().__init__()
        self.per = per
        self.ctx = ctx

    def __repr__(self):
        if isinstance(self.per, Eq):
            return f"self.eqhole([{repr(self.per.logic)}])"
        elif isinstance(self.per, CondEq):
            return f"self.condeqhole({repr(self.per.cond)}, [{repr(self.per.logic)}])"


class CondEqHole(Hole):
    """A synthesis hole for a conditional equality PER"""

    def __init__(self, cond: Expr, per: PER, ctx: Context):
        super().__init__()
        self.cond = cond
        self.per = per
        self.ctx = ctx

    def __repr__(self):
        return f"self.when({repr(self.cond)})({repr(self.per)})"


class CtrAlignHole(Hole):
    def __init__(self, ctr: Logic, sigs: list[Logic]):
        super().__init__()
        self.ctr = ctr
        self.sigs = sigs

    def __repr__(self) -> str:
        return f"self.ctralignhole({repr(self.ctr)}, {repr(self.sigs)})"


class FSMHole(Hole):
    def __init__(self, guardsigs: list[Logic], statesig: Logic):
        super().__init__()
        self.guardsigs = guardsigs
        self.statesig = statesig

    def __repr__(self) -> str:
        return f"self.fsmhole({repr(self.guardsigs)}, {repr(self.statesig)})"


def when(cond: Expr):
    """Create a Conditional equality PER generator. This returns a lambda that returns a CondEq object.

    Args:
        cond (Expr): the condition to apply the PER under
    """

    def _lambda(per: PER):
        if isinstance(per, Eq):
            return CondEq(cond, per.logic)
        elif isinstance(per, CondEq):
            return CondEq(cond & per.cond, per.logic)
        else:
            logger.error(f"Invalid PER type: {per}")
            sys.exit(1)

    return _lambda


class SimulationStep:
    def __init__(self) -> None:
        self._pycinternal__assume: list[Expr] = []
        self._pycinternal__assert: list[Expr] = []

    def _assume(self, expr: Expr):
        self._pycinternal__assume.append(expr)

    def _assert(self, expr: Expr):
        self._pycinternal__assert.append(expr)

    def __repr__(self) -> str:
        stepstr = [f"\t\tself.pycassume({repr(e)})" for e in self._pycinternal__assume]
        stepstr += [f"\t\tself.pycassert({repr(e)})" for e in self._pycinternal__assert]
        return "\n".join(stepstr)


class SimulationSchedule:
    def __init__(self, name: str, depth: int):
        self.name = name
        self.depth = depth
        self._pycinternal__steps: list[SimulationStep] = []

    def step(self, step: SimulationStep):
        self._pycinternal__steps.append(copy.deepcopy(step))

    def __repr__(self) -> str:
        fnstrs = [f"\t@unroll({self.depth})", f"\tdef {self.name}(self, i: int):"]
        for i, step in enumerate(self._pycinternal__steps):
            fnstrs.append(f"\t\t# Step {i}")
            fnstrs.append(f"\t\tif i == {i}:")
            fnstrs.append(indent(repr(step), "\t"))
        return "\n".join(fnstrs)


def kinduct(b: int):
    """Decorator to set the k-induction depth for a state invariant specification."""

    def kind_decorator(func):
        @wraps(func)
        def wrapper(self: SpecModule):
            func(self)

        wrapper._pycinternal__kind_depth = b
        return wrapper

    return kind_decorator


def unroll(b: int):
    """Decorator to set the unroll depth for a function in a SpecModule."""

    def unroll_decorator(func):
        @wraps(func)
        def wrapper(self: SpecModule, i: int):
            func(self, i)

        wrapper._pycinternal__is_unroll_schedule = True
        wrapper._pycinternal__unroll_depth = b

        return wrapper

    return unroll_decorator


class SpecModule:
    """SpecModule class for specifications related to a SV HW module"""

    def __init__(self, name="", **kwargs) -> None:
        """
        Args:
            name (str, optional): non-default module name in the hierarchy. Defaults to '' which is
                overriden by the attribute name.
        """
        # Instance name in the hierarchy
        self.name = name
        # Hierarchical path
        self.path: Path = Path([])
        # Parameters of the module
        self.params = kwargs

        # Internal (private) members

        # Helper variables
        # Has this module been elaborated (based on a top module)
        self._pycinternal__instantiated = False
        # Helper variable to track current declaration/specification scope
        self._pycinternal__context = Context.INPUT
        # Helper variable for unrolling
        self._pycinternal__simstep: SimulationStep = SimulationStep()

        # Objects in the SpecModule: Logic, LogicArray, Struct
        self._pycinternal__signals: dict[str, TypedElem] = {}
        self._pycinternal__groups: dict[str, Group] = {}
        self._pycinternal__functions: dict[str, SVFunc] = {}
        self._pycinternal__submodules: dict[str, SpecModule] = {}
        # PER specifications for input, state and output scopes
        self._pycinternal__input_tt: list[PER] = []
        self._pycinternal__state_tt: list[PER] = []
        self._pycinternal__output_tt: list[PER] = []
        # Single trace specifications for input, state and output scopes
        self._pycinternal__input_invs: list[Inv] = []
        self._pycinternal__state_invs: list[Inv] = []
        self._pycinternal__output_invs: list[Inv] = []
        # Non-invariant properties: mapping from caller function name to a symbolic schedule
        self._pycinternal__simsteps: dict[str, SimulationSchedule] = {}

        # PER holes
        self._pycinternal__perholes: list[PERHole] = []
        # CondEq holes
        # self._condeqholes: list[CondEqHole] = []
        # CtrAlign holes
        self._pycinternal__caholes: list[CtrAlignHole] = []
        # FSM holes
        self._pycinternal__fsmholes: list[FSMHole] = []

        self._pycinternal__auxmodules: dict[str, AuxModule] = {}
        # self._auxregs = dict[str, AuxReg]
        self._pycinternal__prev_signals = {}

    # Invariant functions to be overloaded by descendant specification classes
    def input(self) -> None:
        pass

    def state(self) -> None:
        pass

    def output(self) -> None:
        pass

    def eq(self, *elems: TypedElem) -> None:
        """Create an relational equality invariant"""
        eqs = []
        for elem in elems:
            if isinstance(elem, Logic):
                eqs.append(Eq(elem))
            elif isinstance(elem, LogicArray):
                eqs.extend([Eq(l) for l in elem.logic])
            elif isinstance(elem, Struct):
                logger.error("Structs are not yet supported in Eq invariants.")
        if self._pycinternal__context == Context.INPUT:
            self._pycinternal__input_tt.extend(eqs)
        elif self._pycinternal__context == Context.STATE:
            self._pycinternal__state_tt.extend(eqs)
        elif self._pycinternal__context == Context.OUTPUT:
            self._pycinternal__output_tt.extend(eqs)
        else:
            raise Exception("Invalid context")

    def _eq(self, elem: Logic, ctx: Context) -> None:
        """Add equality property with the specified context.

        Args:
            elem (Logic): the invariant expression.
            ctx (Context): the context to add this invariant under.
        """
        if ctx == Context.INPUT:
            self._pycinternal__input_invs.append(Eq(elem))
        elif ctx == Context.STATE:
            self._pycinternal__state_invs.append(Eq(elem))
        else:
            self._pycinternal__output_invs.append(Eq(elem))

    def when(self, cond: Expr) -> Callable:
        """Conditional equality PER.

        Args:
            cond (Expr): the condition Expr to enforce nested PER under.

        Returns:
            Callable: a lambda that is applied to the nested PER.
        """

        def _lambda(*pers: PER):
            for per in pers:
                if isinstance(per, Logic):
                    ceqs = [CondEq(cond, per)]
                elif isinstance(per, CondEq):
                    ceqs = [CondEq(cond & per.cond, per.logic)]
                elif isinstance(per, LogicArray):
                    ceqs = [CondEq(cond, p) for p in per.logic]
                else:
                    logger.error(f"Invalid PER type: {per}")
                    sys.exit(1)
                if self._pycinternal__context == Context.INPUT:
                    self._pycinternal__input_tt.extend(ceqs)
                elif self._pycinternal__context == Context.STATE:
                    self._pycinternal__state_tt.extend(ceqs)
                elif self._pycinternal__context == Context.OUTPUT:
                    self._pycinternal__output_tt.extend(ceqs)

        return _lambda

    def prev(self, elem: Logic) -> Logic:
        """Get the previous value of a signal.

        Args:
            elem (Logic): the signal to get the previous value of.

        Returns:
            Logic: the previous value of the signal.
        """
        # Check that signal in current specification
        if (
            not isinstance(elem, Logic)
            or elem.name not in self._pycinternal__signals
            or elem.is_arr_elem()
        ):
            logger.error(f"In prev: signal {elem.name} not found or is incompatible.")
            sys.exit(1)
        # Check if the previous signal has already been created
        if elem.name in self._pycinternal__prev_signals:
            return self._pycinternal__prev_signals[elem.name]
        # Create a new signal with the same name and path
        prevsig = copy.deepcopy(elem)
        prevsig.name = f"prev_{elem.name}"
        # Add the signal to the current specification
        self._pycinternal__signals[prevsig.name] = prevsig
        # Add to the previous signals dictionary
        self._pycinternal__prev_signals[elem.name] = prevsig
        return prevsig

    def incr(self, x: Logic):
        return (self.prev(x) + Const(1, x.width)) == x

    def stable(self, x: Logic):
        return self.prev(x) == x

    def eqhole(self, exprs: list[Expr]):
        """Creates an Eq (synthesis) hole.

        Args:
            exprs (list[Expr]): the list of expressions to consider as candidates for filling this hole.
        """
        if (
            self._pycinternal__context == Context.INPUT
            or self._pycinternal__context == Context.OUTPUT
        ):
            logger.error("Holes in input/output contexts currently not supported")
            sys.exit(1)
        for expr in exprs:
            self._pycinternal__perholes.append(
                PERHole(Eq(expr), self._pycinternal__context)
            )

    def condeqhole(self, cond: Expr, exprs: list[Expr]):
        """Creates a Conditional Eq (synthesis) hole.

        Args:
            cond (Expr): the condition for the Eq.
            exprs (list[Expr]): the list of expressions to consider as candidates for filling this hole.
        """
        if (
            self._pycinternal__context == Context.INPUT
            or self._pycinternal__context == Context.OUTPUT
        ):
            logger.error("Holes in input/output contexts currently not supported")
            sys.exit(1)
        for expr in exprs:
            self._pycinternal__perholes.append(
                PERHole(CondEq(cond, expr), self._pycinternal__context)
            )

    def ctralignhole(self, ctr: Logic, sigs: list[Logic]):
        """Creates a Control Alignment hole.

        Args:
            ctr (Logic): the control signal that the branch conditions are based on.
            sigs (list[Logic]): the signals to learn lookup tables for.
        """
        if (
            self._pycinternal__context == Context.INPUT
            or self._pycinternal__context == Context.OUTPUT
        ):
            logger.error("Holes in input/output contexts currently not supported")
            sys.exit(1)
        self._pycinternal__caholes.append(CtrAlignHole(ctr, sigs))

    def fsmhole(self, guardsigs: list[Logic], statesig: Logic):
        """Creates an FSM hole.

        Args:
            guardsigs (list[Logic]): the guard signals for the FSM.
            statesig (Logic): the state signal for the FSM.
        """
        if (
            self._pycinternal__context == Context.INPUT
            or self._pycinternal__context == Context.OUTPUT
        ):
            logger.error("Holes in input/output contexts currently not supported")
            sys.exit(1)
        self._pycinternal__fsmholes.append(FSMHole(guardsigs, statesig))

    def inv(self, expr: Expr) -> None:
        """Add single trace invariants to the current context.

        Args:
            expr (Expr): the invariant expression.
        """
        if self._pycinternal__context == Context.INPUT:
            self._pycinternal__input_invs.append(Inv(expr))
        elif self._pycinternal__context == Context.STATE:
            self._pycinternal__state_invs.append(Inv(expr))
        else:
            self._pycinternal__output_invs.append(Inv(expr))

    def _inv(self, expr: Expr, ctx: Context) -> None:
        """Add single trace invariants with the specified context.

        Args:
            expr (Expr): the invariant expression.
            ctx (Context): the context to add this invariant under.
        """
        if ctx == Context.INPUT:
            self._pycinternal__input_invs.append(Inv(expr))
        elif ctx == Context.STATE:
            self._pycinternal__state_invs.append(Inv(expr))
        else:
            self._pycinternal__output_invs.append(Inv(expr))

    def pycassert(self, expr: Expr) -> None:
        """Add an assertion to the current context.

        Args:
            expr (Expr): the assertion expression.
        """
        if self._pycinternal__context == Context.UNROLL:
            self._pycinternal__simstep._assert(expr)
        else:
            logger.warning(
                "pycassert can only be used in the unroll context, skipping."
            )

    def pycassume(self, expr: Expr) -> None:
        """Add an assumption to the current context.

        Args:
            expr (Expr): the assumption expression.
        """
        if self._pycinternal__context == Context.UNROLL:
            self._pycinternal__simstep._assume(expr)
        else:
            logger.warning(
                "pycassume can only be used in the unroll context, skipping."
            )

    def instantiate(self, path: Path = Path([])) -> "SpecModule":
        """Instantiate the current SpecModule.

        Args:
            path (Path, optional): The path in the hierarchy to place this module at. Defaults to Path([]).

        Returns:
            SpecModule: return the instantiated module.
        """
        if self._pycinternal__instantiated:
            logger.warning("SpecModule already instantiated, skipping.")
            return
        self.path = path
        # Add all signals (Logic, LogicArray, Structs), groups, functions and submodules.
        sigattrs = {}
        groupattrs = {}
        funcattrs = {}
        submoduleattrs = {}
        auxmoduleattrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            # if isinstance(obj, AuxReg):
            #     if len(path.path) != 0:
            #         logger.error(
            #             "AuxReg can only be used at the top level of the elaboration."
            #         )
            #         sys.exit(1)
            #     if obj.name == "":
            #         obj.name = attr
            #     auxregpath = path.add_level("auxreg").add_level(obj.name)
            #     auxregs[obj.name] = obj.instantiate(auxregpath)
            if (
                isinstance(obj, Logic)
                or isinstance(obj, LogicArray)
                or isinstance(obj, Struct)
            ):
                # Allow different dict key and signal names
                if obj.name == "":
                    obj.name = attr
                sigattrs[obj.name] = obj.instantiate(path.add_level(obj.name))
            elif isinstance(obj, Group):
                if obj.name == "":
                    obj.name = attr
                groupattrs[obj.name] = obj.instantiate(path.add_level(obj.name))
            elif isinstance(obj, SVFunc):
                if obj.name == "":
                    obj.name = attr
                funcattrs[obj.name] = obj
            elif isinstance(obj, AuxModule):
                # Current module must be top level
                if obj.name == "":
                    obj.name = attr
                auxmoduleattrs[obj.name] = obj.instantiate(path.add_level(obj.name))
            elif isinstance(obj, SpecModule):
                if obj.name == "":
                    obj.name = attr
                submoduleattrs[obj.name] = obj.instantiate(path.add_level(obj.name))
        self._pycinternal__signals = sigattrs
        self._pycinternal__groups = groupattrs
        self._pycinternal__functions = funcattrs
        self._pycinternal__submodules = submoduleattrs
        self._pycinternal__auxmodules = auxmoduleattrs
        # Call the specification generator methods.
        self._pycinternal__context = Context.INPUT
        self.input()
        self._pycinternal__context = Context.STATE
        self.state()
        if hasattr(self.state, "_pycinternal__kind_depth"):
            self._pycinternal__kind_depth = getattr(
                self.state, "_pycinternal__kind_depth"
            )
        else:
            self._pycinternal__kind_depth: int = 1
        self._pycinternal__context = Context.OUTPUT
        self.output()
        # Run through simulation steps
        self._pycinternal__context = Context.UNROLL
        # Handle all unrolling schedules
        #! removed: self.simstep()
        # TODO: maybe this needs to be self.__class__
        for i, fn in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(fn, "_pycinternal__is_unroll_schedule"):
                if hasattr(fn, "_pycinternal__unroll_depth"):
                    logger.debug(
                        "Unrolling schedule %s with depth %d",
                        fn.__name__,
                        fn._pycinternal__unroll_depth,
                    )
                    # Unroll this schedule
                    fn_name = fn.__name__
                    fn_unroll_depth = fn._pycinternal__unroll_depth
                    # Create a new simulation schedule
                    simschedule = SimulationSchedule(fn_name, fn_unroll_depth)
                    for j in range(fn_unroll_depth):
                        self._pycinternal__simstep = SimulationStep()
                        fn(j)
                        simschedule.step(self._pycinternal__simstep)
                    self._pycinternal__simsteps[fn_name] = simschedule
                else:
                    logger.error(
                        "Unroll schedule must have a depth specified, skipping"
                    )
                    continue

        self._pycinternal__instantiated = True
        return self

    def get_unroll_kind_depths(self):
        if not self._pycinternal__instantiated:
            logger.error(
                "Module not instantiated, call instantiate() before max_counter_width()"
            )
            sys.exit(1)
        # return maximum counter length between kind and unroll schedules
        kd = self._pycinternal__kind_depth
        return (kd, max([kd] + [v.depth for v in self._pycinternal__simsteps.values()]))

    def get_hier_path(self, sep: str = "."):
        """Call inner get_hier_path method."""
        return self.path.get_hier_path(sep)

    def sprint(self):
        s = ""
        s += f"module {self.__class__.__name__}\n"
        s += "signals:\n"
        for k, v in self._pycinternal__signals.items():
            s += f"\t{k} : {v._typ()}\n"
        s += "submodules:\n"
        for k, v in self._pycinternal__submodules.items():
            s += f"\t{k} : {v.__class__.__name__}\n"
        s += "input:\n"
        for i in self._pycinternal__input_tt:
            s += f"\t{i}\n"
        s += "state:\n"
        for i in self._pycinternal__state_tt:
            s += f"\t{i}\n"
        s += "output:\n"
        for i in self._pycinternal__output_tt:
            s += f"\t{i}\n"
        if self._pycinternal__perholes:
            s += "perholes:\n"
            for i in self._pycinternal__perholes:
                s += f"\t{i.per} ({i.ctx})\n"
        return s

    def get_repr(self, reprs):
        # Find all submodules and structs that need to be defined
        reprs[self.__class__.__name__] = repr(self)
        for s, t in self._pycinternal__signals.items():
            if isinstance(t, Struct):
                if t.__class__.__name__ not in reprs:
                    reprs = t.get_repr(reprs)
        for s, t in self._pycinternal__groups.items():
            if t.__class__.__name__ not in reprs:
                reprs = t.get_repr(reprs)

        for s, t in self._pycinternal__submodules.items():
            if t.__class__.__name__ not in reprs:
                reprs = t.get_repr(reprs)
        return reprs

    def full_repr(self):
        reprs = self.get_repr({})
        s = "from pycaliper.per import *\n\n"
        for k, v in reprs.items():
            s += f"{v}\n"
        return s

    def __repr__(self):
        """Create the repr string."""
        inits = [
            "\tdef __init__(self, name = '', **kwargs):",
            f"\t\tsuper().__init__(name, kwargs)",
        ]
        for s, t in self._pycinternal__signals.items():
            if isinstance(t, Logic):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = Logic({t.width}, "{t.name}")'
                )
            elif isinstance(t, LogicArray):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = LogicArray(lambda: Logic({t.typ().width}), {t.size}, "{t.name}")'
                )
            elif isinstance(t, Struct):
                inits.append(
                    f'\t\tself.{nonreserved_or_fresh(t.name)} = {t.__class__.__name__}("{t.name}")'
                )
            else:
                logger.error(f"Invalid signal type: {t}")
                sys.exit(1)
        for s, t in self._pycinternal__groups.items():
            inits.append(
                f'\t\tself.{nonreserved_or_fresh(t.name)} = {t.__class__.__name__}("{t.name}")'
            )
        initstring = "\n".join(inits)
        for s, t in self._pycinternal__submodules.items():
            inits.append(
                f'\t\tself.{nonreserved_or_fresh(t.name)} = {t.__class__.__name__}("{t.name}", {t.params})'
            )
        initstring = "\n".join(inits)

        inputs = (
            ["\tdef input(self):"]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__input_tt]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__input_invs]
            + ["\t\tpass"]
        )
        inputstring = "\n".join(inputs)

        outputs = (
            ["\tdef output(self):"]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__output_tt]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__output_invs]
            + ["\t\tpass"]
        )
        outputstring = "\n".join(outputs)

        states = (
            [f"\t@kind({self._pycinternal__kind_depth})\n\tdef state(self):"]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__state_tt]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__state_invs]
            + [f"\t\t{repr(t)}" for t in self._pycinternal__perholes if t.active]
            + ["\t\tpass"]
        )
        statestring = "\n".join(states)

        simsteps = "\n\n".join(
            [repr(v) for _, v in self._pycinternal__simsteps.items()]
        )

        return f"""

class {self.__class__.__name__}(SpecModule):

{initstring}

{inputstring}

{outputstring}

{statestring}

{simsteps}
        """

    def pprint(self):
        print(self.sprint())


class AuxPort(Logic):
    def __init__(self, width: int = 1, name: str = "", root: str = None) -> None:
        super().__init__(width, name, root)


class AuxModule(SpecModule):
    def __init__(self, portmapping: dict[str, TypedElem], name="", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._pycinternal__ports = {}
        self.portmapping = portmapping

    def instantiate(self, root: Path = Path([])) -> None:
        self.path = Path([])
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, AuxPort):
                self._pycinternal__ports[obj.name] = obj.instantiate(
                    self.path.add_level(obj.name)
                )
                obj.root = root.get_hier_path()
            elif isinstance(obj, TypedElem):
                obj.root = root.get_hier_path()
                obj.instantiate(self.path.add_level(obj.name))
        # Check that all ports are mapped
        for k in self.portmapping:
            if k not in self._pycinternal__ports:
                logger.error(f"Port {k} not found in {self.__class__.__name__}")
                sys.exit(1)
        return self

    def get_instance_str(self, bindm: str):
        portbindings = []
        for k, v in self.portmapping.items():
            portbindings.append(f".{k}({bindm}.{v})")
        portbindings = ",\n\t".join(portbindings)

        aux_mod_decl = f"""
// SpecModule {self.get_hier_path()}
{self.__class__.__name__} {self.name} (
        // Bindings
        {portbindings}
);
"""
        return aux_mod_decl


# class AuxReg(Logic):

#     def __init__(self, width = 1, name: str = "", root: str = None) -> None:
#         super().__init__(width, name, root)
#         self.elaborated = False

#     def instantiate(self, path: Path) -> "AuxReg":
#         logger.warn("AuxReg is only supported for JG and will be deprecated, use AuxModule instead!")
#         self.path = path
#         self.elaborated = True
#         jgoracle.create_auxreg(self.name, self.width)

# SVA-specific functions
past = SVFunc("$past")
stable = SVFunc("$stable")
fell = SVFunc("$fell")
rose = SVFunc("$rose")
