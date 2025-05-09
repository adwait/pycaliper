"""
PyCaliper

Author: Adwait Godbole, UC Berkeley

File: per/expr.py

Expression base class, and Expr operator classes

This module provides the base class for expression abstract syntax trees (AST) with overloaded operators.
Supports operations like addition, subtraction, logical operations, etc.

For licensing information, please refer to the LICENSE file.
"""

import logging

from enum import Enum

logger = logging.getLogger(__name__)


class Expr:
    """
    Base class for expression abstract syntax trees (AST) with overloaded operators.
    Supports operations like addition, subtraction, logical operations, etc.
    """

    # PyCaliper Expression AST base class with overloaded operators.
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        pass

    def __call__(self, hi, low):
        return OpApply(Extract(), [self, hi, low])

    def __and__(self, other):
        return OpApply(LogicalAnd(), [self, other])

    def __or__(self, other):
        return OpApply(LogicalOr(), [self, other])

    def __xor__(self, other):
        return OpApply(BinaryXor(), [self, other])

    def __invert__(self):
        return OpApply(UnaryLogicalNot(), [self])

    def __lshift__(self, other):
        return OpApply(LogicalShiftLeft(), [self, other])

    def __rshift__(self, other):
        return OpApply(LogicalShiftRight(), [self, other])

    def __add__(self, other):
        return OpApply(Add(), [self, other])

    def __sub__(self, other):
        return OpApply(Sub(), [self, other])

    def __lt__(self, other):
        return OpApply(LessThan(), [self, other])

    def __le__(self, other):
        return OpApply(LessThanEqual(), [self, other])

    def __gt__(self, other):
        return OpApply(GreaterThan(), [self, other])

    def __ge__(self, other):
        return OpApply(GreaterThanEqual(), [self, other])

    def __eq__(self, other):
        return OpApply(Equality(), [self, other])

    def __ne__(self, other):
        return OpApply(Inequality(), [self, other])

    def eand(self, other):
        return OpApply(BinaryAnd(), [self, other])

    def eor(self, other):
        return OpApply(BinaryOr(), [self, other])

    def eneg(self):
        return OpApply(UnaryBitwiseNot(), [self])

    def get_sva(self, pref: str = "a") -> str:
        # Convert into SystemVerilog assertion
        raise NotImplementedError(
            f"get_sva not implemented for class {self.__class__.__name__}."
        )


class Op:
    """
    Represents an operator with a string representation, fixity, and a representation string.

    Attributes:
        opstring (str): The operator string.
        fixity (Fixity): The fixity of the operator.
        reprstring (str): The string for creating a representation.
    """

    class Fixity(Enum):
        """Fixity of the operator; used during SVA compilation."""

        INFIX = 0
        PREFIX = 1
        EXTRACT = 2
        CONCAT = 3
        ITE = 4

    def __init__(self, opstring: str, fixity: Fixity, reprstring: str) -> None:
        """
        Args:
            name (str): Operator string
            fixity (Fixity): fixity
            reprstring (str): string for creating repr
        """
        self.opstring = opstring
        self.fixity = fixity
        self.reprstring = reprstring

    def __str__(self) -> str:
        return self.opstring

    def __repr__(self):
        return f"{self.reprstring}()"


# Unary operators
class UnaryPlus(Op):
    """Represents the unary plus operation."""

    def __init__(self) -> None:
        super().__init__("+", Op.Fixity.PREFIX, "UnaryPlus")


class UnaryMinus(Op):
    """Represents the unary minus operation."""

    def __init__(self) -> None:
        super().__init__("-", Op.Fixity.PREFIX, "UnaryMinus")


class UnaryBitwiseAnd(Op):
    """Represents the unary bitwise AND operation."""

    def __init__(self) -> None:
        super().__init__("&", Op.Fixity.PREFIX, "UnaryBitwiseAnd")


class UnaryBitwiseNand(Op):
    """Represents the unary bitwise NAND operation."""

    def __init__(self) -> None:
        super().__init__("~&", Op.Fixity.PREFIX, "UnaryBitwiseNand")


class UnaryBitwiseOr(Op):
    """Represents the unary bitwise OR operation."""

    def __init__(self) -> None:
        super().__init__("|", Op.Fixity.PREFIX, "UnaryBitwiseOr")


class UnaryBitwiseNor(Op):
    """Represents the unary bitwise NOR operation."""

    def __init__(self) -> None:
        super().__init__("~|", Op.Fixity.PREFIX, "UnaryBitwiseNor")


class UnaryBitwiseXor(Op):
    """Represents the unary bitwise XOR operation."""

    def __init__(self) -> None:
        super().__init__("^", Op.Fixity.PREFIX, "UnaryBitwiseXor")


class UnaryBitwiseXnor(Op):
    """Represents the unary bitwise XNOR operation."""

    def __init__(self) -> None:
        super().__init__("~^", Op.Fixity.PREFIX, "UnaryBitwiseXnor")


class UnaryLogicalNot(Op):
    """Represents the unary logical NOT operation."""

    def __init__(self) -> None:
        super().__init__("!", Op.Fixity.PREFIX, "UnaryLogicalNot")


class UnaryBitwiseNot(Op):
    """Represents the unary bitwise NOT operation."""

    def __init__(self) -> None:
        super().__init__("~", Op.Fixity.PREFIX, "UnaryBitwiseNot")


# Binary operators
class Power(Op):
    """Represents the power operation."""

    def __init__(self) -> None:
        super().__init__("**", Op.Fixity.INFIX, "Power")


class Mul(Op):
    """Represents the multiplication operation."""

    def __init__(self) -> None:
        super().__init__("*", Op.Fixity.INFIX, "Mul")


class Div(Op):
    """Represents the division operation."""

    def __init__(self) -> None:
        super().__init__("/", Op.Fixity.INFIX, "Div")


class Mod(Op):
    """Represents the modulus operation."""

    def __init__(self) -> None:
        super().__init__("%", Op.Fixity.INFIX, "Mod")


class Add(Op):
    """Represents the addition operation."""

    def __init__(self) -> None:
        super().__init__("+", Op.Fixity.INFIX, "Add")


class Sub(Op):
    """Represents the subtraction operation."""

    def __init__(self) -> None:
        super().__init__("-", Op.Fixity.INFIX, "Sub")


class LogicalShiftRight(Op):
    """Represents the logical shift right operation."""

    def __init__(self) -> None:
        super().__init__(">>", Op.Fixity.INFIX, "LogicalShiftRight")


class LogicalShiftLeft(Op):
    """Represents the logical shift left operation."""

    def __init__(self) -> None:
        super().__init__("<<", Op.Fixity.INFIX, "LogicalShiftLeft")


class ArithmeticShiftRight(Op):
    """Represents the arithmetic shift right operation."""

    def __init__(self) -> None:
        super().__init__(">>>", Op.Fixity.INFIX, "ArithmeticShiftRight")


class ArithmeticShiftLeft(Op):
    """Represents the arithmetic shift left operation."""

    def __init__(self) -> None:
        super().__init__("<<<", Op.Fixity.INFIX, "ArithmeticShiftLeft")


class LessThan(Op):
    """Represents the less than operation."""

    def __init__(self) -> None:
        super().__init__("<", Op.Fixity.INFIX, "LessThan")


class LessThanEqual(Op):
    """Represents the less than or equal operation."""

    def __init__(self) -> None:
        super().__init__("<=", Op.Fixity.INFIX, "LessThanEqual")


class GreaterThan(Op):
    """Represents the greater than operation."""

    def __init__(self) -> None:
        super().__init__(">", Op.Fixity.INFIX, "GreaterThan")


class GreaterThanEqual(Op):
    """Represents the greater than or equal operation."""

    def __init__(self) -> None:
        super().__init__(">=", Op.Fixity.INFIX, "GreaterThanEqual")


class Equality(Op):
    """Represents the equality operation."""

    def __init__(self) -> None:
        super().__init__("==", Op.Fixity.INFIX, "Equality")


class Inequality(Op):
    """Represents the inequality operation."""

    def __init__(self) -> None:
        super().__init__("!=", Op.Fixity.INFIX, "Inequality")


class CaseEquality(Op):
    """Represents the case equality operation."""

    def __init__(self) -> None:
        super().__init__("===", Op.Fixity.INFIX, "CaseEquality")


class CaseInequality(Op):
    """Represents the case inequality operation."""

    def __init__(self) -> None:
        super().__init__("!==", Op.Fixity.INFIX, "CaseInequality")


class WildcardEquality(Op):
    """Represents the wildcard equality operation."""

    def __init__(self) -> None:
        super().__init__("==?", Op.Fixity.INFIX, "WildcardEquality")


class WildcardInequality(Op):
    """Represents the wildcard inequality operation."""

    def __init__(self) -> None:
        super().__init__("!=?", Op.Fixity.INFIX, "WildcardInequality")


class BinaryAnd(Op):
    """Represents the binary AND operation."""

    def __init__(self) -> None:
        super().__init__("&", Op.Fixity.INFIX, "BinaryAnd")


class BinaryXor(Op):
    """Represents the binary XOR operation."""

    def __init__(self) -> None:
        super().__init__("^", Op.Fixity.INFIX, "BinaryXor")


class BinaryXnor(Op):
    """Represents the binary XNOR operation."""

    def __init__(self) -> None:
        super().__init__("~^", Op.Fixity.INFIX, "BinaryXnor")


class BinaryNor(Op):
    """Represents the binary NOR operation."""

    def __init__(self) -> None:
        super().__init__("~^", Op.Fixity.INFIX, "BinaryNor")


class BinaryOr(Op):
    """Represents the binary OR operation."""

    def __init__(self) -> None:
        super().__init__("|", Op.Fixity.INFIX, "BinaryOr")


class LogicalAnd(Op):
    """Represents the logical AND operation."""

    def __init__(self) -> None:
        super().__init__("&&", Op.Fixity.INFIX, "LogicalAnd")


class LogicalOr(Op):
    """Represents the logical OR operation."""

    def __init__(self) -> None:
        super().__init__("||", Op.Fixity.INFIX, "LogicalOr")


class LogicalImplication(Op):
    """Represents the logical implication operation."""

    def __init__(self) -> None:
        super().__init__("->", Op.Fixity.INFIX, "LogicalImplication")


class LogicalEquivalence(Op):
    """Represents the logical equivalence operation."""

    def __init__(self) -> None:
        super().__init__("<->", Op.Fixity.INFIX, "LogicalEquivalence")


# Special operators
class ITE(Op):
    """Represents the if-then-else operation."""

    def __init__(self) -> None:
        super().__init__("ite", Op.Fixity.ITE, "ITE")


class Extract(Op):
    """Represents the extract operation."""

    def __init__(self) -> None:
        super().__init__("extract", Op.Fixity.EXTRACT, "Extract")


class Concat(Op):
    """Represents the concatenation operation."""

    def __init__(self) -> None:
        super().__init__("++", Op.Fixity.CONCAT, "Concat")


class OpApply(Expr):
    """Apply an operator to a list of arguments.

    Args:
        op (Op): Operator
        args (list): List of arguments.
    """

    """Apply an operator to a list of arguments

    Args:
        op (Op): Operator
        args (list): list of arguments.
    """

    def __init__(self, op: Op, args: list) -> None:
        self.op = op
        self.args = args

    def __str__(self) -> str:
        # Convert into string (not necessarily SVA-compatible)
        if self.op.fixity == Op.Fixity.EXTRACT:
            return f"{self.args[0]}[{self.args[1]}:{self.args[2]}]"
        elif self.op.fixity == Op.Fixity.CONCAT:
            return f"({self.args[0]} {self.op}  {self.args[1]})"
        elif self.op.fixity == Op.Fixity.INFIX:
            return f"({self.args[0]} {self.op} {self.args[1]})"
        elif self.op.fixity == Op.Fixity.PREFIX:
            return f"{self.op} {self.args[0]}"
        else:
            raise ValueError(f"Unknown fixity for operator {self.op}.")

    def get_sva(self, pref: str = "a") -> str:
        """Get the SVA representation of the operator application.

        Args:
            pref (str, optional): Prefix that signals are nested under. Defaults to 'a'.

        Raises:
            ValueError: If the fixity of the operator is unknown.

        Returns:
            str: SVA representation of the operator application.
        """
        if self.op.fixity == Op.Fixity.EXTRACT:
            return f"{self.args[0].get_sva(pref)}[{self.args[1]}:{self.args[2]}]"
        elif self.op.fixity == Op.Fixity.CONCAT:
            argseq = ", ".join([arg.get_sva(pref) for arg in self.args])
            return f"{{ {argseq} }}"
        elif self.op.fixity == Op.Fixity.INFIX:
            return (
                f"({self.args[0].get_sva(pref)} {self.op} {self.args[1].get_sva(pref)})"
            )
        elif self.op.fixity == Op.Fixity.PREFIX:
            return f"({self.op} {self.args[0].get_sva(pref)})"
        else:
            raise ValueError(f"Unknown fixity for operator {self.op}.")

    def __repr__(self):
        # Overloaded ops are pretty-printed
        match self.op:
            case LogicalAnd():
                return f"({' & '.join([repr(a) for a in self.args])})"
            case LogicalOr():
                return f"({' | '.join([repr(a) for a in self.args])})"
            case BinaryXor():
                return f"({' ^ '.join([repr(a) for a in self.args])})"
            case UnaryLogicalNot():
                return f"(~{repr(self.args[0])})"
            case LogicalShiftLeft():
                return f"({' << '.join([repr(a) for a in self.args])})"
            case LogicalShiftRight():
                return f"({' >> '.join([repr(a) for a in self.args])})"
            case Add():
                return f"({' + '.join([repr(a) for a in self.args])})"
            case Sub():
                return f"({' - '.join([repr(a) for a in self.args])})"
            case LessThan():
                return f"({' < '.join([repr(a) for a in self.args])})"
            case LessThanEqual():
                return f"({' <= '.join([repr(a) for a in self.args])})"
            case GreaterThan():
                return f"({' > '.join([repr(a) for a in self.args])})"
            case GreaterThanEqual():
                return f"({' >= '.join([repr(a) for a in self.args])})"
            case Equality():
                return f"({' == '.join([repr(a) for a in self.args])})"
            case Inequality():
                return f"({' != '.join([repr(a) for a in self.args])})"
            case _:
                return f"OpApply({repr(self.op)}, {repr(self.args)})"


class Const(Expr):
    """A constant value with an optional width.

    Args:
        val (int): The constant value.
        width (int, optional): The width of the constant. Defaults to -1.
    """

    """A constant value"""

    def __init__(self, val: int, width: int = -1) -> None:
        self.val = val
        self.width = width

    def __str__(self) -> str:
        if self.width == -1 and self.val == 0:
            # Unspecified width (must be zero)
            return "'0"
        elif self.width > 0:
            return f"{self.width}'d{self.val}"
        else:
            logger.warn(f"Invalid Const expression: {self.val}, {self.width}")
            return f"{self.width}'d{self.val}"

    def get_sva(self, pref: str = "a") -> str:
        return f"{self}"

    def __repr__(self):
        return f"Const({self.val}, {self.width})"
