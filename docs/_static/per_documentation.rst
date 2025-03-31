 PyCaliper PER Module Documentation
==================================

Overview
--------

The `pycaliper/per` directory contains modules that define internal representations for PyCaliper, focusing on expressions, partial equivalence relations, logic signals, and SystemVerilog structures.

Modules
-------

expr.py
~~~~~~~

- **Author**: Adwait Godbole, UC Berkeley
- **Purpose**: Provides a base class for expressions (`Expr`) and various operator classes for constructing expression trees.

Key Classes
^^^^^^^^^^^

1. **Expr**:
   - Base class for expression abstract syntax trees (AST) with overloaded operators.
   - Supports operations like addition, subtraction, logical operations, etc.

2. **Op**:
   - Represents an operator with a string representation, fixity, and a representation string.

3. **Unary Operators**:
   - Includes classes like `UnaryPlus`, `UnaryMinus`, `UnaryBitwiseAnd`, etc.

4. **Binary Operators**:
   - Includes classes like `Power`, `Mul`, `Div`, `Add`, `Sub`, etc.

5. **Special Operators**:
   - Includes classes like `ITE`, `Extract`, and `Concat`.

6. **OpApply**:
   - Represents the application of an operator to a list of arguments.

7. **Const**:
   - Represents a constant value with an optional width.

per.py
~~~~~~

- **Author**: Adwait Godbole, UC Berkeley
- **Purpose**: Defines classes for partial equivalence relations, logic signals, SystemVerilog structs, and specification modules.

Key Classes
^^^^^^^^^^^

1. **Path**:
   - Represents a hierarchical path in the specification.

2. **TypedElem**:
   - Base class for elements in the design hierarchy with a type.

3. **Logic**:
   - Represents single bitvectors/signals.

4. **LogicArray**:
   - Represents an array of logic signals.

5. **PER (Partial Equivalence Relation)**:
   - Base class for partial equivalence relations.
   - Includes subclasses like `Eq` and `CondEq`.

6. **Struct**:
   - Represents SystemVerilog structs.

7. **SpecModule**:
   - Represents a specification module related to a SystemVerilog hardware module.

8. **AuxModule** and **AuxReg**:
   - Represents auxiliary modules and registers.

9. **SimulationStep** and **SimulationSchedule**:
   - Manages simulation steps and schedules.

10. **Decorators**:
    - `kinduct` and `unroll` decorators for specifications.

11. **SVFunc** and **SVFuncApply**:
    - Represents SystemVerilog functions and their applications.

12. **Holes**:
    - Represents synthesis holes for partial equivalence relations and conditional equality.

Functions
^^^^^^^^^

- **nonreserved_or_fresh**: Ensures a name is not a reserved keyword.
- **get_path_from_hierarchical_str**: Converts a hierarchical string into a `Path` object.
- **when**: Creates a conditional equality PER generator.
