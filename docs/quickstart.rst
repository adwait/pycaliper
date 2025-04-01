.. _quickstart:

Quickstart Guide
================

This guide provides a quick introduction to using PyCaliper with a simple example. We will use the ``demo.sv`` file from ``examples/designs/demo`` and the corresponding ``demo.py`` specification from ``tests/specs/demo.py``. We will perform a proof using the ``BTORVerifier1Trace``, PyCaliper's in-built BTOR verification backend.

Prerequisites
-------------
- Ensure you have PyCaliper installed and set up correctly.
- Familiarize yourself with the basic concepts of PyCaliper, such as ``SpecModule`` and BTOR design.

Example Design and Specification
--------------------------------
The ``demo.sv`` file represents a simple design with inputs, outputs, and state logic. The corresponding ``demo.py`` file defines a ``SpecModule`` class named ``demo`` with inputs, outputs, and state logic.

Performing a Proof
------------------
To perform a proof using the ``BTORVerifier1Trace``, follow these steps:

1. Create a BTOR design using the ``mk_btordesign`` function:

   .. code-block:: python

      from pycaliper.proofmanager import mk_btordesign
      prgm = mk_btordesign("demo", "examples/designs/demo/btor/full_design.btor")

2. Instantiate the ``demo`` specification:

   .. code-block:: python

      from tests.specs.demo import demo
      spec = demo()
      spec.instantiate()

3. Perform the proof using ``BTORVerifier1Trace``:

   .. code-block:: python

      from pycaliper.verif.btorverifier import BTORVerifier1Trace
      from pycaliper.pycconfig import DesignConfig

      verifier = BTORVerifier1Trace()
      result = verifier.verify(spec, prgm, DesignConfig(cpy1="a"))
      print("Proof result:", "SAFE" if result.verified else "BUG")

   This will verify the design against the specification and print the result.

Complete Example
----------------

Here is a complete Python script that demonstrates the process:

.. code-block:: python

   from pycaliper.proofmanager import mk_btordesign
   from pycaliper.verif.btorverifier import BTORVerifier1Trace
   from pycaliper.pycconfig import DesignConfig
   from tests.specs.demo import demo

   # Create a BTOR design
   prgm = mk_btordesign("demo", "examples/designs/demo/btor/full_design.btor")

   # Instantiate the demo specification
   spec = demo()
   spec.instantiate()

   # Perform the proof
   verifier = BTORVerifier1Trace()
   result = verifier.verify(spec, prgm, DesignConfig(cpy1="a"))
   print("Proof result:", "SAFE" if result.verified else "BUG")
