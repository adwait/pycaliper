.. _proofmanager_guide:

ProofManager Guide
------------------

The ``ProofManager`` class provides an API for managing the verification workflow. It allows creating specifications, designs, and orchestrating various types of proofs.

Key Features
~~~~~~~~~~~~

- Works with both BTOR and Jasper Gold verification engines
- Proof orchestration: Manages complex verification workflows by maintaining a history of all proofs and their results
- Flexible design creation: Supports both file-based and configuration-based design setup

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pycaliper.proofmanager import ProofManager
   from tests.specs.demo import demo

   # Create a ProofManager instance
   pm = ProofManager()

   # Create a specification
   spec = pm.mk_spec(demo, "demo_spec")

   # Create a design from BTOR file
   design = pm.mk_btor_design_from_file("path/to/design.btor", "demo_design")

   # Run a proof
   result = pm.mk_btor_proof_one_trace(spec, design)
   print(f"Proof result: {'PASS' if result.result else 'FAIL'}")

With GUI Support
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create ProofManager with web GUI
   pm = ProofManager(webgui=True)

   # Or with CLI GUI
   pm = ProofManager(cligui=True)

Creating Specifications
-----------------------

The ProofManager provides a unified interface for creating and managing specification modules.

.. code-block:: python

   # Create a specification with parameters
   spec = pm.mk_spec(MySpecClass, "my_spec", width=32, depth=8)

   # The specification is automatically instantiated and stored
   # You can reference it later by name
   same_spec = pm.specs["my_spec"]

**Key Points:**

- Specifications are automatically instantiated when created
- Each specification must have a unique name
- Additional parameters can be passed as keyword arguments
- Created specifications are stored in the ``pm.specs`` dictionary

Creating Designs
----------------

BTOR Designs
~~~~~~~~~~~~

For designs represented as BTOR2 files:

.. code-block:: python

   # Create from BTOR file
   design = pm.mk_btor_design_from_file("design.btor", "my_design")

Jasper Gold Designs
~~~~~~~~~~~~~~~~~~~

For designs that use Jasper Gold verification:

.. code-block:: python

   # Create from configuration files
   design = pm.mk_jg_design_from_pyc(
       name="jg_design",
       jasper_config_path="jasper_config.json",
       design_config_path="design_config.json"
   )

**Configuration Files:**

- **Jasper config**: Contains Jasper Gold tool settings, port configuration, and file paths
- **Design config**: Contains design-specific settings like signal mappings and constraints

Running Proofs
--------------

The ProofManager supports several types of verification proofs:

One-Trace Proofs
~~~~~~~~~~~~~~~~

Verify safety properties on a single execution trace:

.. code-block:: python

   # BTOR backend
   result = pm.mk_btor_proof_one_trace(spec, design)

   # Jasper Gold backend
   result = pm.mk_jg_proof_one_trace(spec, design)

Two-Trace Proofs
~~~~~~~~~~~~~~~~

Verify equivalence properties between two execution traces:

.. code-block:: python

   # Only available with Jasper Gold backend
   result = pm.mk_jg_proof_two_trace(spec, design)

Bounded Proofs
~~~~~~~~~~~~~~

Verify properties up to a specific depth with custom schedules:

.. code-block:: python

   def my_schedule(step):
       # Define custom scheduling logic
       pass

   result = pm.mk_jg_proof_bounded_spec(spec, design, my_schedule)

Refinement Checking
-------------------

Module-to-Module Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check if one specification refines another:

.. code-block:: python

   from pycaliper.verif.refinementverifier import RefinementMap

   # Define refinement mapping
   rmap = RefinementMap(...)

   result = pm.check_mm_refinement(spec1, spec2, rmap)

Schedule-to-Schedule Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check refinement between different execution schedules:

.. code-block:: python

   result = pm.check_ss_refinement(spec, schedule1, schedule2, flip=False)

Proof Results
-------------

All proof methods return ``ProofResult`` objects with detailed information:

.. code-block:: python

   result = pm.mk_btor_proof_one_trace(spec, design)

   print(f"Result: {result.result}")  # True/False
   print(f"Proof type: {type(result).__name__}")
   print(f"Details: {result}")

**Result Types:**

- ``OneTraceIndPR``: One-trace inductive proof results
- ``TwoTraceIndPR``: Two-trace inductive proof results  
- ``OneTraceBndPR``: Bounded proof results
- ``MMRefinementPR``: Module-to-module refinement results
- ``SSRefinementPR``: Schedule-to-schedule refinement results

Working with Names vs Objects
-----------------------------

The ProofManager accepts both object references and string names for specifications and designs:

.. code-block:: python

   # Using objects directly
   result1 = pm.mk_btor_proof_one_trace(spec_obj, design_obj)

   # Using names (must be previously created)
   result2 = pm.mk_btor_proof_one_trace("spec_name", "design_name")

   # Mixed usage
   result3 = pm.mk_btor_proof_one_trace(spec_obj, "design_name")

GUI Integration
---------------

The ProofManager can optionally integrate with graphical user interfaces for interactive verification workflows.

Web GUI
~~~~~~~

.. code-block:: python

   pm = ProofManager(webgui=True)
   # Web interface will be available at http://localhost:5000

CLI GUI
~~~~~~~

.. code-block:: python

   pm = ProofManager(cligui=True)
   # Rich terminal interface will be displayed

**GUI Features:**

- Real-time proof progress tracking
- Interactive specification and design management
- Visual proof result display
- Progress bars and status updates

Advanced Usage
--------------

Custom Design Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pycaliper.pycconfig import DesignConfig

   # Custom design configuration
   dc = DesignConfig(cpy1="a", cpy2="b", custom_param="value")
   result = pm.mk_btor_proof_one_trace(spec, design, dc)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple specifications
   specs = ["spec1", "spec2", "spec3"]
   designs = ["design1", "design2", "design3"]

   results = []
   for spec, design in zip(specs, designs):
       result = pm.mk_jg_proof_one_trace(spec, design)
       results.append(result)

   # Analyze results
   passed = sum(1 for r in results if r.result)
   print(f"Passed: {passed}/{len(results)}")

Error Handling
--------------

The ProofManager includes comprehensive error handling:

.. code-block:: python

   try:
       result = pm.mk_jg_proof_one_trace("nonexistent_spec", "my_design")
   except ValueError as e:
       print(f"Error: {e}")  # "Spec nonexistent_spec not found."

   try:
       # Wrong design type for Jasper verification
       result = pm.mk_jg_proof_one_trace(spec, btor_design)
   except AssertionError as e:
       print(f"Error: {e}")  # "Design must be a JGDesign for Jasper verification."

Best Practices
--------------

1. **Use descriptive names**: Choose clear, descriptive names for specifications and designs
2. **Check results**: Always check proof results and handle failures appropriately
3. **Manage resources**: Close GUI connections when done with long-running sessions
4. **Organize workflows**: Group related specifications and designs logically
5. **Document configurations**: Keep configuration files well-documented and version-controlled

Example: Complete Verification Workflow
---------------------------------------

Here's a complete example demonstrating a typical verification workflow:

.. code-block:: python

   from pycaliper.proofmanager import ProofManager
   from pycaliper.pycconfig import DesignConfig
   from tests.specs.demo import demo
   from tests.specs.counter import counter

   # Initialize ProofManager with web GUI
   pm = ProofManager(webgui=True)

   # Create specifications
   demo_spec = pm.mk_spec(demo, "demo_spec")
   counter_spec = pm.mk_spec(counter, "counter_spec", width=8)

   # Create designs
   demo_design = pm.mk_btor_design_from_file(
       "examples/designs/demo/btor/full_design.btor", 
       "demo_design"
   )
   
   counter_design = pm.mk_jg_design_from_pyc(
       "counter_design",
       "examples/designs/counter/config.json",
       "examples/designs/counter/design.json"
   )

   # Run various proofs
   results = []

   # BTOR one-trace proof
   result1 = pm.mk_btor_proof_one_trace("demo_spec", "demo_design")
   results.append(("Demo BTOR 1-trace", result1))

   # Jasper one-trace proof  
   result2 = pm.mk_jg_proof_one_trace("counter_spec", "counter_design")
   results.append(("Counter JG 1-trace", result2))

   # Jasper two-trace proof
   result3 = pm.mk_jg_proof_two_trace("counter_spec", "counter_design")
   results.append(("Counter JG 2-trace", result3))

   # Print summary
   print("Verification Results:")
   print("=" * 50)
   for name, result in results:
       status = "PASS" if result.result else "FAIL"
       print(f"{name:25} | {status}")

   # Check overall success
   all_passed = all(r.result for _, r in results)
   print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

This example demonstrates the key capabilities of the ProofManager and shows how to build comprehensive verification workflows.
