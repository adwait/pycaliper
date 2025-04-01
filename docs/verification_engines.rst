Verification Engines
====================

This section provides an overview of the different verification engines used in the project. The verification engines are located in the `verif` directory and include the following:

JG Verifier
-----------
The JG Verifier is responsible for verifying one-trace and two-trace properties of a given module. It uses the JasperGold tool to set up the verification environment and perform the necessary checks. The JG Verifier supports both induction and BMC (Bounded Model Checking) methods.

Example usage for one-trace verification:

.. code-block:: python

    from pycaliper.verif.jgverifier import JGVerifier1Trace
    from pycaliper.per import SpecModule
    from pycaliper.pycconfig import PYConfig

    specmodule = SpecModule()
    pyconfig = PYConfig()
    verifier = JGVerifier1Trace()
    result = verifier.verify(specmodule, pyconfig)
    print("Verification result:", "SAFE" if result else "UNSAFE")

Btor Verifier
-------------
The BTOR Verifier is a component of PyCaliper's verification framework designed to verify BTOR designs. It utilizes the Boolector solver to perform symbolic execution and check the safety of the design. The BTOR Verifier supports both one-trace and two-trace verification methods, allowing for comprehensive analysis of the design's properties. It generates assumptions and assertions based on the design's specifications and checks for counterexamples to ensure the design's correctness.

Example usage for two-trace verification:

.. code-block:: python

    from pycaliper.verif.btorverifier import BTORVerifier2Trace
    from pycaliper.per import SpecModule
    from pycaliper.pycconfig import DesignConfig

    specmodule = SpecModule()
    design_config = DesignConfig()
    verifier = BTORVerifier2Trace()
    result = verifier.verify(specmodule, design_config)
    print("Verification result:", "SAFE" if result.is_safe else "UNSAFE")
