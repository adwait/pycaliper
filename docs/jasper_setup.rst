.. _jasper_setup:

Jasper Configuration Guide
==========================

This document provides instructions specific to setting up the Jasper interface in PyCaliper.

JasperConfig Object
-------------------

The ``JasperConfig`` class in ``pycaliper/pycconfig.py`` is used to configure the Jasper Gold formal verification tool. Below are the attributes and their descriptions:

- **jdir** (*str*): Jasper working directory relative to the pycaliper directory.
- **script** (*str*): TCL script relative to the Jasper working directory.
- **pycfile** (*str*): Location of the generated SVA file relative to the Jasper working directory.
- **context** (*str*): Proof node context.
- **design_list** (*str*): Design list file.
- **port** (*int*): Port number to connect to Jasper server.

Example Usage
-------------

To use the ``JasperConfig``, you can create an instance and set the desired attributes:

.. code-block:: python

    from pycaliper.pycconfig import JasperConfig

    jasper_config = JasperConfig(
        jdir="path/to/jasper/dir",
        script="setup.tcl",
        pycfile="output.sva",
        context="default",
        design_list="design.lst",
        port=8080
    )

This configuration will be used to interface with the Jasper Gold tool, ensuring that the correct directories and files are utilized during the verification process.

Setting up the Jasper Interface
-------------------------------

1. **Configure JasperConfig**: Ensure that the ``JasperConfig`` object is properly configured with the correct paths and settings as per your project requirements.

2. **Jasper Working Directory**: Set the ``jdir`` attribute to point to the directory where Jasper Gold will operate.

3. **TCL Script**: Specify the ``script`` attribute with the path to the TCL script that Jasper Gold will execute.

4. **SVA File Location**: Define the ``pycfile`` attribute to indicate where the generated SystemVerilog Assertions (SVA) file will be stored.

5. **Design List**: Use the ``design_list`` attribute to specify the design list file that Jasper Gold will use.

6. **Port Configuration**: Ensure the ``port`` attribute is set to the correct port number for connecting to the Jasper server.

By following these steps, you can effectively set up the Jasper interface for formal verification tasks in PyCaliper.
