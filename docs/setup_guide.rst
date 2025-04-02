Setup and Install
============================

Prerequisites
-------------

Before you begin, ensure you have the following installed on your system:

- **Git**: To clone the repository.
- **Python**: Ensure you have Python 3.11 or later and ``pip`` installed.

Step 1: Clone the Repository
----------------------------

First, clone the repository from GitHub. Open your terminal and run the following command:

.. code-block:: bash

   git clone https://github.com/pycaliper/pycaliper
   cd pycaliper


Now either perform a quik setup (Step 2a) or build the package (Step 2b).

Step 2a (quick, no build): Add to ``PYTHONPATH``
-----------------------------------------------------------

1. Create a virtual environment (e.g., using ``pyenv``), and install dependencies using:

   .. code-block:: bash

      pip install -r requirements.txt

2. To ensure that the ``pycaliper`` module is accessible, add its path to the ``PYTHONPATH`` environment variable. You can do this by running the following command in your terminal:

   .. code-block:: bash

      export PYTHONPATH=$(pwd):$PYTHONPATH

Step 2b (build): Build and Install the Package
------------------------------------------------

Set up the virtual environment as before. Now install the project using ``pyproject.toml`` with ``build`` (or an alternative package manager such as ``poetry``). For example:

.. code-block:: bash

   # Install build if not already installed
   pip install build
   # Build the package
   python -m build

Step 3: Verify the Installation
-------------------------------

To verify that the installation was successful, run the test suite using ``unittest``:

.. code-block:: bash

    python tests/test.py

This will run all tests that do not depend on the Jasper FV app.

(*optional*) If you have access to Jasper FV, you can also run the Jasper tests:

.. code-block:: bash

   # Start Jasper FV App by running:
   <path_to_jg> jasperserver.tcl
   # INSIDE the Jasper FV App, create a server connection on port 8080
   # jg> jg_start_server 8080

   # Now run the tests with Jasper FV enabled
   ENABLE_JG_TESTS=1 python tests/test.py
