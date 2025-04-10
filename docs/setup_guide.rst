Setup and Install
============================

Prerequisites
-------------

Before you begin, ensure you have the following installed on your system:

- **Git**: To clone the repository.
- **Python**: Ensure you have Python 3.11 or later and ``pip`` installed.

Step 1: Clone the Repositories
------------------------------

First, clone the PyCaliper and btor2ex repositories from GitHub. Open your terminal and run the following commands:

.. code-block:: bash

   git clone https://github.com/pycaliper/pycaliper.git
   git clone https://github.com/pycaliper/btor2ex.git

Now either perform a quick setup (Step 2a) or build the package (Step 2b).

Step 2a (quick, no build): Add to ``PYTHONPATH``
------------------------------------------------

1. Create a virtual environment (e.g., using ``pyenv``), and install dependencies using:

   .. code-block:: bash

      pip install -r requirements.txt

2. To ensure that the ``pycaliper`` and ``btor2ex`` modules are accessible, add their paths to the ``PYTHONPATH`` environment variable. You can do this by running the following commands in your terminal:

   .. code-block:: bash

      export PYTHONPATH=$PYTHONPATH:$(pwd)/pycaliper
      export PYTHONPATH=$PYTHONPATH:$(pwd)/btor2ex

3. (optional) Run tests to verify the setup:

   .. code-block:: bash

      cd pycaliper
      python3 -m tests.test

Step 2b (build): Build and Install the Packages
-----------------------------------------------

Set up the virtual environment as before. Now install the projects using ``pyproject.toml``. For example:

.. code-block:: bash

   cd <path to pycaliper>
   pip install .
   cd <path to btor2ex>
   pip install .

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
