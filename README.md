

# PyCaliper: RTL Formal Specification and Verification Infrastructure

PyCaliper provides infrastructure for verifying and synthesizing specifications for RTL designs based on the Caliper specification language.

----



## Requirements and Setup

PyCaliper has been developed and tested with Python 3.11. We recommend using a [virtual-environment](https://github.com/pyenv/pyenv). PyCaliper can be run without building (see [no-build] below) or by building the package (see [install] below).


### No-build

0. Clone this repository and the `btor2ex` package that provides a symbolic simulator for `btor2`.

```bash
git clone https://github.com/pycaliper/pycaliper.git
git clone https://github.com/pycaliper/btor2ex.git
```

1. Install all Python packages using the `requirements.txt`, making sure to use the virtual environment. For example, using `pip`:

```bash
pip install -r requirements.txt
```


3. Add `pycaliper` and `btor2ex` to your `PYTHONPATH`. For example, in `bash`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/pycaliper
export PYTHONPATH=$PYTHONPATH:$(pwd)/btor2ex
```

4. (optional) Run tests

```bash
cd pycaliper
python3 -m tests.test
```


### Build and install

1. Clone this repository and the `btor2ex` package as above:

```bash
git clone https://github.com/pycaliper/pycaliper.git
git clone https://github.com/pycaliper/btor2ex.git
```

2. Install both packages using the `pyproject.toml`, making sure to use the virtual environment. For example, using `pip`:

```bash
cd <path to pycaliper>
pip install .
cd <path to btor2ex>
pip install .
```


## Basic Use

### Quickstart

The `quickstart.py` script provides a simple example of how to use PyCaliper. It checks the design from the `examples/designs/demo` directory. 
It uses the PyCaliper `demos` specification from the `tests/specs/demo.py` file. 

```bash
python quickstart.py
```

This examples uses the BTOR backend (using the `btor2ex` package) to check the design. 
PyCaliper also supports the Jasper backend.


#### PyCaliper Tool Script

Run the `main` script (after installing PyCaliper) to see the available commands:

```bash
$ pycaliper --help

 Usage: pycaliper [OPTIONS] COMMAND [ARGS]...

 PyCaliper: Specification Synthesis and Verification Infrastructure.

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                         │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.  │
│ --help                        Show this message and exit.                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ persynth   Synthesize invariants using Partial Equivalence Relations (PER).                                     │
│ svagen     Generate SystemVerilog Assertions (SVA) from a PyCaliper specification.                              │
│ verif      Verify invariants in a PyCaliper specification.                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## API and Tool Documentation

We provide a [documentation site](https://pycaliper.github.io) for PyCaliper.


## Contributing

The PyCaliper project welcomes external contributions through pull requests to the main branch.

We use pre-commit, so before contributing, please ensure that you run pre-commit and make sure all checks pass with
```
pre-commit install
pre-commit run --all-files
```

Please also run the provided tests and add further tests targetting newly contributed features.

## Feedback

We encourage feedback and suggestions via [GitHub Issues](https://github.com/pycaliper/pycaliper/issues).
