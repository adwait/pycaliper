import hashlib
import dill as pickle

from pycaliper.pycconfig import PYConfig, Design


class JGDesign(Design):
    def __init__(self, name: str, pyc: PYConfig) -> None:
        assert not pyc.mock, f"JasperDesign {name} cannot operate in mock mode!"
        self.name = name
        self.pyc = pyc

    def __hash__(self):
        return hashlib.md5(pickle.dumps(self.pyc)).hexdigest()
