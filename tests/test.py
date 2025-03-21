import logging
import sys
import os

import unittest

from tempfile import NamedTemporaryFile

from pycaliper.pycmanager import PYCArgs, PYCTask, start
from pycaliper.proofmanager import mk_btordesign, ProofManager

from pycaliper.frontend.pyclex import lexer
from pycaliper.frontend.pycparse import parser
from pycaliper.frontend.pycgen import PYCGenPass

from pycaliper.verif.jgverifier import JGVerifier2Trace
from pycaliper.svagen import SVAGen
from pycaliper.btorinterface.pycbtorsymex import DesignConfig
from pycaliper.verif.btorverifier import BTORVerifier2Trace
from pycaliper.verif.jgverifier import JGVerifier1TraceBMC, JGVerifier1Trace
from pycaliper.verif.refinementverifier import RefinementVerifier

from tests.specs.regblock import regblock
from tests.specs.array_nonzerobase import array_nonzerobase, array_nonzerobase2
from tests.specs.counter import counter
from tests.specs.adder import adder
from tests.specs.refiner_modules import refiner_module1, refiner_module2
from tests.specs.demo import demo

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.INFO)
h1.setFormatter(logging.Formatter("%(levelname)s::%(message)s"))

h2 = logging.FileHandler("test_debug.log", mode="w")
h2.setLevel(logging.DEBUG)
h2.setFormatter(logging.Formatter("%(asctime)s::%(name)s::%(levelname)s::%(message)s"))

# Add filename and line number to log messages
logging.basicConfig(level=logging.DEBUG, handlers=[h1, h2])

logger = logging.getLogger(__name__)


class TestSVAGen(unittest.TestCase):
    def gen_sva(self, mod, svafile):
        svagen = SVAGen()
        # Write to temporary file
        with open(f"tests/out/{svafile}", "w") as f:
            mod.instantiate()
            svagen.create_pyc_specfile(mod, filename=f.name, dc=DesignConfig())
            print(f"Wrote SVA specification to temporary file {f.name}")

    def test_array_nonzerobase(self):
        self.gen_sva(array_nonzerobase(), "array_nonzerobase.pyc.sv")

    def test_array_nonzerobase2(self):
        self.gen_sva(array_nonzerobase2(), "array_nonzerobase2.pyc.sv")

    def test_regblock(self):
        self.gen_sva(regblock(), "regblock.pyc.sv")

    def test_auxmodule(self):
        self.gen_sva(counter(), "counter.pyc.sv")


class TestVerifier(unittest.TestCase):
    def gen_test(self, specpath, jgcpath):
        args = PYCArgs(
            specpath=specpath,
            jgcpath=jgcpath,
            params="",
            sdir="",
            onetrace=True,
            bmc="",
        )
        return start(PYCTask.VERIFBMC, args)

    def test_regblock(self):
        (pyconfig, tmgr, regb) = self.gen_test(
            "tests/specs/regblock", "examples/designs/regblock/config.json"
        )
        invverif = JGVerifier2Trace()
        regb.instantiate()
        invverif.verify(regb, pyconfig)
        tmgr.close()

    def test_counter(self):
        (pyconfig, tmgr, counter) = self.gen_test(
            "tests/specs/counter", "examples/designs/counter/config.json"
        )
        invverif = JGVerifier1Trace()
        counter.instantiate()
        invverif.verify(counter, pyconfig)
        tmgr.close()


class TestParser(unittest.TestCase):
    def load_test(self, testname):
        filename = os.path.join("tests/specs.caliper", testname)
        with open(filename, "r") as f:
            return f.read()

    def lex_file(self, filename):
        lexer.input(self.load_test(filename))
        # Write tokens to temporary file
        with NamedTemporaryFile(mode="w+", delete=False, dir="tests/out") as f:
            while True:
                tok = lexer.token()
                if not tok:
                    break  # No more input
                f.write(f"{tok}\n")
            print(f"Wrote tokens to temporary file {f.name}")

    def test_lexer1(self):
        self.lex_file("test1.caliper")

    def parse_file(self, filename):
        result = parser.parse(self.load_test(filename))
        pycgenpass = PYCGenPass()
        pycgenpass.run(result)

        # Print to named temporary file
        with NamedTemporaryFile(
            mode="w+", delete=False, dir="tests/out", suffix=".py"
        ) as f:
            f.write(pycgenpass.outstream.getvalue())
            print(f"Wrote PYC specification to temporary file {f.name}")

    def test_pycgen1(self):
        self.parse_file("test1.caliper")


class BTORInterfaceTest(unittest.TestCase):
    def test_btorverifier1(self):
        prgm = mk_btordesign("regblock", "tests/btor/regblock.btor")
        engine = BTORVerifier2Trace()
        self.assertTrue(
            engine.verify(
                regblock().instantiate(), prgm, DesignConfig(cpy1="A", cpy2="B")
            )
        )


class SymbolicSimulator(unittest.TestCase):
    def gen_test(self, specpath, jgcpath, bmc):
        args = PYCArgs(
            specpath=specpath,
            jgcpath=jgcpath,
            params="",
            sdir="",
            onetrace=True,
            bmc=bmc,
        )
        return start(PYCTask.VERIFBMC, args)

    def test_adder(self):
        (pyconfig, tmgr, module) = self.gen_test(
            "tests/specs/adder.adder", "examples/designs/adder/config.json", "simstep"
        )
        verifier = JGVerifier1TraceBMC()
        logger.debug("Running BMC verification.")
        module.instantiate()
        verifier.verify(module, pyconfig, "simstep")
        tmgr.close()


class ReprTest(unittest.TestCase):
    def test_adder(self):
        addermod = adder()
        addermod.instantiate()
        addermodstr = repr(addermod)
        self.assertIn(
            "adder(SpecModule)", addermodstr, "Module name not present in repr"
        )
        self.assertIn("@unroll(3)", addermodstr, "Unrolling not present in repr")
        self.assertIn("@kind(1)", addermodstr, "Kind decorator not present in repr")
        self.assertIn(
            "self.pycassume((~self.rst_ni))",
            addermodstr,
            "Assume statement not present in repr",
        )


class RefinementVerifierTest(unittest.TestCase):
    def test_bsr(self):
        rm = refiner_module1()
        rm.instantiate()
        rv = RefinementVerifier()
        res = rv.check_ss_refinement(rm, rm.simsched1, rm.simsched2)
        self.assertTrue(res)

    def test_bsr2(self):
        # This refinement requires you to flip assertions on the first module
        rm = refiner_module2()
        rm.instantiate()
        rv = RefinementVerifier()
        res = rv.check_ss_refinement(rm, rm.simsched1, rm.simsched2, True)
        self.assertTrue(res)


class ProofmanagerTest(unittest.TestCase):
    def test_demo_btor(self):

        pm = ProofManager()
        prgm = pm.mk_btor_design_from_file(
            "examples/designs/demo/btor/full_design.btor", "demo"
        )
        spec = pm.mk_spec(demo, "demo_spec")
        pr = pm.mk_btor_proof_one_trace(spec, prgm)
        self.assertTrue(pr.result)


if __name__ == "__main__":
    unittest.main()
