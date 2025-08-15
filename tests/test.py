import logging
import sys
import os
import unittest
from tempfile import NamedTemporaryFile

from pycaliper.pycsetup import PYCArgs, PYCTask, start
from pycaliper.proofmanager import mk_btordesign, ProofManager
from pycaliper.frontend.pyclex import lexer
from pycaliper.frontend.pycparse import parser
from pycaliper.frontend.pycgen import PYCGenPass
from pycaliper.jginterface.jgdesign import JGDesign
from pycaliper.svagen import SVAGen
from pycaliper.pycconfig import DesignConfig
from pycaliper.verif.btorverifier import BTORVerifier2Trace
from pycaliper.verif.jgverifier import (
    JGVerifier1TraceBMC,
    JGVerifier1Trace,
    JGVerifier2Trace,
)
from pycaliper.verif.refinementverifier import RefinementVerifier
from pycaliper.synth.persynthesis import (
    PERSynthesizer,
    HoudiniSynthesizerJG,
    HoudiniSynthesizerBTOR,
)
from pycaliper.synth.iis_strategy import SeqStrategy
from tests.specs.regblock import regblock
from tests.specs.regblock_syn import regblock_syn
from tests.specs.array_nonzerobase import array_nonzerobase, array_nonzerobase2
from tests.specs.counter import counter
from tests.specs.adder import adder
from tests.specs.refiner_modules import refiner_module1, refiner_module2
from tests.specs.demo import demo

# Configure logging
h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.INFO)
h1.setFormatter(logging.Formatter("%(levelname)s::%(message)s"))

h2 = logging.FileHandler("test_debug.log", mode="w")
h2.setLevel(logging.DEBUG)
h2.setFormatter(logging.Formatter("%(asctime)s::%(name)s::%(levelname)s::%(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[h1, h2])
logger = logging.getLogger(__name__)

# Check if JG-related tests should be enabled
ENABLE_JG_TESTS = "ENABLE_JG_TESTS" in os.environ


class SVAGenTest(unittest.TestCase):
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


class JGVerifierTest(unittest.TestCase):
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
        (pyconfig, regb) = self.gen_test(
            "tests/specs/regblock", "examples/designs/regblock/config.json"
        )
        invverif = JGVerifier2Trace()
        regb.instantiate()
        invverif.verify(regb, pyconfig)

    def test_counter(self):
        (pyconfig, counter) = self.gen_test(
            "tests/specs/counter", "examples/designs/counter/config.json"
        )
        invverif = JGVerifier1Trace()
        counter.instantiate()
        invverif.verify(counter, pyconfig)


class JGSynthesisTest(unittest.TestCase):
    def test_jgsynthesizer(self):
        pyconfig, module = start(
            PYCTask.PERSYNTH,
            PYCArgs(
                specpath="tests/specs/regblock_syn.regblock_syn",
                jgcpath="examples/designs/regblock/config_syn.json",
                dcpath="",
                params="",
                sdir="",
            ),
        )
        synthesizer = HoudiniSynthesizerJG()
        module.instantiate()
        finalmod, _ = synthesizer.synthesize(
            module, JGDesign("regblock", pyconfig), pyconfig.dc, strategy=SeqStrategy()
        )


class JGBMCTest(unittest.TestCase):
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
        (pyconfig, module) = self.gen_test(
            "tests/specs/adder.adder", "examples/designs/adder/config.json", "simstep"
        )
        verifier = JGVerifier1TraceBMC()
        logger.debug("Running BMC verification.")
        module.instantiate()
        verifier.verify(module, pyconfig, "simstep")


class ParserTest(unittest.TestCase):
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


class BTORVerifierTest(unittest.TestCase):
    def test_btorverifier1(self):
        prgm = mk_btordesign("regblock", "examples/designs/regblock/btor/regblock.btor")
        engine = BTORVerifier2Trace()
        self.assertTrue(
            engine.verify(
                regblock().instantiate(), prgm, DesignConfig(cpy1="a", cpy2="b")
            ).verified
        )


class BTORSynthesisTest(unittest.TestCase):
    def test_btorsynthesizer(self):
        prgm = mk_btordesign("regblock", "examples/designs/regblock/btor/regblock.btor")
        synthesizer = HoudiniSynthesizerBTOR()
        module = regblock_syn()
        module.instantiate()
        finalmod, status = synthesizer.synthesize(
            module, prgm, DesignConfig(), strategy=SeqStrategy()
        )
        self.assertTrue(status.success)
        self.assertTrue(finalmod is not None)


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


class ProofManagerTest(unittest.TestCase):
    def test_demo_btor(self):

        pm = ProofManager()
        prgm = pm.mk_btor_design_from_file(
            "examples/designs/demo/btor/full_design.btor", "demo"
        )
        spec = pm.mk_spec(demo, "demo_spec")
        pr = pm.mk_btor_proof_one_trace(spec, prgm)
        self.assertTrue(pr.result)


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestSuite()

    # Add non-JG-related tests
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(SVAGenTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ParserTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(BTORVerifierTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(BTORSynthesisTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ReprTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(RefinementVerifierTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ProofManagerTest))

    # Add JG-related tests if enabled
    if ENABLE_JG_TESTS:
        logger.info("JG tests are enabled.")
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(JGVerifierTest))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(JGSynthesisTest))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(JGBMCTest))
    else:
        logger.info("JG tests are disabled.")
        # Remove JG-related tests from the suite
        for test in [JGVerifierTest, JGSynthesisTest, JGBMCTest]:
            suite._tests = [t for t in suite._tests if not isinstance(t, test)]

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)
