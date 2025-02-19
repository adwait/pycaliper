from tqdm import tqdm
import logging

from vcdvcd import VCDVCD
from pycaliper.vcdutils import get_subtrace, XVALUE

from pycaliper.synth import tracesampler
from pycaliper.pycmanager import PYCArgs, setup_pyc_tmgr_jg
from pycaliper.per import Const, get_path_from_hierarchical_str

from myspecs import sdram

# Log to file
logging.basicConfig(level=logging.DEBUG, filename="debug.log", filemode="w")
logger = logging.getLogger(__name__)


args = PYCArgs(
    specpath="myspecs/sdram.sdram_controller_cover",
    jgcpath="designs/sdram/config_cover.json",
    sdir="sdram_traces_new",
)
is_conn, pyconfig, tmgr = setup_pyc_tmgr_jg(args)
pyconfig.clk = get_path_from_hierarchical_str("miter.a.clk")
pyconfig.ctx = "miter.a"

sdramspec = sdram.sdram_controller_cover(pyconfig)

ts = tracesampler.TraceSampler(tmgr, pyconfig, sdramspec)
ts.get_t_traces(2)


tmgr.save()
print("Saved traces to {}".format(tmgr.sdir))
