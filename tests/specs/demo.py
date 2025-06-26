from pycaliper.per import *
from pycaliper.per.per import unroll

DONE = Const(3, 2)


LIMIT = 6 # SAFE
# LIMIT = 7 # UNSAFE

class demo(SpecModule):
    def __init__(self, name="", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.clk = Clock()
        self.rst = Logic(1, "rst")

        # Inputs
        self.start = Logic(1, "start")
        self.stop = Logic(1, "stop")

        # Outputs
        self.state_ = Logic(2, "state")
        self.counter = Logic(3, "counter")

    def state(self) -> None:
        # self.inv(self.counter <= Const(6, 3))
        self.inv(~(self.state_ == DONE) | (self.counter >= Const(5, 3)))

    def output(self) -> None:
        # self.inv(~(self.state_ == DONE) | (self.counter >= Const(5, 3)))
        pass

    # Bounded proofs
    def get_reset_seq(self, i: int) -> None:
        if i == 0:
            self.pycassume(~self.rst)
        else:
            self.pycassume(self.rst)

    @unroll(9)
    def simstep(self, i: int = 0) -> None:
        if i > 0:
            self.pycassert(~(self.state_ == DONE) | (self.counter >= Const(LIMIT, 3)))
        self.get_reset_seq(i)
