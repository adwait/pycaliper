from pycaliper.per import *
from pycaliper.per.per import unroll


class refiner_module1(SpecModule):
    def __init__(self, name="", **kwargs):
        super().__init__(name, **kwargs)
        self.sig1 = Logic(8)
        self.sig2 = Logic(8)

    @unroll(3)
    def simsched1(self, i: int = 0) -> None:
        if i < 3:
            self.pycassume(self.sig1 == self.sig2)

    @unroll(4)
    def simsched2(self, i: int = 0) -> None:
        if i < 3:
            self.pycassume(self.sig1 <= self.sig2)


class refiner_module2(SpecModule):
    def __init__(self, name="", **kwargs):
        super().__init__(name, **kwargs)
        self.sig1 = Logic(8)
        self.sig2 = Logic(8)

    @unroll(4)
    def simsched1(self, i: int = 0) -> None:
        if i < 3:
            self.pycassume(self.sig1 == self.sig2)
        if i == 3:
            self.pycassert(self.sig1 == self.sig2)

    @unroll(4)
    def simsched2(self, i: int = 0) -> None:
        if i < 4:
            self.pycassume(self.sig1 <= self.sig2)
