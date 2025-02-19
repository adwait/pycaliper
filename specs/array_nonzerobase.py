from pycaliper.per import SpecModule, Logic, LogicArray
from pycaliper.per.per import kinduct, unroll
import math


class array_nonzerobase(SpecModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.DEPTH = kwargs.get("depth", 8)
        self.DEPTHIND = int(math.log2(self.DEPTH) + 1)

        self.WIDTH = kwargs.get("width", 64)

        ELEM_T = lambda: Logic(self.WIDTH)

        self.array_ents = LogicArray(ELEM_T, self.DEPTH, base=1)

    def input(self):
        for i in range(self.DEPTH):
            self.eq(self.array_ents[i])

    def output(self):
        pass

    def state(self):
        pass


class array_nonzerobase2(SpecModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.DEPTH = kwargs.get("depth", 8)
        self.DEPTHIND = int(math.log2(self.DEPTH) + 1)
        self.WIDTH = kwargs.get("width", 64)

        ELEM_T = lambda: Logic(self.WIDTH)

        self.array_ents = LogicArray(ELEM_T, self.DEPTH, base=1)

    def input(self):
        for i in range(self.DEPTH):
            self.eq(self.array_ents[i])

    def output(self):
        pass

    @kinduct(10)
    def state(self):
        pass

    @unroll(5)
    def ssim(self, i: int = 0):
        if i > 0:
            for j in range(self.DEPTH):
                self.pycassert(self.array_ents[j] == self.array_ents[j - 1])
