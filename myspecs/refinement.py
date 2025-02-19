from pycaliper.per import *


class FIFO(SpecModule):
    def __init__(self, name="", width=3, **kwargs):
        super().__init__(name)
        self.width = width

        self.push = Logic()
        self.pop = Logic()

        self.rd_ptr = Logic(width)
        self.wr_ptr = Logic(width)

    def input(self):
        self.inv(~self.push | ~self.pop)
        self.inv(self.push | self.pop)

    def state(self):
        pass

    def output(self):
        self.inv(
            ~self.prev(self.push) | (self.incr(self.wr_ptr) & self.stable(self.rd_ptr))
        )
        self.inv(
            ~self.prev(self.pop) | (self.incr(self.rd_ptr) & self.stable(self.wr_ptr))
        )


class Counter(SpecModule):
    def __init__(self, name="", width=3, **kwargs):
        super().__init__(name)
        self.width = width

        self.add = Logic()

        self.count = Logic(width)

    def input(self):
        pass

    def state(self):
        pass

    def output(self):
        self.inv(
            ~self.prev(self.add)
            | ((self.prev(self.count) + Const(1, self.width)) == self.count)
        )
        self.inv(
            self.prev(self.add)
            | ((self.prev(self.count) - Const(1, self.width)) == self.count)
        )
