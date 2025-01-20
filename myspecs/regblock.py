from pycaliper.per import Module, Logic, Const


class reg_en(Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.rst = Logic()
        self.en = Logic()
        self.d = Logic(self.width)
        self.q = Logic(self.width)

    def input(self):
        self.inv(~self.en)
        self.inv(~self.rst)
        pass

    def state(self):
        # self.inv(Const(0, 1))
        self.inv(self.q == Const(100, self.width))

    def output(self):
        # self.inv(Const(0, 1))
        pass
