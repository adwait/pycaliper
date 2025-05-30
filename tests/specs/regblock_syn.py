from pycaliper.per import SpecModule, Logic, Clock


class reg_en(SpecModule):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.rst = Logic()
        self.en = Logic()
        self.d = Logic(self.width)
        self.q = Logic(self.width)

    def input(self):
        self.eq(self.rst)
        self.eq(self.en)
        self.when(self.en)(self.d)

    def state(self):
        self.eq(self.q)
        # self.hole([self.q, self.d])

    def output(self):
        self.eq(self.q)


class regblock_syn(SpecModule):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.clk = Clock()

        # Submodules
        self.reg1 = reg_en()
        self.reg2 = reg_en()

        # Wires
        self.rst = Logic()
        self.en = Logic()
        self.d = Logic(self.width)
        self.q = Logic(self.width)
        self.rd_index = Logic()
        self.wr_index = Logic()

    def input(self):
        self.eq(self.rst)
        self.eq(self.en)
        self.when(self.en)(self.d)
        self.when(self.en)(self.wr_index)
        # self.eq(self.d)
        # self.eq(self.wr_index)
        self.eq(self.rd_index)

    def state(self):
        self.eqhole([self.reg1.q, self.reg2.q, self.q])
        # self.eq(self.reg2.q)

    def output(self):
        self.eq(self.q)
