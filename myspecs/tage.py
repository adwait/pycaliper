from pycaliper.per import *
from pycaliper.per.per import unroll

from dataclasses import dataclass


INIT = Const(0b00, 2)
USER = Const(0b01, 2)
PRIV = Const(0b10, 2)
SHAR = Const(0b11, 2)
BOUNDARY = Const(0x80000000, 32)


@dataclass
class tage_config:
    # `define BR_TAKEN        1'b1
    BR_TAKEN: int = 1
    # `define BR_NOT_TAKEN    1'b0
    BR_NOT_TAKEN: int = 0

    # `define BHT_IDX_WIDTH       11
    # BHT_IDX_WIDTH: int = 11
    BHT_IDX_WIDTH: int = 6

    # `define TAGE_IDX_WIDTH      (`BHT_IDX_WIDTH - 2)
    # TAGE_IDX_WIDTH: int = 9
    TAGE_IDX_WIDTH: int = BHT_IDX_WIDTH - 2
    # `define TAGE_WEAK_TAKEN     3'b100
    # TAGE_STRONG_TAKEN: Expr = Const(4, 3)
    # `define TAGE_WEAK_NOT_TAKEN 3'b011
    # TAGE_STRONG_NOT_TAKEN: Expr = Const(3, 3)

    # `define GHIST_LEN   130
    GHIST_LEN: int = 130
    # `define PHIST_LEN   16
    PHIST_LEN: int = 16

    # `define PP_THRESHOLD    2863311530
    PP_THRESHOLD: int = 2863311530


class bht(Module):
    def __init__(self, name="", config: tage_config = tage_config(), **kwargs) -> None:
        super().__init__(name, **kwargs)
        # Inputs
        self.br_result_i = Logic(1, "br_result_i")
        self.update_en_i = Logic(1, "update_en_i")
        self.idx_i = Logic(config.BHT_IDX_WIDTH, "idx_i")
        # State
        self.bht_data = LogicArray(
            lambda: Logic(2), 1 << config.BHT_IDX_WIDTH, name="bht_data"
        )

        self.domain_i = Logic(2, "domain_i")
        self.bht_data_priv = LogicArray(
            lambda: Logic(2), 1 << config.BHT_IDX_WIDTH, name="bht_data_priv"
        )
        self.bht_data_user = LogicArray(
            lambda: Logic(2), 1 << config.BHT_IDX_WIDTH, name="bht_data_user"
        )
        self.bht_targ_priv = LogicArray(
            lambda: Logic(32), 1 << config.BHT_IDX_WIDTH, name="bht_targ_priv"
        )
        self.bht_targ_user = LogicArray(
            lambda: Logic(32), 1 << config.BHT_IDX_WIDTH, name="bht_targ_user"
        )

        self.prev_domain = Logic(2, "prev_domain")

        # Outputs
        self.prediction_o = Logic(1, "prediction_o")
        self.targ_o = Logic(32, "targ_o")


class tage_table(Module):
    def __init__(self, name="", config: tage_config = tage_config(), **kwargs) -> None:
        super().__init__(name, **kwargs)

        # Inputs
        # input logic br_result_i, update_u_i, dec_u_i, alloc_i, provider_i,
        self.br_result_i = Logic(1, "br_result_i")
        self.update_u_i = Logic(1, "update_u_i")
        self.dec_u_i = Logic(1, "dec_u_i")
        self.alloc_i = Logic(1, "alloc_i")
        self.provider_i = Logic(1, "provider_i")
        # input logic [`TAGE_IDX_WIDTH-1:0] hash_idx_i,
        self.hash_idx_i = Logic(config.TAGE_IDX_WIDTH, "hash_idx_i")
        # input logic [8:0] hash_tag_i,
        self.hash_tag_i = Logic(9, "hash_tag_i")
        self.targ_i = Logic(32, "targ_i")
        self.domain_i = Logic(2, "domain_i")

        # State
        self.ctr = LogicArray(lambda: Logic(3), 1 << config.TAGE_IDX_WIDTH, name="ctr")
        self.tag = LogicArray(lambda: Logic(9), 1 << config.TAGE_IDX_WIDTH, name="tag")
        self.u = LogicArray(lambda: Logic(2), 1 << config.TAGE_IDX_WIDTH, name="u")

        self.targ = LogicArray(
            lambda: Logic(32), 1 << config.TAGE_IDX_WIDTH, name="targ"
        )
        self.isolation_state = LogicArray(
            lambda: Logic(2), 1 << config.TAGE_IDX_WIDTH, name="isolation_state"
        )

        self.prev_idx = Logic(config.TAGE_IDX_WIDTH, "prev_idx")
        self.prev_hash_tag = Logic(9, "prev_hash_tag")
        self.prev_domain = Logic(2, "prev_domain")

        self.u_clear_ctr = Logic(18, "u_clear_ctr")
        self.u_clear_col = Logic()

        # Outputs
        # output logic prediction_o, tag_hit_o, new_entry_o,
        self.prediction_o = Logic(1, "prediction_o")
        self.targ_o = Logic(32, "targ_o")
        self.tag_hit_o = Logic(1, "tag_hit_o")
        self.new_entry_o = Logic(1, "new_entry_o")
        # output logic [1:0] u_o
        self.u_o = Logic(2, "u_o")


class tage_predictor(Module):
    def __init__(self, name="", config: tage_config = tage_config(), **kwargs) -> None:
        super().__init__(name, **kwargs)

        # Inputs
        self.br_result_i = Logic(1, "br_result_i")
        self.update_en_i = Logic(1, "update_en_i")
        self.correct_i = Logic(1, "correct_i")
        self.idx_i = Logic(32, "idx_i")

        # Outputs
        self.tage_prediction_o = Logic(33, "tage_prediction_o")

        self.ghist = Logic(config.GHIST_LEN, "ghist")
        # logic [`PHIST_LEN-1:0] phist;
        # self.phist = Logic(config.PHIST_LEN, "phist")
        # logic [3:0] alt_ctr;
        self.alt_ctr = Logic(4, "alt_ctr")

        self.allocs = Logic(4)
        # info: should really be unneccessary
        self.dec_us = Logic(4)

        self.c_T0 = bht("c_T0", config=config)
        self.c_T1 = tage_table("c_T1", config=config)
        self.c_T2 = tage_table("c_T2", config=config)
        self.c_T3 = tage_table("c_T3", config=config)
        self.c_T4 = tage_table("c_T4", config=config)

        self.prev_domain = Logic(2, "prev_domain")


class top_(Module):
    def __init__(self, name="", config: tage_config = tage_config(), **kwargs) -> None:
        super().__init__(name, **kwargs)

        self.rst_i = Logic(1, "rst_i")

        self.br_result_i = Logic(1, "br_result_i")
        self.update_en_i = Logic(1, "update_en_i")
        self.correct_i = Logic(1, "correct_i")
        self.idx_i = Logic(32, "idx_i")

        self.domain_i = Logic(2, "domain_i")

        # self.tage_prediction_o = Logic(33, "tage_prediction_o")
        self.prediction_o = Logic(1, "prediction_o")
        self.targ_o = Logic(32, "targ_o")

        self.tp = tage_predictor("tp", config=config)

        self.config = config


class top(top_):
    def __init__(self, name="", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def input(self):
        # Reset input
        self.eq(self.rst_i)

        # top
        self.eq(self.br_result_i)
        self.eq(self.update_en_i)
        self.eq(self.correct_i)
        self.eq(self.idx_i)

        # ! Extra: to be removed
        self.eq(self.tp.allocs)
        self.eq(self.tp.dec_us)

        self.eq(self.domain_i)

        # self.when(self.domain_i == PRIV)(self.br_result_i)

    def state(self):
        self.eq(self.tp.ghist)
        self.eq(self.tp.alt_ctr)

        # bht
        self.eq(self.tp.c_T0.bht_data)
        self.eq(self.tp.c_T0.prediction_o)

        for tab in [self.tp.c_T1, self.tp.c_T2, self.tp.c_T3, self.tp.c_T4]:
            # Outputs
            self.eq(tab.prediction_o)
            self.eq(tab.tag_hit_o)
            self.eq(tab.new_entry_o)
            self.eq(tab.u_o)
            # State
            self.eq(tab.ctr)
            self.eq(tab.u)
            self.eq(tab.tag)

            # when(tab.isolation_state[i] == PRIV)
            # when(tab.isolation_state[i] == PRIV)
            # when(tab.isolation_state[i] == PRIV)

            self.eq(tab.isolation_state)

            # Aux state
            self.eq(tab.u_clear_ctr)
            self.eq(tab.u_clear_col)

    def output(self):
        # Outer main equivalence signal
        # self.eq(self.prediction_o)
        pass


class boundary_spec(top_):
    def __init__(self, name="", config: tage_config = tage_config(), **kwargs) -> None:
        super().__init__(name, config, **kwargs)

    def input(self):
        self.inv((~(self.domain_i == PRIV)) | (self.tp.idx_i < BOUNDARY))

    def state(self):
        for i in range(1 << self.config.BHT_IDX_WIDTH):
            self.inv(self.tp.c_T0.bht_targ_priv[i] < BOUNDARY)
        for tab in [self.tp.c_T1, self.tp.c_T2, self.tp.c_T3, self.tp.c_T4]:
            for i in range(1 << self.config.TAGE_IDX_WIDTH):
                self.inv((~(tab.isolation_state[i] == PRIV)) | (tab.targ[i] < BOUNDARY))

            # self.inv((~(tab.domain_i == PRIV)) | (tab.targ_i < BOUNDARY))

    def output(self):
        self.inv((~(self.tp.prev_domain == PRIV)) | (self.targ_o < BOUNDARY))
