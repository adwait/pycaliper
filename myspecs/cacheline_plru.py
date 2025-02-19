from pycaliper.per import SpecModule, Logic, LogicArray, unroll, kinduct
from pycaliper.per.expr import *
import math

from enum import Enum


def is_nonzero(x: Logic):
    return OpApply(UnaryBitwiseOr(), [x])


class TMode(Enum):
    ADV = 0
    VIC = 1


class cacheline_plru(SpecModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.NUM_WAYS = kwargs.get("NUM_WAYS", 8)
        self.NUM_WAYS_WIDTH = int(math.log(self.NUM_WAYS, 2))
        self.mode = int(kwargs.get("MODE"))
        # Reset input
        self.reset = Logic()

        # OS request and requested address field
        self.os_req = Logic()
        self.hitmap = Logic(self.NUM_WAYS)

        # Auxilliary state
        self.attacker_domain = Logic(root="miter")
        self.attacker_hitmap = Logic(self.NUM_WAYS, root="miter")

        # User request and input
        self.user_req = Logic()
        self.addr = Logic(32)
        self.tags = LogicArray(lambda: Logic(32), self.NUM_WAYS)

        # Outputs
        self.hit = Logic()

        self.policy_hitmap = Logic(self.NUM_WAYS)

        # Logic
        self.valid = Logic(self.NUM_WAYS)
        self.plru_mask = Logic(self.NUM_WAYS)
        self.plru_policy = Logic(self.NUM_WAYS)
        self.metadata = Logic(self.NUM_WAYS)
        self.victim_way = Logic(self.NUM_WAYS_WIDTH)
        self.hit_way = Logic(self.NUM_WAYS_WIDTH)

    def input(self):
        # Don't resets
        self.inv(~self.reset)

        # Only consider user requests
        self.inv(~self.os_req & self.user_req)

        self.inv(is_nonzero(self.policy_hitmap))
        self.inv(is_nonzero(self.attacker_hitmap))
        # self.inv(~(self.attacker_domain) | (self.hitmap == self.attacker_hitmap))
        self.when(self.attacker_domain)(self.addr)

    def output(self):
        self.when(self.attacker_domain)(self.hit)

    @kinduct(2)
    def state(self):

        # Attacker query
        self.inv(~self.attacker_domain | (self.policy_hitmap == self.attacker_hitmap))
        # Non-attacker query
        self.inv(
            self.attacker_domain
            | OpApply(
                UnaryBitwiseAnd(), [(~self.policy_hitmap) | (~self.attacker_hitmap)]
            )
        )

        for i in range(self.NUM_WAYS):
            self.when(self.attacker_hitmap(i))(self.policy_hitmap(i))
            self.when(self.attacker_hitmap(i, i))(self.valid(i, i))
            self.when(self.attacker_hitmap(i, i) & self.valid(i, i))(self.tags[i])

        self.when(self.attacker_domain)(self.policy_hitmap)

        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap(1, 0)]))(
            self.metadata(4, 4)
        )
        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap(3, 2)]))(
            self.metadata(5, 5)
        )
        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap(5, 4)]))(
            self.metadata(6, 6)
        )
        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap(7, 6)]))(
            self.metadata(7, 7)
        )
        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap(3, 0)]))(
            self.metadata(2, 2)
        )
        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap(7, 4)]))(
            self.metadata(3, 3)
        )
        self.when(OpApply(UnaryBitwiseOr(), [self.attacker_hitmap]))(
            self.metadata(1, 1)
        )

        # if self.mode == TMode.ADV.value:
        #     self.eq(self.tags)
        #     self.eq(self.metadata)
        #     self.eq(self.valid)
        #     self.eq(self.policy_hitmap)

        #     self.eq(self.plru_policy(7, 1))
        #     self.eq(self.plru_mask(7, 1))

        #     self.eq(self.victim_way)
        #     self.eq(self.hit_way)
        # else:
        #     pass

    @unroll(3)
    def simstep(self, i):
        if i == 0:
            self.pycassert(self.attacker_domain)
