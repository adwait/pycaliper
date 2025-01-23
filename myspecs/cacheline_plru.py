from pycaliper.per import Module, Logic, LogicArray
from pycaliper.per.per import unroll
import math


from enum import Enum


class TMode(Enum):
    ADV = 0
    VIC = 1


class cacheline_plru(Module):
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
        self.eq(self.reset)
        # The cacheline receives the same kind of request
        self.eq(self.os_req)
        self.eq(self.user_req)
        # Either it is a User request or an OS request
        self.inv(~self.os_req | ~self.user_req)
        self.inv(self.os_req | self.user_req)
        # self.inv(self.user_req)

        # self.inv((~(self.os_req & self.attacker_domain)) | (~`attacker_domain))

        # Attacker
        if self.mode == TMode.ADV.value:
            self.eq(self.hitmap)
            self.eq(self.addr)
        # User
        else:
            pass

    def output(self):
        if self.mode == TMode.ADV.value:
            # self.eq(self.policy_hitmap(7, 1))
            self.eq(self.hit)
        else:
            pass

    def state(self):

        if self.mode == TMode.ADV.value:
            self.eq(self.tags)
            self.eq(self.metadata)
            self.eq(self.valid)
            self.eq(self.policy_hitmap)

            self.eq(self.plru_policy(7, 1))
            self.eq(self.plru_mask(7, 1))

            self.eq(self.victim_way)
            self.eq(self.hit_way)
        else:
            pass

    @unroll(3)
    def simstep(self, i):
        if i == 0:
            self.pycassert(self.attacker_domain)
