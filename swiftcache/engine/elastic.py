"""Paper Algorithm 1 arithmetic, independent of CUDA and transport."""
import math
from collections import deque


def elastic_units(master_block_elements, worker_block_elements):
    if min(master_block_elements, worker_block_elements) <= 0:
        raise ValueError('Block sizes must be positive')
    shared = math.lcm(master_block_elements, worker_block_elements)
    return shared//master_block_elements, shared//worker_block_elements


def align_up(blocks, unit):
    if blocks < 0 or unit <= 0:
        raise ValueError('Invalid alignment')
    return ((blocks+unit-1)//unit)*unit


class DemandWindow:
    def __init__(self, seconds, clock):
        if seconds <= 0:
            raise ValueError('Demand window must be positive')
        self.seconds = seconds
        self.clock = clock
        self.samples = deque()

    def observe(self, blocks):
        self.samples.append((self.clock(), blocks))

    def maximum(self):
        cutoff = self.clock()-self.seconds
        while self.samples and self.samples[0][0] <= cutoff:
            self.samples.popleft()
        return max((b for _,b in self.samples), default=0)
