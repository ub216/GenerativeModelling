import random
from typing import List

from helpers.utils import log_once_warning


def drop_condition(conditioning: List[str], r: float) -> List[str]:
    """
    conditioning: Conditioning text
    r: percentage (0–1) of elements to replace with null condition ""
    """
    assert r <= 1.0
    N = len(conditioning)
    if r == 1.0:
        log_once_warning("Drop conditioning is set to 1.0. All conditioning will be dropped")
        return [""] * N
    k = int(N * r)  # how many to blank out
    indices = random.sample(range(N), k)  # pick k random positions

    # copy list so original is not modified
    drop_conditioning = conditioning[:]
    for i in indices:
        drop_conditioning[i] = ""
    return drop_conditioning
