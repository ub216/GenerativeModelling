import random
from typing import List


def drop_condition(conditioning: List[str], r: float) -> List[str]:
    """
    conditioning: Conditioning text
    r: percentage (0–1) of elements to replace with null condition ""
    """
    assert r < 1
    N = len(conditioning)
    k = int(N * r)  # how many to blank out
    indices = random.sample(range(N), k)  # pick k random positions

    # copy list so original is not modified
    drop_conditioning = conditioning[:]
    for i in indices:
        drop_conditioning[i] = ""
    return drop_conditioning
