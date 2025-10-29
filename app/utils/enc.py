import numpy as np
from typing import Optional

def rle_encode(mask) -> Optional[str]:
    if mask is None:
        return None
    m = (mask.astype(np.uint8).flatten(order="F") > 0).astype(np.uint8)
    if m.size == 0:
        return ""
    diffs = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    lengths = ends - starts
    pairs = []
    for s, l in zip(starts, lengths):
        pairs.append(str(int(s)))
        pairs.append(str(int(l)))
    return " ".join(pairs)
