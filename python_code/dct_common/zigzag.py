"""Standard JPEG zig-zag scan order for an 8x8 DCT block.

Generated algorithmically rather than hand-typed: this order is a fixed
property of the 8x8 DCT block itself, independent of capture resolution
(that only changes how many blocks there are, not the traversal order
within each block).
"""

import numpy as np


def generate_zigzag_order(size: int = 8) -> list:
    """Standard zig-zag scan order for a `size x size` block, as (row, col) pairs."""
    order = []
    row, col = 0, 0
    going_up = True
    for _ in range(size * size):
        order.append((row, col))
        if going_up:
            if col == size - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == size - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1
    return order


ZIGZAG_ORDER = generate_zigzag_order(8)
assert len(ZIGZAG_ORDER) == 64
assert len(set(ZIGZAG_ORDER)) == 64


def generate_axis_first_order(size: int = 8) -> list:
    """DC, then pure-vertical/pure-horizontal AC positions alternating --
    (1,0),(0,1),(2,0),(0,2),...,(size-1,0),(0,size-1) -- tried before any
    mixed/diagonal term, unlike standard zig-zag order which interleaves
    diagonal terms in early (e.g. (1,1) is its 5th position). Tests
    whether a model benefits more from single-axis frequency information
    than from those early diagonal terms.

    After the 2*(size-1) pure-axis positions are exhausted, the remaining
    (row>0 and col>0) positions are appended in their relative standard
    zig-zag order -- keeps this a total ordering of all size*size
    positions, so any num_coeffs up to size*size is valid, same as
    ZIGZAG_ORDER."""
    order = [(0, 0)]
    for k in range(1, size):
        order.append((k, 0))
        order.append((0, k))
    order.extend((r, c) for r, c in ZIGZAG_ORDER if r > 0 and c > 0)
    return order


AXIS_FIRST_ORDER = generate_axis_first_order(8)
assert len(AXIS_FIRST_ORDER) == 64
assert len(set(AXIS_FIRST_ORDER)) == 64

# Registry keyed by Config.coeff_scan_order's valid values.
SCAN_ORDERS = {"zigzag": ZIGZAG_ORDER, "axis_first": AXIS_FIRST_ORDER}


def zigzag_extract(block: np.ndarray, num_coeffs: int, order: list = ZIGZAG_ORDER) -> np.ndarray:
    """First `num_coeffs` values of an 8x8 block in the given scan `order`
    (default: standard JPEG zig-zag; pass SCAN_ORDERS[cfg.coeff_scan_order]
    for a config-selected order)."""
    return np.array([block[r, c] for r, c in order[:num_coeffs]], dtype=block.dtype)


def self_test() -> None:
    for order in (ZIGZAG_ORDER, AXIS_FIRST_ORDER):
        block = np.zeros((8, 8), dtype=np.int32)
        for rank, (r, c) in enumerate(order):
            block[r, c] = rank
        extracted = zigzag_extract(block, 64, order=order)
        assert np.array_equal(extracted, np.arange(64)), extracted
    assert AXIS_FIRST_ORDER[:15] == [
        (0, 0),
        (1, 0), (0, 1), (2, 0), (0, 2), (3, 0), (0, 3),
        (4, 0), (0, 4), (5, 0), (0, 5), (6, 0), (0, 6), (7, 0), (0, 7),
    ], AXIS_FIRST_ORDER[:15]
