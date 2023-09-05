import numpy as np

from typing import Any, List, Tuple, NamedTuple

class Dimension(NamedTuple):
    width: float
    height: float

class AntennaParameters:
    def __init__(self, bounds: Tuple[List[float]], targets: Tuple[List[float]], freqs: np.ndarray, num_nodes: int, stats_per_node: int):

        assert len(bounds) == 2 and len(bounds[0]) == len(bounds[1])
        self._bounds = bounds
        self._num_nodes = num_nodes
        self._stats_per_node = stats_per_node
        self._targets = targets
        self._freqs = freqs

    @property
    def bounds(self) -> int:
        return self._bounds

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def stats_per_node(self) -> int:
        return self._stats_per_node

    @property
    def targets(self) -> Any:
        return self._targets

    @property
    def freqs(self) -> Any:
        return self._freqs

   
    def calculate_score(self, preds: np.ndarray) -> np.ndarray:
        assert preds.ndim == 2 and preds.shape[1] == len(self.freqs)
        values = np.zeros(len(preds))
        orig_values = np.zeros(len(preds))
        for fl, fh, t, w in self.targets:
            freq_mask = np.arange(len(self.freqs))[(fl * 0.9999 < self.freqs) & (self.freqs < fh * 1.0001)]
            pred = np.clip(preds[:, freq_mask], 1e-10, float('inf'))
            try:
                diff = 20.0 * np.log10(pred) - t
                values = np.fmax(values,np.clip(diff, 0.0, float('inf')).max(axis=1) * w)
                # orig_values = np.fmax( orig_values, diff.max(axis=1) * w)
                # values += np.clip(diff, 0.0, float('inf')).max(axis=1) * w
                # orig_values += diff.max(axis=1) * w
            except Exception as err:
                print(f'error found: {err}')
        #idx = np.argwhere(values == 0.0)
        #values[idx] = orig_values[idx]
        return values



PATCH_SIZES =      [Dimension(0.75, 4),
                    Dimension(17.64, 1.7),
                    Dimension(11.38, 3),
                    Dimension(18, 0.56),
                    Dimension(0.99, 2.43),
                    Dimension(0.6, 4),
                    ]
NUM_PATCHES = 6
SIXPATCH_TARGETS: List[Tuple[float]] = [(2.4, 2.5, -6.0, 1), (5.1, 7.0, -6.0, 1)]

# 12 parameters: x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
SIXPATCH_GRP2_BOUNDS: Tuple[List[float]] = (
        [    1, 0.5,      1,   2,    1 ,  2,     1,     2,     1,    -1,    1,    -1], # the lower bound array of the 12 parameters
        [   21, 0.5,      6,   5,   13,   5,      6,    5,    21,     5,   21,     5]   # the upper bound array of the 12 parameters
    )
SIXPATCH_FREQS = np.linspace(0,7,71)

six_patch_antenna = AntennaParameters(bounds=SIXPATCH_GRP2_BOUNDS, targets=SIXPATCH_TARGETS, freqs=SIXPATCH_FREQS, num_nodes=6, stats_per_node=2)


