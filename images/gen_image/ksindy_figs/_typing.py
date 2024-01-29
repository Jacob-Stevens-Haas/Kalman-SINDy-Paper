from typing import Any
from typing import NewType
from typing import TypeVar

import numpy as np
from numpy.typing import NBitBase


NpFlt = np.dtype[np.floating[NBitBase]]
NTimesteps = NewType("NTimesteps", int)
NSeries = NewType("NSeries", int)
Float1D = np.ndarray[NTimesteps, NpFlt]
Float2D = np.ndarray[tuple[NSeries, NTimesteps], NpFlt]
FloatND = np.ndarray[Any, NpFlt]
FloatOrArray = TypeVar("FloatOrArray", float, FloatND)
