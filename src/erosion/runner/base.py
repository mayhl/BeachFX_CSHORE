from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ..profile import Profile


class CSHOREResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    zb: np.ndarray  # final bed elevation, meters, shape (N,)
    x: np.ndarray  # cross-shore positions, meters, shape (N,)
    eta: np.ndarray  # mean water level at final BC timestep (OSETUP "setup")
    Hs: np.ndarray  # significant wave height at final BC timestep
    runup_m: float  # 2% runup from ODOC, meters (0.0 if not reported)
    # Landward wet-computation limit node count (hydro valid over nodes < jr); 0 = unknown.
    jr: int = 0


class CSHORERunner(ABC):
    @abstractmethod
    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult: ...
