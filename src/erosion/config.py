from __future__ import annotations

from pydantic import BaseModel, Field

from .interstorm import ErosionConfig, GenCadeErosionConfig, SLCConfig, UniformErosionConfig
from .nourishment import NourishmentConfig
from .profile import ProfileGeometryConfig
from .runner.local import CSHOREParams
from .storm import StormConfig

__all__ = [
    "ProfileGeometryConfig",
    "CSHOREParams",
    "StormConfig",
    "SLCConfig",
    "UniformErosionConfig",
    "GenCadeErosionConfig",
    "ErosionConfig",
    "NourishmentConfig",
    "ReachConfig",
]


class ReachConfig(BaseModel):
    """All policy parameters for one reach."""

    storm: StormConfig = Field(default_factory=StormConfig)
    cshore: CSHOREParams = Field(default_factory=CSHOREParams)
    erosion: ErosionConfig | None = None
    slc: SLCConfig | None = None
    nourishment: NourishmentConfig | None = None
