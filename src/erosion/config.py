from __future__ import annotations

from pydantic import BaseModel, Field

from .interstorm import ErosionConfig, GenCadeErosionConfig, SLCConfig, UniformErosionConfig
from .nourishment import GeometryThresholds, NourishmentConfig
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
    "GeometryThresholds",
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
    # Mean sea level (m, internal datum): the reference plane for subaerial
    # volume / erosion metrics.  Constant in time — relative SLC is applied to
    # the bed (see SLCConfig), so a moving MSL would double-count it.
    msl: float = 0.0
    # Depth of closure (m, positive depth below MSL): the seaward limit of the
    # active profile.  Reach-wide default; a profile may override it via
    # ProfileGeometryConfig.depth_of_closure.  Used by the nourishment placement
    # model to extend the subaerial deficit down to the full active height
    # (see nourishment.VolumeAssessor / FittedAssessor).  0.0 = no extension.
    depth_of_closure: float = 0.0
