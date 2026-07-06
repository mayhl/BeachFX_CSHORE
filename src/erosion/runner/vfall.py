"""Sediment fall-velocity calculation.

Port of Brad's MATLAB code (originally Jarrell Smith, ERDC-CHL). Comments
starting with "%" are from the original MATLAB.

    %VFALL estimates fall velocity based on Soulsby's (1997) optimization.
    %      w = kvis/d* [sqrt(10.36^2 + 1.049 D^3) - 10.36]
    %
    %  where w = sediment fall speed (m/s)
    %        d = grain diameter (mm)
    %        T = temperature (deg C)
    %        S = salinity (ppt)
    %
    % Jarrell Smith, Coastal and Hydraulics Laboratory, ERDC, Vicksburg, MS
"""

from __future__ import annotations

import numpy as np


def _water_density(T: float, S: float) -> float:
    """Water density (kg/m^3) from temperature and salinity.

    Approximation from VanRijn, L.C. (1993) Handbook for Sediment Transport
    by Currents and Waves.
    """
    CL = (S - 0.03) / 1.805  # %VanRijn
    return 1000 + 1.455 * CL - 6.5e-3 * (T - 4 + 0.4 * CL) ** 2


def _kinematic_viscosity(T: float) -> float:
    """Kinematic viscosity of water (m^2/s).

    Approximation from VanRijn, L.C. (1989) Handbook of Sediment Transport.
    """
    return 1e-6 * (1.14 - 0.031 * (T - 15) + 6.8e-4 * (T - 15) ** 2)


def fall_velocity(d50: float, T: float, S: float) -> float:
    """Sediment fall (settling) velocity in m/s via Soulsby (1997).

    Parameters
    ----------
    d50 : grain diameter (mm)
    T   : water temperature (deg C)
    S   : salinity (ppt)
    """
    g = 9.81
    rho = _water_density(T, S)
    kvis = _kinematic_viscosity(T)
    rhos = 2650
    d = d50 / 1000.0  # %convert mm to m
    s = rhos / rho
    D = (g * (s - 1) / kvis**2) ** (1 / 3) * d
    return kvis / d * (np.sqrt(10.36**2 + 1.049 * D**3) - 10.36)  # %settling speed
