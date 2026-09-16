"""Lattice SOS v4.

Mantiene la misma topologia que v3; los cambios de v4 estan en parametros
y motor.
"""

try:
    from .lattice_v3 import LatticeSOS_v3 as LatticeSOS_v4
except ImportError:  # pragma: no cover - compatibility with legacy script-style imports
    from lattice_v3 import LatticeSOS_v3 as LatticeSOS_v4

# Alias local para compatibilidad con consumidores previos.
LatticeSOS = LatticeSOS_v4
