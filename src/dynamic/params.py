"""Parámetros de la línea DINÁMICA: `KMCParamsDynamic`.

Origen: `params.py` (clase `KMCParams`), movido por error a
`.descartables/scripts/versions/` aunque `bkl.py` lo seguía importando. Idéntico al
commit 5415fbb salvo el nombre de la clase y los comentarios.

Nota: `fixed_sigma` NO es un campo del dataclass. El motor dinámico lo lee con
`getattr(params, "fixed_sigma", None)`, así que se puede asignar como atributo
después de crear el objeto (`params.fixed_sigma = 2.0`). En esta línea se interpreta
directamente como S (no como σ), a diferencia de la línea estática
(auditoria.md §5.2.3).
"""

from dataclasses import dataclass
from typing import Optional  # noqa: F401  (se conserva el import del original)


@dataclass
class KMCParamsDynamic:
    T: float              # temperatura (solo informativa; las energías ya van /kT)
    K0_plus: float        # prefactor ads/des/mig
    K_inc_plus: float     # prefactor de incorporación
    E_pb_over_kT: float   # energía de enlace lateral / kT
    phi_over_kT: float    # energía de superficie / kT
    delta: float          # parámetro adaptativo de la adsorción
    # Sobresaturación dinámica: S = ln(C / C_eq), con C = N_bulk / V
    V: float
    C_eq: float
    # fixed_sigma: Optional[float] = None
    S_floor: float = -5.0
    S_ceil: float = 8.0
