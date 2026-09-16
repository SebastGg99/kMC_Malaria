import numpy as np

# =============================
# Utils numéricos robustos
# =============================
_MAX_EXP_ARG = 700.0  # np.exp(700) ~ 1e304 (límite seguro en float64)

EMPTY, HEMIN, STEP = 0, 1, 2
VECINOS = [(1,0), (-1,0), (0,1), (0,-1)]  # von Neumann

def _safe_exp(x: float) -> float:
    """exp(x) con protección contra overflow."""
    if x > _MAX_EXP_ARG:
        x = _MAX_EXP_ARG
    elif x < -_MAX_EXP_ARG:
        x = -_MAX_EXP_ARG
    return float(np.exp(x))

def _finite_or_zero(x: float) -> float:
    """Devuelve x si es finito; 0.0 en caso contrario."""
    return float(x) if np.isfinite(x) else 0.0

# Comentario: datos experimentales aproximados leídos de la figura objetivo.
FACE_DATA = {
    "110": {
        "delta": 0.63,
        "E_pb_over_kT": 0.48,
        "phi_over_kT": 3.76,
        "exp_sigma": np.array([1.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0], dtype=float),
        "exp_rate": np.array([0.00, 0.10, 0.17, 0.21, 0.28, 0.35, 0.42], dtype=float),
        "title": "(a) Cara 110",
        "marker": "+",
    },
    "101": {
        "delta": 0.30,
        "E_pb_over_kT": 2.12,
        "phi_over_kT": 4.27,
        "exp_sigma": np.array([1.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0], dtype=float),
        "exp_rate": np.array([0.00, 0.00, 0.09, 0.10, 0.17, 0.21, 0.28, 0.42], dtype=float),
        "title": "(b) Cara 101",
        "marker": "*",
    },
}

def in_bounds(i, j, Lx, Ly):
    """Verifica si (i,j) está dentro de la grilla"""
    return (0 <= i < Lx) and (0 <= j < Ly)

def count_bonds(grid, i, j):
    """Contador de vecinos estabilizantes HEMIN/STEP"""
    cnt = 0
    Lx, Ly = grid.shape
    for dx, dy in VECINOS:
        x, y = i+dx, j+dy
        if in_bounds(x, y, Lx, Ly) and grid[x, y] in (HEMIN, STEP):
            cnt += 1
    return cnt

def count_bonds_all(grid):
    """Crea un mapa con el número de vecinos para cada celda"""
    Lx, Ly = grid.shape
    bonds = np.zeros_like(grid)
    for i in range(Lx):
        for j in range(Ly):
            bonds[i, j] = count_bonds(grid, i, j)
    return bonds

# Si el sitio contiene una molécula de hemina, se revisa si tiene algún vecino vacío.
# Si lo tiene, está en el borde de una isla → devuelve True.
def is_step_site(grid, i, j):
    """Sitio de borde/kink"""

    if grid[i,j] == HEMIN:
        return any(in_bounds(i+dx,j+dy,*grid.shape) and grid[i+dx,j+dy]==EMPTY
                   for dx,dy in VECINOS)
    if grid[i,j] == EMPTY:
        return any(in_bounds(i+dx,j+dy,*grid.shape) and grid[i+dx,j+dy]==HEMIN
                   for dx,dy in VECINOS)
    return False

def arrhenius(nu0, Ea, T, kB):
    """k = nu0 * exp(-max(Ea,0)/(kBT))"""
    Ea_eff = max(0.0, Ea)
    return nu0 * np.exp(-Ea_eff/(kB*T))

def barrier_decrease(Ea_base, bonds, eps):
    """Para ads/inc: más vecinos -> barrera menor"""
    return max(0.0, Ea_base - bonds*eps)

def barrier_increase(Ea_base, bonds, eps):
    """Para des/diff: más vecinos -> barrera mayor"""
    return Ea_base + bonds*eps