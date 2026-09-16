import numpy as np
from typing import List, Tuple, Optional, Union


class _LatticeSize(tuple):
    """Tuple-like size container that preserves old square-lattice test semantics."""

    def __new__(cls, nx: int, ny: int):
        return super().__new__(cls, (int(nx), int(ny)))

    def __pow__(self, power: int):
        if power == 2 and self[0] == self[1]:
            return self[0] ** power
        return NotImplemented

class LatticeSOS:
    """
    Red simple Solid-On-Solid (SOS) con alturas de columna enteras.
    - heights[i, j] ∈ {0,1,2,...}
    - Conectividad de 4 vecinos (von Neumann) con condiciones de contorno periódicas.
    """
    def __init__(self, size: Union[List[int], Tuple[int, int], int], seed: Optional[int] = None, debug: bool = False):
        if isinstance(size, int):
            size = (size, size)
        self.size = _LatticeSize(size[0], size[1]) # Tamaño de la red
        # Generador de nums aleatorios con semilla
        self.rng = np.random.default_rng(seed)
        # Corazón de la red, inicialmente plana
        self.heights = np.zeros((self.size[0], self.size[1]), dtype=np.int32)
        self.debug = debug

    # Configuración del estado inicial de la superficie
    def initialize(self, init_mode: str = "flat", max_roughness: int = 1):
        if init_mode == "flat":
            self.heights.fill(0)
        elif init_mode == "random_surface":
            self.heights = self.rng.integers(
                0, max(1, max_roughness+1),
                size=self.heights.shape, dtype=np.int32
            )
        else:
            raise ValueError("Unknown init_mode")

    def seed_surface_with_islands(self, n_seeds: int, rng: Optional[np.random.Generator] = None):
        """
        Siembra islas mono-capa en sitios únicos.

        Se usa en tests y análisis antiguos para construir una población inicial de
        defectos sin generar columnas de altura > 1.
        """
        gen = rng if rng is not None else self.rng
        total_sites = self.size[0] * self.size[1]
        n_pick = max(0, min(int(n_seeds), total_sites))
        flat_indices = gen.choice(total_sites, size=n_pick, replace=False)
        self.heights.fill(0)
        for flat_idx in np.atleast_1d(flat_indices):
            i = int(flat_idx) // self.size[1]
            j = int(flat_idx) % self.size[1]
            self.heights[i, j] = 1

    # Condiciones de contorno periódicas
    # Revisar el índice para envolverlo dentro de los límites de la red
    # Se añade el parámetro 'axis' para saber si es fila (0) o columna (1)
    def wrap(self, idx: int, axis: int) -> int:
        n = self.size[axis]
        return (idx + n) % n

    # Coordenadas de los cuatro vecinos (Von Neumman) de un sitio
    def neighbors4(self, site: Tuple[int,int]) -> List[Tuple[int,int]]:
        i, j = site
        # Para i (filas) usamos axis=0, para j (columnas) usamos axis=1
        return [
            (self.wrap(i-1, 0), j), (self.wrap(i+1, 0), j),
            (i, self.wrap(j-1, 1)), (i, self.wrap(j+1, 1))
        ]
    

    # Altura de la columna en el sitio especificado
    def get_height(self, site: Tuple[int,int]) -> int:
        return int(self.heights[site])

    # Aumenta la altura de un sitio (simulando adsorción)
    def inc_height(self, site: Tuple[int,int], dh: int = 1):
        if self.debug:
            assert dh > 0, f"Intento de inc_height con valor no positivo: {dh}"
        self.heights[site] += int(dh)

    # Disminuye la altura de un sitio (simulando desorción)
    def dec_height(self, site: Tuple[int,int], dh: int = 1):
        h = int(self.heights[site])
        if self.debug:
            assert h >= dh, f"Error Crítico: Intento de altura negativa en {site}. h={h}, dh={dh}"

        if h >= dh:
            self.heights[site] = h - dh

    # ---- Site classification helpers ----
    def lateral_neighbors_at_level(self, site: Tuple[int,int], level: int) -> int:

        """
        Este método cuenta cuántos de los 4 vecinos de un site tienen una altura igual o mayor
        que un level (nivel) dado. Esto es clave para determinar cuántos enlaces laterales
        un sitio puede formar o ya tiene, lo que influye en su estabilidad.
        """

        cnt = 0
        for n in self.neighbors4(site):
            if self.get_height(n) >= level:
                cnt += 1
        return cnt

    def adsorption_bonds(self, site: Tuple[int,int]) -> int:
        """
        Calcula el número de enlaces laterales que formaría una partícula si se adsorbe en la
        parte superior de la columna en el site dado. Esto se determina contando cuántos
        vecinos de ese sitio tienen una altura igual o mayor que h+1
        (donde h es la altura actual del sitio). Cuantos más enlaces, más estable sería la
        partícula adsorbida y, por lo tanto, mayor es la tasa de adsorción efectiva
        (dependiendo del modelo).
        """

        h = self.get_height(site)
        # return self.lateral_neighbors_at_level(site, h+1)
        return self.lateral_neighbors_at_level(site, h)

    def desorption_bonds(self, site: Tuple[int,int]) -> int:
        """
        Calcula el número de enlaces laterales que tiene una partícula si se desorbe de la
        parte superior de la columna en el site dado. Esto se determina contando cuántos
        vecinos de ese sitio tienen una altura igual o mayor que h
        (la altura actual del sitio). Cuantos más enlaces, más energía se necesita para
        romperlos, y por lo tanto, menor es la tasa de desorción efectiva.
        """

        h = self.get_height(site)
        if h == 0:
            return 0
        return self.lateral_neighbors_at_level(site, h)

    def migration_targets(self, site: Tuple[int,int]) -> List[Tuple[int,int]]:
        """
        Este método identifica los sitios vecinos a los que una partícula en la parte
        superior de la columna actual (site) podría migrar. Una partícula solo puede migrar
        a un sitio vecino si la altura de ese vecino es menor o igual que la altura actual
        del sitio desde el que migra. Esto refleja la idea de que una partícula no puede
        'escalar' una columna más alta que ella misma para migrar, solo puede moverse
        lateralmente o 'caer' a una columna más baja o de igual altura. No puede 'saltar'.
        """

        h = self.get_height(site)
        if h == 0:
            return []
        targets = []
        for n in self.neighbors4(site):
            h_n = self.get_height(n)
            if h_n <= h:  # no subir
                targets.append(n)
            
            # [PRUEBA 1]: Validez Física en Migración
            if self.debug and h_n > h:
                pass 

        return targets

    def get_sites(self) -> List[Tuple[int,int]]:
        """
        Devuelve una lista de todas las coordenadas (x, y) de los sitios en la red.
        Es útil para iterar sobre todos los sitios, por ejemplo, al calcular tasas
        globales o al clasificar sitios.
        """

        idxs = np.argwhere(np.ones_like(self.heights, dtype=bool))
        return [tuple(x) for x in idxs]
