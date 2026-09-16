import numpy as np
from typing import List, Tuple, Optional, Union


class _LatticeSize(tuple):
    """Contenedor tuple-like para preservar semánticas antiguas de red cuadrada."""

    def __new__(cls, nx: int, ny: int):
        return super().__new__(cls, (int(nx), int(ny)))

    def __pow__(self, power: int):
        if power == 2 and self[0] == self[1]:
            return self[0] ** power
        return NotImplemented


class LatticeSOS_v2:
    """
    Red Solid-On-Solid (SOS) con columnas de altura entera y 4 vecinos.

    Esta versión no codifica restricciones geométricas específicas de caras
    cristalinas; la diferencia entre sistemas o caras debe entrar por los
    parámetros del modelo y no por máscaras sobre la red.
    """

    def __init__(
        self,
        size: Union[List[int], Tuple[int, int], int],
        seed: Optional[int] = None,
        debug: bool = False,
    ):
        if isinstance(size, int):
            size = (size, size)
        self.size = _LatticeSize(size[0], size[1])
        self.nx, self.ny = self.size
        self.rng = np.random.default_rng(seed)
        self.heights = np.zeros((self.nx, self.ny), dtype=np.int32)
        self.debug = debug

    def initialize(
    self,
    mode: str = "flat",
    max_height: int = 1,
    n_seeds: int = 0,
    rng: Optional[np.random.Generator] = None,
    screw_center: Optional[Tuple[float, float]] = None,
    screw_charge: int = 1,
    screw_pitch: int = 1,
    screw_core_radius: float = 1.0,
    screw_noise: int = 0,
    ):
        """
        Configura el estado inicial.

        Modos:
        - flat: toda la superficie a altura 0
        - random: alturas enteras aleatorias entre 0 y max_height
        - seeds: n_seeds sitios aleatorios de altura 1
        - screw: rampa helicoidal discreta tipo dislocación de tornillo
        """
        gen = rng if rng is not None else self.rng

        if mode == "flat":
            self.heights.fill(0)

        elif mode == "random":
            self.heights =  gen.integers(
                0,
                max(1, int(max_height) + 1),
                size=self.heights.shape,
                dtype=np.int32,
            )

        elif mode == "seeds":
            self.heights.fill(0)
            total_sites = self.nx * self.ny
            n_pick = max(0, min(int(n_seeds), total_sites))
            flat_indices = gen.choice(total_sites, size=n_pick, replace=False)
            for flat_idx in np.atleast_1d(flat_indices):
                i = int(flat_idx) // self.ny
                j = int(flat_idx) % self.ny
                self.heights[i, j] = 1

        elif mode == "screw":
            self.heights.fill(0)

            # Centro del núcleo helicoidal
            if screw_center is None:
                cx = (self.nx - 1) / 2.0
                cy = (self.ny - 1) / 2.0
            else:
                cx, cy = screw_center

            screw_charge = 1 if screw_charge >= 0 else -1
            screw_pitch = max(1, int(screw_pitch))
            core_r2 = float(screw_core_radius) ** 2

            # Coordenadas de la red
            ii, jj = np.indices((self.nx, self.ny), dtype=float)
            dx = ii - cx
            dy = jj - cy

            # Ángulo polar en [-pi, pi]
            theta = np.arctan2(dy, dx)

            # Lo llevamos a [0, 2pi)
            phase = (theta + 2.0 * np.pi) % (2.0 * np.pi)

            # Dislocación de tornillo discreta:
            # una vuelta completa alrededor del centro incrementa la altura en screw_pitch
            if screw_charge > 0:
                heights = np.floor((phase / (2.0 * np.pi)) * screw_pitch)
            else:
                heights = np.floor(((2.0 * np.pi - phase) / (2.0 * np.pi)) * screw_pitch)

            heights = heights.astype(np.int32)

            # Núcleo: evita singularidad artificial en el centro
            r2 = dx * dx + dy * dy
            heights[r2 <= core_r2] = 0

            # Opcional: pequeña rugosidad local para romper simetría exacta
            if screw_noise > 0:
                noise = gen.integers(
                    0,
                    int(screw_noise) + 1,
                    size=heights.shape,
                    dtype=np.int32,
                )
                heights = heights + noise

            # Normalizar para que la altura mínima sea 0
            heights -= heights.min()

            self.heights = heights.astype(np.int32)

        else:
            raise ValueError(f"Modo de inicialización desconocido: {mode}")


    # def initialize(
    #     self,
    #     mode: str = "flat",
    #     max_height: int = 1,
    #     n_seeds: int = 0,
    #     rng: Optional[np.random.Generator] = None,
    # ):
    #     """
    #     Configura el estado inicial sin imponer morfologías artificiales del paper.

    #     Modos:
    #     - flat: toda la superficie a altura 0
    #     - random: alturas enteras aleatorias entre 0 y max_height
    #     - seeds: n_seeds sitios aleatorios de altura 1
    #     """
    #     gen = rng if rng is not None else self.rng

    #     if mode == "flat":
    #         self.heights.fill(0)
    #     elif mode == "random":
    #         self.heights = gen.integers(
    #             0,
    #             max(1, int(max_height) + 1),
    #             size=self.heights.shape,
    #             dtype=np.int32,
    #         )
    #     elif mode == "seeds":
    #         self.heights.fill(0)
    #         total_sites = self.nx * self.ny
    #         n_pick = max(0, min(int(n_seeds), total_sites))
    #         flat_indices = gen.choice(total_sites, size=n_pick, replace=False)
    #         for flat_idx in np.atleast_1d(flat_indices):
    #             i = int(flat_idx) // self.ny
    #             j = int(flat_idx) % self.ny
    #             self.heights[i, j] = 1
    #     else:
    #         raise ValueError(f"Modo de inicialización desconocido: {mode}")

    def wrap(self, idx: int, axis: int) -> int:
        n = self.size[axis]
        return (idx + n) % n

    def neighbors4(self, site: Tuple[int, int]) -> List[Tuple[int, int]]:
        i, j = site
        return [
            (self.wrap(i - 1, 0), j),
            (self.wrap(i + 1, 0), j),
            (i, self.wrap(j - 1, 1)),
            (i, self.wrap(j + 1, 1)),
        ]

    def neighbor_in_direction(self, site: Tuple[int, int], direction_idx: int) -> Tuple[int, int]:
        """
        Vecino cardinal por índice:
        0 = norte, 1 = sur, 2 = oeste, 3 = este.
        """
        return self.neighbors4(site)[int(direction_idx) % 4]

    def get_height(self, site: Tuple[int, int]) -> int:
        return int(self.heights[site])

    def set_height(self, site: Tuple[int, int], h: int):
        self.heights[site] = int(h)

    def inc_height(self, site: Tuple[int, int], dh: int = 1):
        if self.debug and dh <= 0:
            raise ValueError(f"inc_height requiere dh>0, got {dh}")
        self.heights[site] += int(dh)

    def dec_height(self, site: Tuple[int, int], dh: int = 1):
        h = int(self.heights[site])
        if self.debug and h < dh:
            raise AssertionError(f"Altura negativa: h={h}, dh={dh} en {site}")
        if h >= dh:
            self.heights[site] = h - dh

    def lateral_neighbors_at_level(self, site: Tuple[int, int], level: int) -> int:
        count = 0
        for n in self.neighbors4(site):
            if self.heights[n] >= level:
                count += 1
        return count

    def adsorption_bonds(self, site: Tuple[int, int]) -> int:
        h = self.get_height(site)
        # return self.lateral_neighbors_at_level(site, h + 1)
        return self.lateral_neighbors_at_level(site, h + 1)

    def desorption_bonds(self, site: Tuple[int, int]) -> int:
        h = self.get_height(site)
        if h == 0:
            return 0
        return self.lateral_neighbors_at_level(site, h)

    def migration_targets(self, site: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retorna solo vecinos con altura estrictamente menor que la del origen.
        """
        h = self.get_height(site)
        if h == 0:
            return []
        targets = []
        for n in self.neighbors4(site):
            if self.get_height(n) < h:
                targets.append(n)
        return targets

    def has_lower_neighbor(self, site: Tuple[int, int]) -> bool:
        h = self.get_height(site)
        if h == 0:
            return False
        for n in self.neighbors4(site):
            if self.get_height(n) < h:
                return True
        return False

    def get_sites(self) -> List[Tuple[int, int]]:
        idx_i, idx_j = np.indices(self.size)
        return list(zip(idx_i.flatten(), idx_j.flatten()))

    def mean_height(self) -> float:
        return float(np.mean(self.heights))

    def roughness(self) -> float:
        return float(np.std(self.heights))

    def step_density(self) -> float:
        count = 0
        total = 0
        for site in self.get_sites():
            if self.get_height(site) > 0:
                if self.desorption_bonds(site) == 1:
                    count += 1
                total += 1
        return count / total if total > 0 else 0.0

    def count_by_coordination(self) -> dict:
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for site in self.get_sites():
            i = self.adsorption_bonds(site)
            counts[i] += 1
        return counts


# Alias ligero para compatibilidad local si algún consumidor aún usa el nombre antiguo.
LatticeSOS = LatticeSOS_v2
