import numpy as np
from typing import List, Tuple, Optional, Union


class _LatticeSize(tuple):
    """Tuple-like size container for square-lattice semantics."""

    def __new__(cls, nx: int, ny: int):
        return super().__new__(cls, (int(nx), int(ny)))

    def __pow__(self, power: int):
        if power == 2 and self[0] == self[1]:
            return self[0] ** power
        return NotImplemented


class LatticeSOS_v3:
    """Solid-On-Solid (SOS) lattice with directional bond counting.

    This version mirrors LatticeSOS_v2 and adds directional bond counts
    to support shape anisotropy (x vs y).
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
        max_height: int = 0,
        n_seeds: int = 0,
        rng: Optional[np.random.Generator] = None,
        screw_center: Optional[Tuple[float, float]] = None,
        screw_charge: int = 1,
        screw_pitch: int = 1,
        screw_core_radius: float = 1.0,
        screw_noise: int = 0,
    ):
        """Configure initial surface state.

        Modes:
        - flat: all heights to 0
        - random: random integer heights in [0, max_height]
        - seeds: compact center island with n_seeds sites at height 1
        - screw: discrete screw dislocation profile
        """
        gen = rng if rng is not None else self.rng

        if mode == "flat":
            self.heights.fill(0)

        elif mode == "random":
            self.heights = gen.integers(
                0,
                max(1, int(max_height) + 1),
                size=self.heights.shape,
                dtype=np.int32,
            )

        # elif mode == "seeds":
        #     self.heights.fill(0)
        #     total_sites = self.nx * self.ny
        #     n_pick = max(0, min(int(n_seeds), total_sites))
        #     # Build a compact island by selecting the closest sites to the lattice center.
        #     if n_pick > 0:
        #         cx = (self.nx - 1) / 2.0
        #         cy = (self.ny - 1) / 2.0
        #         ii, jj = np.indices((self.nx, self.ny), dtype=float)
        #         dist2 = (ii - cx) * (ii - cx) + (jj - cy) * (jj - cy)
        #         flat_order = np.argsort(dist2, axis=None)
        #         for flat_idx in flat_order[:n_pick]:
        #             i = int(flat_idx) // self.ny
        #             j = int(flat_idx) % self.ny
        #             self.heights[i, j] = 1

        elif mode == "seeds":
            import heapq

            # Reiniciar toda la lattice
            self.heights.fill(0)

            # Verificación básica de consistencia geométrica
            assert self.heights.shape == (self.nx, self.ny), (
                f"La forma de self.heights {self.heights.shape} no coincide con "
                f"(nx, ny)=({self.nx}, {self.ny})"
            )

            total_sites = self.nx * self.ny
            n_pick = max(0, min(int(n_seeds), total_sites))

            # Si no hay semillas que poner, terminar aquí
            if n_pick == 0:
                return

            # Centro geométrico discreto de la red
            # Para tamaños pares, toma la celda central inferior/izquierda.
            cx = self.nx // 2
            cy = self.ny // 2

            # Conjunto de sitios ya descubiertos para no repetirlos
            discovered = np.zeros((self.nx, self.ny), dtype=bool)

            # Cola de prioridad para crecer desde el centro hacia afuera
            # Orden:
            #   1) distancia Manhattan al centro
            #   2) distancia euclidiana al cuadrado
            #   3) coordenadas (i, j) para desempate determinista
            heap = []

            def push_site(i: int, j: int) -> None:
                """Inserta un sitio vecino válido en la cola de expansión."""
                if 0 <= i < self.nx and 0 <= j < self.ny and not discovered[i, j]:
                    discovered[i, j] = True
                    d1 = abs(i - cx) + abs(j - cy)
                    d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy)
                    heapq.heappush(heap, (d1, d2, i, j))

            # Semilla inicial en el centro
            push_site(cx, cy)

            # Vecindad 4-conexa: arriba, abajo, izquierda, derecha
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            placed = 0

            while heap and placed < n_pick:
                _, _, i, j = heapq.heappop(heap)

                # Colocar la semilla en este sitio
                self.heights[i, j] = 1
                placed += 1

                # Expandir solo hacia vecinos contiguos
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    push_site(ni, nj)

        elif mode == "screw":
            self.heights.fill(0)

            # Helical core center
            if screw_center is None:
                cx = (self.nx - 1) / 2.0
                cy = (self.ny - 1) / 2.0
            else:
                cx, cy = screw_center

            screw_charge = 1 if screw_charge >= 0 else -1
            screw_pitch = max(1, int(screw_pitch))
            core_r2 = float(screw_core_radius) ** 2

            ii, jj = np.indices((self.nx, self.ny), dtype=float)
            dx = ii - cx
            dy = jj - cy

            theta = np.arctan2(dy, dx)
            phase = (theta + 2.0 * np.pi) % (2.0 * np.pi)

            if screw_charge > 0:
                heights = np.floor((phase / (2.0 * np.pi)) * screw_pitch)
            else:
                heights = np.floor(((2.0 * np.pi - phase) / (2.0 * np.pi)) * screw_pitch)

            heights = heights.astype(np.int32)

            r2 = dx * dx + dy * dy
            heights[r2 <= core_r2] = 0

            if screw_noise > 0:
                noise = gen.integers(
                    0,
                    int(screw_noise) + 1,
                    size=heights.shape,
                    dtype=np.int32,
                )
                heights = heights + noise

            heights -= heights.min()
            self.heights = heights.astype(np.int32)

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

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
        """Cardinal neighbor by index: 0=N, 1=S, 2=W, 3=E."""
        return self.neighbors4(site)[int(direction_idx) % 4]

    def get_height(self, site: Tuple[int, int]) -> int:
        return int(self.heights[site])

    def set_height(self, site: Tuple[int, int], h: int):
        self.heights[site] = int(h)

    def inc_height(self, site: Tuple[int, int], dh: int = 1):
        if self.debug and dh <= 0:
            raise ValueError(f"inc_height requires dh>0, got {dh}")
        self.heights[site] += int(dh)

    def dec_height(self, site: Tuple[int, int], dh: int = 1):
        h = int(self.heights[site])
        if self.debug and h < dh:
            raise AssertionError(f"Negative height: h={h}, dh={dh} at {site}")
        if h >= dh:
            self.heights[site] = h - dh

    def lateral_neighbors_at_level(self, site: Tuple[int, int], level: int) -> int:
        """Count neighbors with height exactly equal to level."""
        count = 0
        for n in self.neighbors4(site):
            if self.heights[n] == level:
                count += 1
        return count

    def lateral_neighbors_xy_at_level(self, site: Tuple[int, int], level: int) -> Tuple[int, int]:
        """Directional count of neighbors (x/y) at the given level."""
        ix = 0
        iy = 0
        i0, j0 = site
        for n in self.neighbors4(site):
            if self.heights[n] == level:
                if n[0] != i0:
                    ix += 1
                else:
                    iy += 1
        return ix, iy

    def adsorption_bonds(self, site: Tuple[int, int]) -> int:
        h = self.get_height(site)
        return self.lateral_neighbors_at_level(site, h + 1)

    def adsorption_bonds_xy(self, site: Tuple[int, int]) -> Tuple[int, int]:
        h = self.get_height(site)
        return self.lateral_neighbors_xy_at_level(site, h + 1)

    def desorption_bonds(self, site: Tuple[int, int]) -> int:
        h = self.get_height(site)
        if h == 0:
            return 0
        return self.lateral_neighbors_at_level(site, h)

    def desorption_bonds_xy(self, site: Tuple[int, int]) -> Tuple[int, int]:
        h = self.get_height(site)
        if h == 0:
            return (0, 0)
        return self.lateral_neighbors_xy_at_level(site, h)

    def migration_targets(self, site: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return neighbors with strictly lower height than the origin."""
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
