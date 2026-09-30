import numpy as np
import matplotlib.pyplot as plt
from params_v2 import KMCParams_v2
from lattice_v2 import LatticeSOS_v2
from bkl_v2 import KMC_BKL_v2

def compute_probabilities(sigmas, times, size=(75, 75), mode="flat", n_seeds=1, delta=1, t_scale=1):

    prob_sta = []
    kmcs = []
    snaps_list = []

    # L = LatticeSOS_v2(size=size, seed=42)
    # # L.initialize(mode="random", max_height=1, n_seeds=1905)
    # L.initialize(mode=mode)

    for sigma in sigmas:
        print(f"\n\n=== σ = {sigma} ===")

        L = LatticeSOS_v2(size=size, seed=42+int(sigma))
        # L.initialize(mode="random", max_height=1, n_seeds=1905)
        L.initialize(mode=mode, max_height=0, n_seeds=n_seeds)

        params = KMCParams_v2(
        K0_plus= 2.116718e-13, #0.211,
        E_pb_over_kT= 1.2704636027368605, #.48,
        phi_over_kT= 1.792478012079329, #3.76,
        delta=delta,
        V=1,
        C_eq=15.0,
        fixed_sigma=sigma,   # usa un valor como 3.0 si quieres sigma estático
        S_floor=-5.0,
        S_ceil=9.0,
        )

        kmc = KMC_BKL_v2(
            lattice=L,
            params=params,
            N_bulk0=1000,
            rng_seed=123,
            time_scale=t_scale,
            n_seeds=n_seeds,
            constant_concentration=True,  # batch dinámico
        )
        kmcs.append(kmc)

        snaps, stats = kmc.run(t_end=times[-1], snapshot_times=times)
        snaps_list.append(snaps)

        prob_sta.append(stats['adsorption_probs_history'])

        print(f"Datos para σ = {sigma} guardados.\n")

    return prob_sta, kmcs, snaps_list

# ============================================
# Extraer p0, p1, p2 desde prob_sta
# ============================================

def extract_probs(prob_sta):
    # Devuelve probabilidades promedio en la cola temporal para cada sigma
    p0, p1, p2 = [], [], []
    sigmas_plot = []

    for hist in prob_sta:
        # Si no hay historial para ese sigma, guardar NaN
        if len(hist) == 0:
            p0.append(np.nan); p1.append(np.nan); p2.append(np.nan); sigmas_plot.append(np.nan)
            continue

        # Sigma asociado a esa corrida (constante en esa historia)
        sigmas_plot.append(float(hist[-1][1]))

        # Tomar la cola temporal para aproximar estado cuasi-estacionario
        #n = len(hist)
        #k0 = max(0, int((1.0 - frac_tail) * n))
        tail = hist[:]

        vals0, vals1, vals2 = [], [], []
        for _, _, probs in tail:
            vals0.append(float(probs.get(0, 0.0)))
            vals1.append(float(probs.get(1, 0.0)))
            vals2.append(float(probs.get(2, 0.0)))

        # Promedio por sigma
        p0.append(np.mean(vals0) if len(vals0) else np.nan)
        p1.append(np.mean(vals1) if len(vals1) else np.nan)
        p2.append(np.mean(vals2) if len(vals2) else np.nan)

    return np.array(sigmas_plot), np.array(p0), np.array(p1), np.array(p2)


def plot_probs_vs_sigma(sigmas_plot, p0, p1, p2, path=None):
    # Ordenar por sigma por seguridad (por si llegan desordenados)
    idx = np.argsort(sigmas_plot)
    sigmas_plot = sigmas_plot[idx]
    p0 = p0[idx]
    p1 = p1[idx]
    p2 = p2[idx]

    # ============================================
    # Gráfica
    # ============================================
    fig, ax = plt.subplots()
    plt.gca().set_facecolor("#e9e9e9")

    # Curvas de probabilidad extraídas desde prob_sta
    plt.plot(sigmas_plot, p0, color="#0072BD", linewidth=2.5, label='adatom (i = 0)')
    plt.plot(sigmas_plot, p1, color="#D95319", linewidth=2.5, label='step (i = 1)')
    plt.plot(sigmas_plot, p2, color="#EDB120", linewidth=2.5, label='kink (i = 2)')

    plt.xlim(1, 8)
    plt.ylim(0, 1)
    plt.xlabel("Sobresaturación (σ)")
    plt.ylabel("Probabilidad")
    plt.title("Probabilidades por sitio de adsorción")
    plt.legend()

    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True
    )
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.grid(False)
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


times = np.arange(0, 151, 1)
sigmas = np.arange(1, 8.2, 0.2)

#prob_sta, kmcs, snaps_list = compute_probabilities(sigmas, times, delta=4.18, size=(75, 75), mode="flat", n_seeds=126)
#sigmas_plot, p0, p1, p2 = extraer_probs_desde_prob_sta(prob_sta)
#sigmas_plot, p0, p1, p2 = extract_probs(prob_sta)
#plot_probs_vs_sigma(sigmas_plot, p0, p1, p2, path = "probabilidades_by_site.png")
#plot_probs_vs_sigma(sigmas_plot, p0, p1, p2)