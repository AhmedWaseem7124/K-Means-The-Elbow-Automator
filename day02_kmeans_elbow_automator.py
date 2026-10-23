"""
================================================================================
  Day 02 - K-Means++ Clustering From Scratch & Mathematical Elbow Automator
================================================================================
  Stack  : NumPy . Matplotlib . scikit-learn (make_blobs + silhouette_score only)
  Output : day02_kmeans_artifact.png  (2x2 dark-themed dashboard)
================================================================================
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Python 3.7+ – safe on Windows
except AttributeError:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL AESTHETICS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
GRID_COLOR   = "#21262d"
TEXT_COLOR   = "#e6edf3"
ACCENT       = "#58a6ff"
ACCENT2      = "#f78166"
ACCENT3      = "#56d364"
ACCENT4      = "#d2a8ff"

PALETTE = [
    "#58a6ff", "#f78166", "#56d364", "#d2a8ff",
    "#ffa657", "#79c0ff", "#ff7b72", "#3fb950",
    "#a5d6ff", "#ffdcd7",
]

plt.rcParams.update({
    "figure.facecolor"  : DARK_BG,
    "axes.facecolor"    : PANEL_BG,
    "axes.edgecolor"    : GRID_COLOR,
    "axes.labelcolor"   : TEXT_COLOR,
    "axes.titlecolor"   : TEXT_COLOR,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "grid.color"        : GRID_COLOR,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "text.color"        : TEXT_COLOR,
    "font.family"       : "DejaVu Sans",
    "legend.facecolor"  : PANEL_BG,
    "legend.edgecolor"  : GRID_COLOR,
    "legend.labelcolor" : TEXT_COLOR,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
K_TRUE      = 5
N_SAMPLES   = 6000
K_SWEEP_MAX = 10

rng = np.random.default_rng(SEED)

X, y_true = make_blobs(
    n_samples    = N_SAMPLES,
    centers      = K_TRUE,
    cluster_std  = [0.8, 1.2, 0.6, 1.5, 0.9],   # heterogeneous variances
    random_state = SEED,
)

print("=" * 62)
print("  Day 02 - K-Means++ From Scratch & Elbow Automator")
print("=" * 62)
print(f"  Data : {N_SAMPLES} points | {K_TRUE} true clusters | seed={SEED}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FROM-SCRATCH K-MEANS++ ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class KMeansScratch:
    """
    Vectorised K-Means with K-Means++ initialisation.

    Parameters
    ----------
    k            : number of clusters
    max_iter     : EM iteration cap
    tol          : centroid-shift convergence tolerance
    init         : 'kmeans++'  or  'random'
    random_state : integer seed
    """

    def __init__(self, k: int, max_iter: int = 300,
                 tol: float = 1e-4, init: str = "kmeans++",
                 random_state: int = 42):
        self.k            = k
        self.max_iter     = max_iter
        self.tol          = tol
        self.init         = init
        self.random_state = random_state

        # populated after fit()
        self.centroids_        = None   # final centroids  (k, d)
        self.labels_           = None   # cluster index per point (n,)
        self.inertia_          = None   # WCSS
        self.centroid_history_ = None   # list of (k, d) arrays per iteration
        self.n_iter_           = 0

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Vectorised squared Euclidean distances.
        Returns shape (n_samples, n_centroids).
        Uses ||x - c||² = ||x||² - 2 x·c + ||c||²  for efficiency.
        """
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)   # (n, 1)
        C_sq = np.sum(C ** 2, axis=1, keepdims=True).T  # (1, k)
        cross = X @ C.T                                  # (n, k)
        return np.maximum(X_sq - 2 * cross + C_sq, 0.0)

    # ── initialisation ────────────────────────────────────────────────────────
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        rng_local = np.random.default_rng(self.random_state)
        n, d      = X.shape

        if self.init == "random":
            idx = rng_local.choice(n, size=self.k, replace=False)
            return X[idx].copy()

        # ── K-Means++ ──────────────────────────────────────────────────────
        # Step 1: choose first centroid uniformly at random
        centroids = [X[rng_local.integers(0, n)].copy()]

        for _ in range(1, self.k):
            # D²(x): squared distance to nearest existing centroid
            sq_d = self._sq_dists(X, np.array(centroids))   # (n, len(centroids))
            min_sq_d = sq_d.min(axis=1)                      # (n,)

            # Avoid division-by-zero when a point coincides with a centroid
            probs = min_sq_d / (min_sq_d.sum() + 1e-12)
            chosen_idx = rng_local.choice(n, p=probs)
            centroids.append(X[chosen_idx].copy())

        return np.array(centroids)   # (k, d)

    # ── EM loop ───────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> "KMeansScratch":
        centroids = self._init_centroids(X)
        history   = [centroids.copy()]

        for iteration in range(1, self.max_iter + 1):
            # ── E-step: assign each point to the nearest centroid ──────────
            sq_d  = self._sq_dists(X, centroids)   # (n, k)
            labels = sq_d.argmin(axis=1)            # (n,)

            # ── M-step: recompute centroids as cluster means ───────────────
            new_centroids = np.array([
                X[labels == ki].mean(axis=0) if np.any(labels == ki)
                else centroids[ki]            # keep old if cluster is empty
                for ki in range(self.k)
            ])

            history.append(new_centroids.copy())
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids

            if shift < self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.centroids_        = centroids
        self.labels_           = sq_d.argmin(axis=1)   # re-assign with final
        self.centroid_history_ = history

        # WCSS / Inertia
        self.inertia_ = float(
            sum(
                np.sum((X[self.labels_ == ki] - centroids[ki]) ** 2)
                for ki in range(self.k)
            )
        )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MATHEMATICAL ELBOW AUTOMATOR  (Kneedle-style)
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_k_elbow(k_values: np.ndarray,
                         wcss_values: np.ndarray
                         ) -> tuple[int, np.ndarray]:
    """
    Detect the elbow of the WCSS curve via maximum perpendicular distance
    from the secant line connecting (K_min, WCSS_max) → (K_max, WCSS_min).

    Returns
    -------
    k_star         : int   — the optimal cluster count
    perp_distances : array — perpendicular distance for each K
    """
    k_vals   = np.asarray(k_values,  dtype=float)
    wcss     = np.asarray(wcss_values, dtype=float)

    # Secant-line endpoints
    P1 = np.array([k_vals[0],  wcss[0]])    # (K_min, WCSS_max)
    P2 = np.array([k_vals[-1], wcss[-1]])   # (K_max, WCSS_min)

    # Direction vector of the secant line (unit vector)
    line_vec  = P2 - P1
    line_len  = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len

    perp_distances = np.zeros(len(k_vals))
    for i, (k, w) in enumerate(zip(k_vals, wcss)):
        point_vec       = np.array([k, w]) - P1
        # Scalar projection onto the secant
        proj_len        = np.dot(point_vec, line_unit)
        proj_point      = P1 + proj_len * line_unit
        # Perpendicular distance = |point - projection|
        perp_distances[i] = np.linalg.norm(np.array([k, w]) - proj_point)

    k_star = int(k_vals[np.argmax(perp_distances)])
    return k_star, perp_distances


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SWEEP  K ∈ {1 … 10}  — WCSS  +  Silhouette
# ─────────────────────────────────────────────────────────────────────────────
k_range      = np.arange(1, K_SWEEP_MAX + 1)
wcss_list    = []
sil_list     = []         # silhouette only for k ≥ 2
fitted_models: dict[int, KMeansScratch] = {}

print(f"  Sweeping K in {{1 ... {K_SWEEP_MAX}}} ...\n")

for k in k_range:
    model = KMeansScratch(k=k, init="kmeans++", random_state=SEED)
    model.fit(X)
    fitted_models[k] = model
    wcss_list.append(model.inertia_)

    if k >= 2:
        sil = silhouette_score(X, model.labels_)
        sil_list.append(sil)
    else:
        sil_list.append(np.nan)

wcss_arr = np.array(wcss_list)
sil_arr  = np.array(sil_list)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ELBOW DETECTION
# ─────────────────────────────────────────────────────────────────────────────
K_STAR, perp_dists = find_optimal_k_elbow(k_range, wcss_arr)
best_model         = fitted_models[K_STAR]

# Also run random-init at K* for convergence comparison (Panel 4)
model_random = KMeansScratch(k=K_STAR, init="random", random_state=SEED)
model_random.fit(X)

# Track per-iteration WCSS for both models
def wcss_history(X: np.ndarray, history: list, labels_final: np.ndarray) -> list:
    """Recompute WCSS at each saved centroid snapshot."""
    wcss_iters = []
    for centroids in history:
        sq_d   = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = sq_d.argmin(axis=1)
        w      = sum(
            np.sum((X[labels == ki] - centroids[ki]) ** 2)
            for ki in range(centroids.shape[0])
        )
        wcss_iters.append(w)
    return wcss_iters

wcss_kpp    = wcss_history(X, best_model.centroid_history_,    best_model.labels_)
wcss_rand   = wcss_history(X, model_random.centroid_history_,  model_random.labels_)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TERMINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
col_w = [6, 14, 22, 18]
header = (
    f"{'K':^{col_w[0]}}|"
    f"{'WCSS':^{col_w[1]}}|"
    f"{'Perp. Elbow Dist.':^{col_w[2]}}|"
    f"{'Silhouette':^{col_w[3]}}"
)
sep = "-" * (sum(col_w) + 3)

print(sep)
print(header)
print(sep)
for i, k in enumerate(k_range):
    sil_str = f"{sil_arr[i]:.6f}" if not np.isnan(sil_arr[i]) else "   --  "
    star    = " <-- K*" if k == K_STAR else ""
    print(
        f"{k:^{col_w[0]}}|"
        f"{wcss_arr[i]:^{col_w[1]}.2f}|"
        f"{perp_dists[i]:^{col_w[2]}.6f}|"
        f"{sil_str:^{col_w[3]}}"
        f"{star}"
    )
print(sep)
print(f"\n  [*] Elbow Automator selected  K* = {K_STAR}")
print(f"  [*] Best Silhouette @ K={np.nanargmax(sil_arr)+1}  "
      f"({np.nanmax(sil_arr):.4f})\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION — 2×2 DARK DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2,
    figsize      = (18, 14),
    facecolor    = DARK_BG,
    gridspec_kw  = {"hspace": 0.38, "wspace": 0.30},
)
fig.suptitle(
    "K-Means++ From Scratch  ·  Mathematical Elbow Automator",
    fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.97,
)

ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.45, linewidth=0.7)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


# ── Panel 1: Clustered Scatter + Voronoi mesh + Centroid migration ────────────
def panel1_cluster_map(ax, model: KMeansScratch, X: np.ndarray):
    ax.set_title(
        f"Panel 1 — Final Clustering  (K* = {model.k}, K-Means++)",
        fontsize=12, fontweight="bold", pad=10,
    )

    k     = model.k
    clrs  = PALETTE[:k]
    cmap  = ListedColormap(clrs)

    # Voronoi decision-boundary background mesh
    margin    = 0.8
    res       = 400
    x_lo, x_hi = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_lo, y_hi = X[:, 1].min() - margin, X[:, 1].max() + margin
    gx, gy    = np.meshgrid(
        np.linspace(x_lo, x_hi, res),
        np.linspace(y_lo, y_hi, res),
    )
    grid_pts  = np.c_[gx.ravel(), gy.ravel()]
    sq_d_grid = np.sum(
        (grid_pts[:, None, :] - model.centroids_[None, :, :]) ** 2, axis=2
    )
    Z = sq_d_grid.argmin(axis=1).reshape(gx.shape)

    ax.contourf(gx, gy, Z, levels=k - 1, cmap=cmap, alpha=0.12)
    ax.contour( gx, gy, Z, levels=k - 1, colors=GRID_COLOR, linewidths=0.6, alpha=0.5)

    # Scatter — cluster-coloured points
    for ki in range(k):
        mask = model.labels_ == ki
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=18, color=clrs[ki], alpha=0.75, linewidths=0,
            label=f"Cluster {ki}",
        )

    # Centroid migration arrows (initial → final)
    init_centroids  = model.centroid_history_[0]
    final_centroids = model.centroids_

    for ki in range(k):
        xs = [h[ki, 0] for h in model.centroid_history_]
        ys = [h[ki, 1] for h in model.centroid_history_]
        ax.plot(xs, ys, color=clrs[ki], lw=1.0, alpha=0.5, linestyle=":")

        ax.annotate(
            "",
            xy     = (final_centroids[ki, 0], final_centroids[ki, 1]),
            xytext = (init_centroids[ki, 0],  init_centroids[ki, 1]),
            arrowprops=dict(
                arrowstyle="->", color=clrs[ki],
                lw=1.8, mutation_scale=14,
            ),
        )
        # Initial seed marker
        ax.scatter(
            init_centroids[ki, 0], init_centroids[ki, 1],
            s=80, marker="x", color=clrs[ki], linewidths=1.8, zorder=5,
        )
        # Final centroid marker
        ax.scatter(
            final_centroids[ki, 0], final_centroids[ki, 1],
            s=200, marker="*", color="white",
            edgecolors=clrs[ki], linewidths=1.5, zorder=6,
        )

    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)

    legend_handles = [
        mpatches.Patch(color=clrs[ki], label=f"Cluster {ki}") for ki in range(k)
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="x", color="w",
                   markerfacecolor="w", markersize=7, label="Initial seed"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="w", markersize=10, label="Final centroid"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper left",
              framealpha=0.6, ncol=2)


panel1_cluster_map(ax1, best_model, X)


# ── Panel 2: Automated Elbow Curve ────────────────────────────────────────────
def panel2_elbow(ax, k_range, wcss_arr, perp_dists, K_STAR):
    ax.set_title("Panel 2 — Automated Elbow Curve  (Kneedle Method)",
                 fontsize=12, fontweight="bold", pad=10)

    # Main WCSS line
    ax.plot(k_range, wcss_arr, color=ACCENT, lw=2.5, marker="o",
            markersize=7, zorder=4, label="WCSS")

    # Secant line P1→P2
    ax.plot(
        [k_range[0], k_range[-1]],
        [wcss_arr[0], wcss_arr[-1]],
        color=ACCENT2, lw=1.4, linestyle="--", alpha=0.8, label="Secant line",
        zorder=3,
    )

    # Secant unit vector
    P1       = np.array([k_range[0],  wcss_arr[0]], dtype=float)
    P2       = np.array([k_range[-1], wcss_arr[-1]], dtype=float)
    line_vec = P2 - P1
    line_unit = line_vec / np.linalg.norm(line_vec)

    # Draw perpendicular distance vectors
    for i, (k, w) in enumerate(zip(k_range, wcss_arr)):
        pt        = np.array([k, w], dtype=float)
        proj_len  = np.dot(pt - P1, line_unit)
        proj_pt   = P1 + proj_len * line_unit
        # Only draw if distance is meaningful
        if perp_dists[i] > 1e-3:
            ax.annotate(
                "",
                xy     = tuple(proj_pt),
                xytext = (k, w),
                arrowprops=dict(
                    arrowstyle="-",
                    color=ACCENT3 if k == K_STAR else GRID_COLOR,
                    lw=1.6 if k == K_STAR else 0.8,
                    alpha=1.0 if k == K_STAR else 0.5,
                ),
                zorder=2,
            )

    # Highlight K*
    k_star_idx  = K_STAR - k_range[0]
    ax.scatter([K_STAR], [wcss_arr[k_star_idx]],
               s=200, color=ACCENT3, zorder=6, label=f"K* = {K_STAR}")
    ax.axvline(K_STAR, color=ACCENT3, lw=1.0, linestyle=":", alpha=0.6)

    ax.set_xlabel("Number of Clusters  K", fontsize=10)
    ax.set_ylabel("WCSS  (Inertia)", fontsize=10)
    ax.set_xticks(k_range)
    ax.legend(fontsize=9, framealpha=0.6)
    ax.annotate(
        f"K* = {K_STAR}\n(max ⊥ dist)",
        xy=(K_STAR, wcss_arr[k_star_idx]),
        xytext=(K_STAR + 0.5, wcss_arr[k_star_idx] + (wcss_arr.max() - wcss_arr.min()) * 0.07),
        color=ACCENT3, fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1.2),
    )


panel2_elbow(ax2, k_range, wcss_arr, perp_dists, K_STAR)


# ── Panel 3: Mean Silhouette Score across K ───────────────────────────────────
def panel3_silhouette(ax, k_range, sil_arr, K_STAR):
    ax.set_title("Panel 3 — Silhouette Score  (Cross-Validation)",
                 fontsize=12, fontweight="bold", pad=10)

    k_valid  = k_range[k_range >= 2]
    sil_valid = sil_arr[k_range >= 2]

    bars = ax.bar(
        k_valid, sil_valid,
        color=[ACCENT3 if k == K_STAR else ACCENT for k in k_valid],
        alpha=0.82, width=0.6, zorder=3,
        edgecolor=PANEL_BG, linewidth=0.5,
    )

    # Best silhouette marker
    best_sil_k   = k_valid[np.argmax(sil_valid)]
    best_sil_val = np.max(sil_valid)
    ax.axhline(best_sil_val, color=ACCENT4, lw=1.0, linestyle="--", alpha=0.6,
               label=f"Peak @ K={best_sil_k}")

    # Value labels on bars
    for bar, val in zip(bars, sil_valid):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, color=TEXT_COLOR,
        )

    # Highlight K*
    ax.axvline(K_STAR, color=ACCENT3, lw=1.2, linestyle=":", alpha=0.8,
               label=f"Elbow K* = {K_STAR}")

    ax.set_xlabel("Number of Clusters  K", fontsize=10)
    ax.set_ylabel("Mean Silhouette Score", fontsize=10)
    ax.set_xticks(k_valid)
    ax.set_ylim(0, min(1.0, sil_valid.max() * 1.2))
    ax.legend(fontsize=9, framealpha=0.6)


panel3_silhouette(ax3, k_range, sil_arr, K_STAR)


# ── Panel 4: Convergence — Random Init vs K-Means++ at K* ────────────────────
def panel4_convergence(ax, wcss_kpp, wcss_rand, K_STAR):
    ax.set_title(
        f"Panel 4 — Convergence Comparison  (K = {K_STAR})",
        fontsize=12, fontweight="bold", pad=10,
    )

    iters_kpp  = np.arange(len(wcss_kpp))
    iters_rand = np.arange(len(wcss_rand))

    ax.plot(iters_kpp,  wcss_kpp,  color=ACCENT,  lw=2.3,
            marker="o", markersize=5, label="K-Means++ init", zorder=4)
    ax.plot(iters_rand, wcss_rand, color=ACCENT2, lw=2.3,
            marker="s", markersize=5, label="Random init",    zorder=3,
            linestyle="--")

    # Annotate final WCSS values
    ax.annotate(
        f"Final: {wcss_kpp[-1]:.1f}",
        xy     = (len(wcss_kpp) - 1,  wcss_kpp[-1]),
        xytext = (len(wcss_kpp) - 1 + 0.3, wcss_kpp[-1] * 1.02),
        color  = ACCENT, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0),
    )
    ax.annotate(
        f"Final: {wcss_rand[-1]:.1f}",
        xy     = (len(wcss_rand) - 1,  wcss_rand[-1]),
        xytext = (len(wcss_rand) - 1 + 0.3, wcss_rand[-1] * 1.02),
        color  = ACCENT2, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.0),
    )

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("WCSS  (Inertia)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.6)

    note = (
        f"K-Means++ converged in {len(wcss_kpp)-1} iters\n"
        f"Random init converged in {len(wcss_rand)-1} iters"
    )
    ax.text(
        0.97, 0.97, note,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG,
                  edgecolor=GRID_COLOR, alpha=0.8),
    )


panel4_convergence(ax4, wcss_kpp, wcss_rand, K_STAR)


# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "K-Means++ from scratch · Kneedle Elbow Automator · "
    "scikit-learn used only for make_blobs & silhouette_score",
    ha="center", fontsize=8.5, color="#8b949e", style="italic",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "day02_kmeans_artifact.png"
plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()

print(f"  [*] Artifact saved -> {OUTPUT_PATH}")
print("=" * 62)
