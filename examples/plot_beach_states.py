"""
Illustrates the possible states of a cross-shore beach profile relevant to
the BeachFX-CSHORE pipeline:

  1. Initial / pre-storm
  2. Post-storm    (CSHORE output — eroded berm, scarped dune face, offshore bar)
  3. Post-recovery (beach_recover — exponential return toward equilibrium)
  4. Background erosion (beach_translate — seaward shift of berm zone)
  5. Post-nourishment  (raised berm and dune restored to design template)

Run:
    uv run examples/plot_beach_states.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── synthetic profile (feet, Beach-FX seaward-positive → x increases offshore) ─
def make_initial_profile():
    """Idealised profile: upland → dune → berm → foreshore → nearshore."""
    segments = [
        # x_start  x_end   z_start  z_end    description
        (0,        200,    6.0,     6.0),    # upland
        (200,      260,    6.0,     10.0),   # landward dune face
        (260,      360,    10.0,    10.0),   # dune crest
        (360,      420,    10.0,    5.5),    # seaward dune face
        (420,      700,    5.5,     5.5),    # berm (dry beach)
        (700,      800,    5.5,     0.0),    # foreshore
        (800,      1400,   0.0,     -12.0),  # nearshore slope
        (1400,     2000,   -12.0,   -15.0),  # offshore
    ]
    pts = []
    for x0, x1, z0, z1 in segments:
        n = max(3, int((x1 - x0) / 5))
        xs = np.linspace(x0, x1, n)
        zs = np.linspace(z0, z1, n)
        pts.append(np.column_stack([xs, zs]))
    xy = np.vstack(pts)
    # deduplicate x
    _, idx = np.unique(xy[:, 0], return_index=True)
    return xy[idx, 0], xy[idx, 1]


def storm_profile(x, z):
    """Post-storm: eroded berm, scarped dune face, offshore bar deposit."""
    zs = z.copy()
    # berm erosion: lower berm by ~2 ft, shoreline retreats ~80 ft
    berm_mask = (x >= 420) & (x <= 750)
    zs[berm_mask] -= np.interp(x[berm_mask], [420, 750], [3.0, 1.5])
    # dune scarping: chop seaward dune face
    dune_mask = (x >= 340) & (x <= 430)
    zs[dune_mask] -= np.interp(x[dune_mask], [340, 430], [0.5, 2.5])
    # offshore bar: small accretion in nearshore
    bar_mask = (x >= 900) & (x <= 1100)
    zs[bar_mask] += 1.5 * np.exp(-((x[bar_mask] - 1000) ** 2) / (2 * 60 ** 2))
    return zs


def recovery_profile(z_initial, z_storm, x, z_berm=5.6):
    """Partial recovery: 50% of the way back toward initial, below berm only."""
    zr = z_storm.copy()
    below_berm = z_initial <= z_berm
    zr[below_berm] = z_storm[below_berm] + 0.5 * (z_initial[below_berm] - z_storm[below_berm])
    return zr


def background_erosion_profile(x, z, dX=60, z_berm=5.6):
    """Seaward translation of the berm zone by dX feet."""
    zb = z.copy()
    below_berm = z <= z_berm
    x_shifted = x.copy().astype(float)
    x_shifted[below_berm] += dX
    zb_shifted = np.interp(x, x_shifted, z, left=z[0], right=z[-1])
    wt = (z <= z_berm).astype(float)
    return (1 - wt) * z + wt * zb_shifted


def nourishment_profile(x, z):
    """Post-nourishment: restore berm + dune to slightly above initial design."""
    zn = z.copy()
    berm_mask = (x >= 350) & (x <= 750)
    zn[berm_mask] = np.maximum(zn[berm_mask],
                               np.interp(x[berm_mask], [350, 450, 700, 750],
                                         [9.5, 5.8, 5.8, 2.0]))
    return zn


# ── plot ──────────────────────────────────────────────────────────────────────
x, z_init = make_initial_profile()
z_storm    = storm_profile(x, z_init)
z_recover  = recovery_profile(z_init, z_storm, x)
z_bg       = background_erosion_profile(x, z_init)
z_nourish  = nourishment_profile(x, z_storm)

fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
fig.suptitle("Beach Profile States — BeachFX-CSHORE Pipeline", fontsize=13, y=0.98)

XLIM = (0, 1500)
YLIM = (-18, 14)
SEA  = "#cce5ff"
LAND = "#f5e8c0"

def fill_profile(ax, x, z, color, alpha=0.25):
    ax.fill_between(x, -20, z, color=color, alpha=alpha, zorder=1)

def base_plot(ax, x, z_ref, z_show, label, color, lw=1.8, ls="-"):
    fill_profile(ax, x, z_ref, LAND)
    ax.fill_between(x, -20, 0, color=SEA, alpha=0.35, zorder=1)
    ax.plot(x, z_ref, color="lightgray", lw=1, ls="--", zorder=2, label="Initial")
    ax.plot(x, z_show, color=color, lw=lw, ls=ls, zorder=3, label=label)
    ax.axhline(0, color="#4a90d9", lw=0.8, ls="-", alpha=0.7)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_ylabel("Elevation (ft)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

def annotate(ax, annotations):
    for x_a, y_a, txt, ha in annotations:
        ax.annotate(txt, xy=(x_a, y_a), fontsize=7.5, ha=ha, va="bottom",
                    color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7),
                    xytext=(x_a, y_a + 1.2))

# ── Panel 1: Initial profile ──────────────────────────────────────────────────
ax = axes[0, 0]
fill_profile(ax, x, z_init, LAND)
ax.fill_between(x, -20, 0, color=SEA, alpha=0.35, zorder=1)
ax.plot(x, z_init, "k", lw=2, zorder=3, label="Initial profile")
ax.axhline(0, color="#4a90d9", lw=0.8, alpha=0.7)
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_ylabel("Elevation (ft)")
ax.set_title("① Initial / pre-storm")
ax.grid(True, alpha=0.25)
# labels
for xp, yp, txt, dx in [
    (330, 10.0, "Dune\ncrest", 0), (230, 8.5, "Dune\nface", 0),
    (560, 5.5, "Berm", 0), (750, 3.5, "Foreshore", 0),
    (1000, -6, "Nearshore", 0), (80, 6.0, "Upland", 0)]:
    ax.text(xp, yp + 0.6, txt, fontsize=7.5, ha="center", color="#333")
ax.annotate("", xy=(700, 0), xytext=(700, -1),
            arrowprops=dict(arrowstyle="->", color="#4a90d9", lw=1))
ax.text(700, -2.5, "SWL = 0", fontsize=7.5, ha="center", color="#4a90d9")

# ── Panel 2: Post-storm ───────────────────────────────────────────────────────
ax = axes[0, 1]
base_plot(ax, x, z_init, z_storm, "Post-storm (CSHORE)", "#d62728")
ax.set_title("② Post-storm  (CSHORE output)")
dz = z_storm - z_init
ax.fill_between(x, z_init, z_storm, where=dz < 0, color="#d62728", alpha=0.2,
                label="Erosion")
ax.fill_between(x, z_init, z_storm, where=dz > 0, color="#2ca02c", alpha=0.2,
                label="Accretion (bar)")
ax.legend(fontsize=7.5, loc="upper right")
for xp, yp, txt in [(540, 3.5, "Berm\neroded"), (380, 7.0, "Dune\nscarped"),
                     (1000, -10, "Offshore\nbar")]:
    ax.text(xp, yp, txt, fontsize=7.5, ha="center", color="#d62728",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

# ── Panel 3: Recovery ─────────────────────────────────────────────────────────
ax = axes[1, 0]
base_plot(ax, x, z_storm, z_recover, "Post-recovery (~50%)", "#2ca02c")
ax.plot(x, z_init, color="gray", lw=0.8, ls=":", zorder=2, label="Equilibrium target")
ax.set_title("③ Post-storm recovery  (beach_recover)")
ax.set_ylabel("Elevation (ft)")
ax.legend(fontsize=7.5, loc="upper right")
ax.annotate("Berm rebuilds\n(below z_berm only)",
            xy=(560, z_recover[(np.abs(x - 560)).argmin()]),
            xytext=(400, 3.5), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
            color="#2ca02c")
ax.annotate("Dune unchanged",
            xy=(310, z_recover[(np.abs(x - 310)).argmin()]),
            xytext=(150, 8.0), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            color="gray")

# ── Panel 4: Background erosion ───────────────────────────────────────────────
ax = axes[1, 1]
base_plot(ax, x, z_init, z_bg, "After background erosion", "#ff7f0e", ls="-")
ax.set_title("④ Background erosion  (beach_translate)")
ax.set_ylabel("Elevation (ft)")
dz_bg = z_bg - z_init
ax.fill_between(x, z_init, z_bg, where=dz_bg < -0.05, color="#ff7f0e", alpha=0.2)
ax.annotate("Profile shifts seaward\n(dXdt × T, below berm)",
            xy=(700, z_bg[(np.abs(x - 700)).argmin()]),
            xytext=(500, -5), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8),
            color="#ff7f0e")
ax.annotate("Dune / upland\nunchanged",
            xy=(310, z_bg[(np.abs(x - 310)).argmin()]),
            xytext=(150, 8.0), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            color="gray")

# ── Panel 5: Nourishment ──────────────────────────────────────────────────────
ax = axes[2, 0]
base_plot(ax, x, z_storm, z_nourish, "Post-nourishment", "#1f77b4")
ax.set_title("⑤ Post-nourishment  (profile raise)")
ax.set_xlabel("Cross-shore distance (ft)")
ax.set_ylabel("Elevation (ft)")
ax.fill_between(x, z_storm, z_nourish, where=z_nourish > z_storm,
                color="#1f77b4", alpha=0.25, label="Added sand")
ax.legend(fontsize=7.5, loc="upper right")
ax.annotate("Sand placed on\nberm and dune",
            xy=(500, z_nourish[(np.abs(x - 500)).argmin()]),
            xytext=(300, 2.5), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
            color="#1f77b4")

# ── Panel 6: Full lifecycle sequence ─────────────────────────────────────────
ax = axes[2, 1]
ax.fill_between(x, -20, 0, color=SEA, alpha=0.35, zorder=1)
profiles = [
    (z_init,    "k",       "Initial",             2.0, "-"),
    (z_storm,   "#d62728", "Post-storm",          1.5, "-"),
    (z_recover, "#2ca02c", "Post-recovery",       1.5, "--"),
    (z_bg,      "#ff7f0e", "Background erosion",  1.2, ":"),
    (z_nourish, "#1f77b4", "Post-nourishment",    1.5, "-."),
]
for z_p, col, lbl, lw, ls in profiles:
    ax.plot(x, z_p, color=col, lw=lw, ls=ls, label=lbl, zorder=3)
ax.fill_between(x, -20, z_init, color=LAND, alpha=0.15, zorder=1)
ax.axhline(0, color="#4a90d9", lw=0.8, alpha=0.7)
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_title("All states overlaid")
ax.set_xlabel("Cross-shore distance (ft)")
ax.legend(fontsize=7.5, loc="lower left", ncol=1)
ax.grid(True, alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "beach_profile_states.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved → {os.path.abspath(out_path)}")
