# =============================================================
#  FightVision — JOUR 2 : Radar Chart + Octogone 3D
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)
df = df[df['Winner'].isin(['Red', 'Blue'])].copy()

# ─────────────────────────────────────────────────────────────
# HELPER : profil fighter
# ─────────────────────────────────────────────────────────────
def get_fighter_profile(fighter_name):
    """Cherche dans Red ou Blue corner."""
    for corner in ['R', 'B']:
        mask = df[f'{corner}_fighter'] == fighter_name
        sub  = df[mask]
        if len(sub) > 0:
            p = corner
            return {
                'name'        : fighter_name,
                'wins'        : int(sub[f'{p}_wins'].iloc[0]),
                'losses'      : int(sub[f'{p}_losses'].iloc[0]),
                'KO_wins'     : int(sub[f'{p}_win_by_KO/TKO'].iloc[0]),
                'sub_wins'    : int(sub[f'{p}_win_by_Submission'].iloc[0]),
                'sig_str_pct' : float(sub[f'{p}_avg_SIG_STR_pct'].mean()),
                'td_pct'      : float(sub[f'{p}_avg_TD_pct'].mean()),
                'kd_avg'      : float(sub[f'{p}_avg_KD'].mean()),
                'ctrl_time'   : float(sub[f'{p}_avg_CTRL_time(seconds)'].mean()),
                'head_landed' : float(sub[f'{p}_avg_HEAD_landed'].mean()),
                'body_landed' : float(sub[f'{p}_avg_BODY_landed'].mean()),
                'leg_landed'  : float(sub[f'{p}_avg_LEG_landed'].mean()),
                'reach'       : float(sub[f'{p}_Reach_cms'].iloc[0]),
                'height'      : float(sub[f'{p}_Height_cms'].iloc[0]),
                'stance'      : str(sub[f'{p}_Stance'].iloc[0]),
                'win_streak'  : int(sub[f'{p}_current_win_streak'].iloc[0]),
                'corner'      : corner,
            }
    return None

# ─────────────────────────────────────────────────────────────
# 1. RADAR CHART
# ─────────────────────────────────────────────────────────────
def radar_chart(f1_name, f2_name, ax=None):
    """Compare 2 fighters sur 6 axes normalisés."""
    p1 = get_fighter_profile(f1_name)
    p2 = get_fighter_profile(f2_name)
    if not p1 or not p2:
        print(f"⚠️  Fighter introuvable : {f1_name} ou {f2_name}")
        return

    categories = ['Sig. Strike %', 'Takedown %', 'KD moyen',
                  'Ctrl Time', 'Frappes Tête', 'Frappes Corps']

    def normalize(val, min_v, max_v):
        if max_v == min_v: return 0.5
        return (val - min_v) / (max_v - min_v)

    # Valeurs brutes
    raw = {
        'sig_str_pct' : (0, 1),
        'td_pct'      : (0, 1),
        'kd_avg'      : (0, 3),
        'ctrl_time'   : (0, 200),
        'head_landed' : (0, 80),
        'body_landed' : (0, 25),
    }
    keys = list(raw.keys())

    v1 = [normalize(p1.get(k, 0) or 0, *raw[k]) for k in keys]
    v2 = [normalize(p2.get(k, 0) or 0, *raw[k]) for k in keys]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    v1 += v1[:1]
    v2 += v2[:1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.set_facecolor('#1a1a2e')
    ax.figure.patch.set_facecolor('#1a1a2e')

    # Grille
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25%', '50%', '75%'], color='grey', size=8)

    # Fighter 1 (Rouge)
    ax.plot(angles, v1, 'o-', linewidth=2, color='#e74c3c', label=f1_name)
    ax.fill(angles, v1, alpha=0.25, color='#e74c3c')

    # Fighter 2 (Bleu)
    ax.plot(angles, v2, 'o-', linewidth=2, color='#3498db', label=f2_name)
    ax.fill(angles, v2, alpha=0.25, color='#3498db')

    ax.set_title(f'{f1_name}  vs  {f2_name}', color='white', size=13,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              facecolor='#1a1a2e', labelcolor='white')
    return ax

# ─────────────────────────────────────────────────────────────
# 2. OCTOGONE 3D AVEC HEATMAP
# ─────────────────────────────────────────────────────────────
def build_octagon_vertices(r=1.0, z=0.0):
    """Retourne les 8 sommets d'un octogone."""
    angles = [np.pi/8 + i * np.pi/4 for i in range(8)]
    return [(r * np.cos(a), r * np.sin(a), z) for a in angles]

def draw_octagon_3d(ax, verts, color='#2c3e50', alpha=0.3, z_base=0):
    """Dessine le sol de l'octogone."""
    xs = [v[0] for v in verts] + [verts[0][0]]
    ys = [v[1] for v in verts] + [verts[0][1]]
    zs = [z_base] * (len(verts) + 1)
    ax.plot(xs, ys, zs, color='#f39c12', linewidth=2.5)

    # Sol rempli
    face = [[v[0], v[1], z_base] for v in verts]
    poly = Poly3DCollection([face], alpha=alpha, facecolor=color,
                             edgecolor='#f39c12', linewidth=1.5)
    ax.add_collection3d(poly)

def draw_cage_walls(ax, verts, height=0.8):
    """Dessine les murs de la cage."""
    n = len(verts)
    for i in range(n):
        x0, y0, _ = verts[i]
        x1, y1, _ = verts[(i+1) % n]
        # Lignes verticales
        ax.plot([x0, x0], [y0, y0], [0, height], color='#7f8c8d', alpha=0.4, linewidth=1)
        # Haut de la cage
        ax.plot([x0, x1], [y0, y1], [height, height], color='#7f8c8d', alpha=0.4, linewidth=1)

def heatmap_on_octagon(ax, p1, p2):
    """Projette des cercles de chaleur sur le sol selon les zones de frappe."""
    # Zones fictives sur le sol (Head centre, Body devant, Leg bas)
    zones = {
        'HEAD' : (0.0,  0.3, 0.3),
        'BODY' : (0.0,  0.0, 0.15),
        'LEG'  : (0.0, -0.3, 0.05),
    }
    for zone, (x, y, base_r) in zones.items():
        key = f'{zone.lower()}_landed'
        v1  = (p1.get(key) or 0)
        v2  = (p2.get(key) or 0)
        total = v1 + v2 + 0.001

        # Cercle Rouge (p1)
        theta = np.linspace(0, 2*np.pi, 60)
        r1 = base_r * (v1 / total) * 0.9 + 0.05
        ax.plot(x + r1*np.cos(theta), y + r1*np.sin(theta),
                np.zeros(60), color='#e74c3c', alpha=0.7, linewidth=2)

        # Cercle Bleu (p2) — décalé légèrement
        r2 = base_r * (v2 / total) * 0.9 + 0.05
        ax.plot(x + 0.15 + r2*np.cos(theta), y + r2*np.sin(theta),
                np.zeros(60), color='#3498db', alpha=0.7, linewidth=2)

        ax.text(x - 0.05, y, 0.02, zone, color='white', fontsize=8,
                ha='center', va='bottom')

def octagon_3d(f1_name, f2_name, proba_red=0.5, ax=None):
    """Visualisation 3D de l'octogone avec les 2 fighters."""
    p1 = get_fighter_profile(f1_name)
    p2 = get_fighter_profile(f2_name)

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('#0d0d1a')
    ax.figure.patch.set_facecolor('#0d0d1a')

    verts  = build_octagon_vertices(r=1.0)
    draw_octagon_3d(ax, verts, color='#1c2833', alpha=0.5)
    draw_cage_walls(ax, verts, height=0.7)

    # Heatmap zones
    if p1 and p2:
        heatmap_on_octagon(ax, p1, p2)

    # Silhouettes fighters (cônes simples)
    fighter_positions = [(-0.45, 0), (0.45, 0)]
    colors = ['#e74c3c', '#3498db']
    names  = [f1_name, f2_name]
    probas = [proba_red, 1 - proba_red]

    for (fx, fy), col, name, prob in zip(fighter_positions, colors, names, probas):
        # Corps (cylindre simplifié)
        theta = np.linspace(0, 2*np.pi, 30)
        r_body = 0.06
        z_body = np.linspace(0, 0.45, 10)
        for z in z_body:
            ax.plot(fx + r_body*np.cos(theta), fy + r_body*np.sin(theta),
                    [z]*30, color=col, alpha=0.4, linewidth=0.8)

        # Tête (sphère approximée)
        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        xs = fx + 0.08 * np.cos(u) * np.sin(v)
        ys = fy + 0.08 * np.sin(u) * np.sin(v)
        zs = 0.53 + 0.08 * np.cos(v)
        ax.plot_surface(xs, ys, zs, color=col, alpha=0.6)

        # Label + proba
        ax.text(fx, fy, 0.75, f"{name.split()[-1]}\n{prob*100:.0f}%",
                color=col, fontsize=9, fontweight='bold',
                ha='center', va='bottom')

    # UFC logo central
    ax.text(0, 0, 0.01, "UFC", color='#f39c12', fontsize=14,
            fontweight='bold', ha='center', va='bottom')

    # Titres et axes
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(0, 1.0)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.grid(False)
    ax.set_title(f"🥊 {f1_name}  vs  {f2_name}\nProba victoire : {proba_red*100:.0f}% vs {(1-proba_red)*100:.0f}%",
                 color='white', fontsize=11, pad=10)

    # Légende
    p1_patch = mpatches.Patch(color='#e74c3c', label=f'{f1_name} (Red corner)')
    p2_patch = mpatches.Patch(color='#3498db', label=f'{f2_name} (Blue corner)')
    ax.legend(handles=[p1_patch, p2_patch], loc='upper left',
              facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    return ax

# ─────────────────────────────────────────────────────────────
# 3. FIGURE COMPLÈTE : RADAR + OCTOGONE côte à côte
# ─────────────────────────────────────────────────────────────
def full_fight_viz(f1_name, f2_name, proba_red=None):
    """Figure 1×2 : radar à gauche, octogone 3D à droite."""
    # Estimation proba si non fournie
    if proba_red is None:
        p1 = get_fighter_profile(f1_name)
        p2 = get_fighter_profile(f2_name)
        if p1 and p2:
            s1 = (p1.get('wins', 0) or 0) + (p1.get('win_streak', 0) or 0)
            s2 = (p2.get('wins', 0) or 0) + (p2.get('win_streak', 0) or 0)
            proba_red = s1 / (s1 + s2 + 0.001)
        else:
            proba_red = 0.5

    fig = plt.figure(figsize=(16, 7), facecolor='#0d0d1a')
    fig.suptitle("⚡ FightVision — Analyse de Combat UFC ⚡",
                 color='#f39c12', fontsize=15, fontweight='bold')

    # Radar (polaire)
    ax1 = fig.add_subplot(121, polar=True)
    radar_chart(f1_name, f2_name, ax=ax1)

    # Octogone 3D
    ax2 = fig.add_subplot(122, projection='3d')
    octagon_3d(f1_name, f2_name, proba_red=proba_red, ax=ax2)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/fight_{f1_name.replace(' ','_')}_vs_{f2_name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
    plt.show()
    print(f"✅ Figure sauvegardée → {fname}")

# ─────────────────────────────────────────────────────────────
# MAIN — modifie les noms ici
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Exemples de fighters présents dans le dataset :
    FIGHTER_1 = "Conor McGregor"
    FIGHTER_2 = "Khabib Nurmagomedov"

    print(f"\n🥊 Analyse : {FIGHTER_1}  vs  {FIGHTER_2}")
    full_fight_viz(FIGHTER_1, FIGHTER_2)
    print("\n✅ JOUR 2 TERMINÉ — Lance maintenant : streamlit run jour3_app.py")
