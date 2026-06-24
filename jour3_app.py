# =============================================================
#  FightVision — JOUR 3 : Application Streamlit complète
#  Lancement : streamlit run jour3_app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Config page ───────────────────────────────────────────────
st.set_page_config(
    page_title="FightVision UFC",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS custom ────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0d0d1a; }
    .stApp { background-color: #0d0d1a; color: white; }
    h1, h2, h3 { color: #f39c12 !important; }
    .winner-box { background: linear-gradient(135deg, #e74c3c, #c0392b);
                  border-radius: 12px; padding: 20px; text-align: center;
                  color: white; font-size: 24px; font-weight: bold; margin: 10px 0;}
    .stat-box   { background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
                  padding: 12px; text-align: center; }
    .red-text   { color: #e74c3c; font-weight: bold; }
    .blue-text  { color: #3498db; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DATA & MODÈLE
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("outputs/model.pkl")
    except:
        return None

df  = load_data()
model = load_model()

# Liste des fighters
all_fighters = sorted(set(df['R_fighter'].tolist() + df['B_fighter'].tolist()))

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_profile(name):
    for corner in ['R', 'B']:
        mask = df[f'{corner}_fighter'] == name
        sub  = df[mask]
        if len(sub) > 0:
            p = corner
            return {
                'name'        : name,
                'wins'        : int(sub[f'{p}_wins'].iloc[0]),
                'losses'      : int(sub[f'{p}_losses'].iloc[0]),
                'KO_wins'     : int(sub[f'{p}_win_by_KO/TKO'].iloc[0]),
                'sub_wins'    : int(sub[f'{p}_win_by_Submission'].iloc[0]),
                'sig_str_pct' : float(sub[f'{p}_avg_SIG_STR_pct'].mean() or 0),
                'td_pct'      : float(sub[f'{p}_avg_TD_pct'].mean() or 0),
                'kd_avg'      : float(sub[f'{p}_avg_KD'].mean() or 0),
                'ctrl_time'   : float(sub[f'{p}_avg_CTRL_time(seconds)'].mean() or 0),
                'head_landed' : float(sub[f'{p}_avg_HEAD_landed'].mean() or 0),
                'body_landed' : float(sub[f'{p}_avg_BODY_landed'].mean() or 0),
                'leg_landed'  : float(sub[f'{p}_avg_LEG_landed'].mean() or 0),
                'reach'       : float(sub[f'{p}_Reach_cms'].iloc[0] or 0),
                'height'      : float(sub[f'{p}_Height_cms'].iloc[0] or 0),
                'weight'      : float(sub[f'{p}_Weight_lbs'].iloc[0] or 0),
                'stance'      : str(sub[f'{p}_Stance'].iloc[0]),
                'win_streak'  : int(sub[f'{p}_current_win_streak'].iloc[0]),
                'total_bouts' : int(sub[f'{p}_total_rounds_fought'].iloc[0]),
            }
    return None

def compute_features(p1, p2):
    """Reproduit les mêmes features que jour1_model.py."""
    def safe(v): return v if v == v else 0
    feats = {
        'reach_diff'      : safe(p1['reach'])      - safe(p2['reach']),
        'height_diff'     : safe(p1['height'])     - safe(p2['height']),
        'age_diff'        : 0,
        'wins_diff'       : safe(p1['wins'])        - safe(p2['wins']),
        'losses_diff'     : safe(p1['losses'])      - safe(p2['losses']),
        'win_streak_diff' : safe(p1['win_streak'])  - safe(p2['win_streak']),
        'lose_streak_diff': 0,
        'title_bouts_diff': 0,
        'KO_wins_diff'    : safe(p1['KO_wins'])     - safe(p2['KO_wins']),
        'sub_wins_diff'   : safe(p1['sub_wins'])    - safe(p2['sub_wins']),
        'sig_str_pct_diff': safe(p1['sig_str_pct']) - safe(p2['sig_str_pct']),
        'td_pct_diff'     : safe(p1['td_pct'])      - safe(p2['td_pct']),
        'kd_diff'         : safe(p1['kd_avg'])      - safe(p2['kd_avg']),
        'ctrl_time_diff'  : safe(p1['ctrl_time'])   - safe(p2['ctrl_time']),
        'sub_att_diff'    : 0,
        'R_head_acc'      : safe(p1['head_landed']) / (safe(p1['head_landed']) + 1),
        'R_body_acc'      : safe(p1['body_landed']) / (safe(p1['body_landed']) + 1),
        'R_leg_acc'       : safe(p1['leg_landed'])  / (safe(p1['leg_landed'])  + 1),
        'B_head_acc'      : safe(p2['head_landed']) / (safe(p2['head_landed']) + 1),
        'B_body_acc'      : safe(p2['body_landed']) / (safe(p2['body_landed']) + 1),
        'B_leg_acc'       : safe(p2['leg_landed'])  / (safe(p2['leg_landed'])  + 1),
        'R_stance'        : 0,
        'B_stance'        : 0,
        'title_bout'      : 0,
    }
    return pd.DataFrame([feats])

# ─────────────────────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────────────────────
def make_radar(p1, p2):
    categories = ['Sig Strike%', 'Takedown%', 'KD moyen',
                  'Ctrl Time', 'Frappes Tête', 'Frappes Corps']
    ranges = [(0,1), (0,1), (0,3), (0,200), (0,80), (0,25)]
    keys   = ['sig_str_pct', 'td_pct', 'kd_avg', 'ctrl_time', 'head_landed', 'body_landed']

    def norm(v, lo, hi):
        v = v if v == v else 0
        return max(0, min(1, (v - lo) / (hi - lo + 1e-9)))

    v1 = [norm(p1.get(k, 0), *r) for k, r in zip(keys, ranges)]
    v2 = [norm(p2.get(k, 0), *r) for k, r in zip(keys, ranges)]
    N  = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    v1 += v1[:1]; v2 += v2[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True),
                           facecolor='#0d0d1a')
    ax.set_facecolor('#1a1a2e')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25', '50', '75'], color='grey', size=7)

    ax.plot(angles, v1, 'o-', lw=2, color='#e74c3c', label=p1['name'])
    ax.fill(angles, v1, alpha=0.2, color='#e74c3c')
    ax.plot(angles, v2, 'o-', lw=2, color='#3498db', label=p2['name'])
    ax.fill(angles, v2, alpha=0.2, color='#3498db')

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15),
              facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    ax.set_title("Profil Comparatif", color='#f39c12', pad=15, fontweight='bold')
    return fig

# ─────────────────────────────────────────────────────────────
# OCTOGONE 3D
# ─────────────────────────────────────────────────────────────
def make_octagon(p1, p2, proba_red):
    fig = plt.figure(figsize=(6, 5.5), facecolor='#0d0d1a')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d1a')

    # Sol octogone
    r = 1.0
    angles_oct = [np.pi/8 + i * np.pi/4 for i in range(8)]
    verts = [(r*np.cos(a), r*np.sin(a), 0) for a in angles_oct]
    xs = [v[0] for v in verts] + [verts[0][0]]
    ys = [v[1] for v in verts] + [verts[0][1]]
    zs = [0] * 9
    ax.plot(xs, ys, zs, color='#f39c12', lw=2.5)

    face = [[v[0], v[1], 0] for v in verts]
    poly = Poly3DCollection([face], alpha=0.35, facecolor='#1c2833', edgecolor='#f39c12')
    ax.add_collection3d(poly)

    # Murs cage
    for i in range(len(verts)):
        x0, y0, _ = verts[i]; x1, y1, _ = verts[(i+1)%len(verts)]
        ax.plot([x0,x0],[y0,y0],[0,0.7], color='#7f8c8d', alpha=0.4, lw=1)
        ax.plot([x0,x1],[y0,y1],[0.7,0.7], color='#7f8c8d', alpha=0.4, lw=1)

    # Heatmap zones frappe
    for zone, (zx, zy, lbl) in {'HEAD':(0,0.3,'HEAD'), 'BODY':(0,0,'BODY'), 'LEG':(0,-0.3,'LEG')}.items():
        k = f'{zone.lower()}_landed'
        v1v = p1.get(k, 1) or 1; v2v = p2.get(k, 1) or 1
        tot  = v1v + v2v
        theta = np.linspace(0, 2*np.pi, 60)
        r1 = 0.08 + 0.12 * (v1v / tot)
        r2 = 0.08 + 0.12 * (v2v / tot)
        ax.plot(zx + r1*np.cos(theta), zy + r1*np.sin(theta), [0.01]*60,
                color='#e74c3c', alpha=0.6, lw=1.5)
        ax.plot(zx+0.2 + r2*np.cos(theta), zy + r2*np.sin(theta), [0.01]*60,
                color='#3498db', alpha=0.6, lw=1.5)
        ax.text(zx, zy, 0.025, lbl, color='white', fontsize=7, ha='center')

    # Silhouettes (cônes)
    for (fx, fy), col, name, prob in [
        (-0.45, 0, '#e74c3c', p1['name'], proba_red),
        (0.45,  0, '#3498db', p2['name'], 1-proba_red)
    ]:
        th = np.linspace(0, 2*np.pi, 20)
        for z in np.linspace(0.02, 0.42, 8):
            ax.plot(fx+0.05*np.cos(th), fy+0.05*np.sin(th), [z]*20, color=col, alpha=0.35, lw=0.7)
        u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
        ax.plot_surface(fx+0.07*np.cos(u)*np.sin(v), fy+0.07*np.sin(u)*np.sin(v),
                        0.5+0.07*np.cos(v), color=col, alpha=0.55)
        ax.text(fx, fy, 0.72, f"{name.split()[-1]}\n{prob*100:.0f}%",
                color=col, fontsize=8, fontweight='bold', ha='center')

    ax.text(0, 0, 0.02, "UFC", color='#f39c12', fontsize=12, fontweight='bold', ha='center')
    ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.4,1.4); ax.set_zlim(0,1)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.grid(False)
    ax.set_title("Octogone 3D — Zones de frappe", color='white', fontsize=10, pad=5)
    return fig

# ─────────────────────────────────────────────────────────────
# UI PRINCIPALE
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>⚡ FightVision UFC ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa'>Analyse prédictive et visualisation 3D des combats</p>",
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar sélection
with st.sidebar:
    st.markdown("### 🥊 Sélection des combattants")
    f1 = st.selectbox("🔴 Red Corner", all_fighters,
                      index=all_fighters.index("Conor McGregor") if "Conor McGregor" in all_fighters else 0)
    f2 = st.selectbox("🔵 Blue Corner", all_fighters,
                      index=all_fighters.index("Khabib Nurmagomedov") if "Khabib Nurmagomedov" in all_fighters else 1)
    analyze = st.button("⚡ ANALYSER LE COMBAT", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**ℹ️ Dataset**")
    st.markdown(f"- {len(df)} combats historiques")
    st.markdown(f"- {len(all_fighters)} fighters")
    if model:
        st.markdown("- ✅ Modèle ML chargé")
    else:
        st.markdown("- ⚠️ Lance jour1_model.py d'abord")

if analyze or True:  # Affiche directement
    if f1 == f2:
        st.error("Sélectionne deux fighters différents !")
        st.stop()

    p1 = get_profile(f1)
    p2 = get_profile(f2)

    if not p1 or not p2:
        st.error("Profil introuvable pour un des fighters.")
        st.stop()

    # ── Prédiction ML ──────────────────────────────────────────
    proba_red = 0.5
    if model:
        feats = compute_features(p1, p2)
        try:
            proba_red = model.predict_proba(feats)[0][1]
        except:
            s1 = p1['wins'] + p1['win_streak']
            s2 = p2['wins'] + p2['win_streak']
            proba_red = s1 / (s1 + s2 + 0.001)
    else:
        s1 = p1['wins'] + p1['win_streak']
        s2 = p2['wins'] + p2['win_streak']
        proba_red = s1 / (s1 + s2 + 0.001)

    predicted_winner = f1 if proba_red >= 0.5 else f2
    winner_color     = "#e74c3c" if proba_red >= 0.5 else "#3498db"

    # ── Bloc prédiction ────────────────────────────────────────
    st.markdown(f"""
    <div class="winner-box" style="background: linear-gradient(135deg, {winner_color}, #1a1a2e)">
        🏆 Prédiction : {predicted_winner} gagne<br>
        <small>Proba Red : {proba_red*100:.1f}%  |  Proba Blue : {(1-proba_red)*100:.1f}%</small>
    </div>""", unsafe_allow_html=True)

    # ── Barres de proba ────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"🔴 {f1}", f"{proba_red*100:.1f}%",
                  f"{'✅ Favori' if proba_red >= 0.5 else '❌ Outsider'}")
    with col2:
        st.metric(f"🔵 {f2}", f"{(1-proba_red)*100:.1f}%",
                  f"{'✅ Favori' if proba_red < 0.5 else '❌ Outsider'}")

    st.progress(proba_red)

    st.markdown("---")

    # ── Stats comparatives ─────────────────────────────────────
    st.markdown("### 📊 Statistiques comparatives")
    stat_cols = st.columns(4)
    stats_display = [
        ("🏅 Victoires",       p1['wins'],        p2['wins']),
        ("💀 KO / TKO",        p1['KO_wins'],     p2['KO_wins']),
        ("🤼 Soumissions",     p1['sub_wins'],    p2['sub_wins']),
        ("📏 Reach (cm)",      p1['reach'],       p2['reach']),
    ]
    for col, (label, v1, v2) in zip(stat_cols, stats_display):
        with col:
            winner_v = "🔴" if v1 > v2 else ("🔵" if v2 > v1 else "🟡")
            st.markdown(f"""
            <div class="stat-box">
                <div style='font-size:12px;color:#aaa'>{label}</div>
                <div style='font-size:18px'>
                    <span class='red-text'>{v1:.0f}</span> vs 
                    <span class='blue-text'>{v2:.0f}</span>
                    <span style='font-size:14px'> {winner_v}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Visualisations ─────────────────────────────────────────
    st.markdown("### 🎨 Visualisations")
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown(f"**🕸️ Radar Chart — {f1} vs {f2}**")
        fig_radar = make_radar(p1, p2)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)

    with vcol2:
        st.markdown("**🏟️ Octogone 3D — Zones de frappe**")
        fig_oct = make_octagon(p1, p2, proba_red)
        st.pyplot(fig_oct, use_container_width=True)
        plt.close(fig_oct)

    # ── Bio fighters ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🥋 Fiche des combattants")
    bio1, bio2 = st.columns(2)

    def bio_card(p, color):
        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid {color};border-radius:10px;padding:15px'>
            <h4 style='color:{color}'>{p['name']}</h4>
            <table style='width:100%;color:white;font-size:13px'>
                <tr><td>🏆 Victoires</td><td><b>{p['wins']}</b></td></tr>
                <tr><td>❌ Défaites</td><td><b>{p['losses']}</b></td></tr>
                <tr><td>💥 KO wins</td><td><b>{p['KO_wins']}</b></td></tr>
                <tr><td>🤼 Sub wins</td><td><b>{p['sub_wins']}</b></td></tr>
                <tr><td>📏 Reach</td><td><b>{p['reach']:.0f} cm</b></td></tr>
                <tr><td>📐 Taille</td><td><b>{p['height']:.0f} cm</b></td></tr>
                <tr><td>🥋 Stance</td><td><b>{p['stance']}</b></td></tr>
                <tr><td>🎯 Sig Strike%</td><td><b>{p['sig_str_pct']*100:.1f}%</b></td></tr>
                <tr><td>🤸 Takedown%</td><td><b>{p['td_pct']*100:.1f}%</b></td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with bio1:
        bio_card(p1, '#e74c3c')
    with bio2:
        bio_card(p2, '#3498db')

    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#555;font-size:12px'>FightVision — Projet IGRV + IA | ENP Alger 2025</p>",
                unsafe_allow_html=True)
