# =============================================================
#  FightVision — Clustering des styles de combat
#  Regroupe les fighters par profil statistique (K-Means) pour
#  faire émerger des archétypes de style, indépendamment des
#  catégories de poids ou de la réputation.
# =============================================================

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

FIGHTERS_PATH = "fighters.json"
OUTPUT_DIR    = "outputs"
MIN_BOUTS     = 3   # fighters with fewer bouts are too noisy to define a cluster centroid
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────
print("📂 Chargement de fighters.json...")
with open(FIGHTERS_PATH, encoding="utf-8") as f:
    fighters = json.load(f)

df = pd.DataFrame(fighters.values())
df.index = list(fighters.keys())
print(f"   {len(df)} fighters")

df['KO_rate']  = (df['KO_wins']  / df['wins'].replace(0, np.nan)).fillna(0)
df['sub_rate'] = (df['sub_wins'] / df['wins'].replace(0, np.nan)).fillna(0)

FEATURES = ['sig_str_pct', 'td_pct', 'kd_avg', 'ctrl_time',
            'head_landed', 'body_landed', 'leg_landed',
            'KO_rate', 'sub_rate']

# ─────────────────────────────────────────────────────────────
# 2. FIT sur les fighters avec un historique suffisant
#    (évite que des profils à 1-2 combats déforment les centroïdes)
# ─────────────────────────────────────────────────────────────
fit_df = df[df['bouts'] >= MIN_BOUTS].copy()
print(f"   {len(fit_df)} fighters avec >= {MIN_BOUTS} combats (utilisés pour fitter le modèle)")

X_fit = fit_df[FEATURES].fillna(0)
scaler = StandardScaler().fit(X_fit)
Xs_fit = scaler.transform(X_fit)

# ─────────────────────────────────────────────────────────────
# 3. CHOIX DE K — silhouette score, pas une supposition
# ─────────────────────────────────────────────────────────────
print("\n🔎 Sélection de k (silhouette score)...")
sil_scores = {}
for k in range(2, 7):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(Xs_fit)
    sil_scores[k] = silhouette_score(Xs_fit, labels_k)
    print(f"   k={k} → silhouette={sil_scores[k]:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n   Meilleur k retenu : {best_k} (silhouette={sil_scores[best_k]:.4f})")

# ─────────────────────────────────────────────────────────────
# 4. CLUSTERING FINAL + LABELS DESCRIPTIFS
#    Les libellés sont dérivés des centroïdes réels (voir analyse),
#    pas choisis a priori.
# ─────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
fit_df['cluster'] = kmeans.fit_predict(Xs_fit)

centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES
)
print("\n📊 Centroïdes (valeurs réelles, non standardisées) :")
print(centroids.round(3))

STYLE_LABELS = {
    # dérivés empiriquement de l'analyse des centroïdes (voir README)
    'striker_explosive': "Frappeur explosif",
    'striker_volume':    "Frappeur volumineux",
    'grappler':           "Grappler / Finisseur au sol",
    'balanced':            "Polyvalent",
}

def label_cluster(row):
    """Règle simple basée sur la dimension la plus distinctive du centroïde."""
    if row['KO_rate'] > 0.4 and row['ctrl_time'] < 110:
        return STYLE_LABELS['striker_explosive']
    if row['td_pct'] > 0.3 and row['sub_rate'] > 0.2:
        return STYLE_LABELS['grappler']
    if row['head_landed'] > centroids['head_landed'].mean():
        return STYLE_LABELS['striker_volume']
    return STYLE_LABELS['balanced']

cluster_to_label = {i: label_cluster(centroids.iloc[i]) for i in range(best_k)}
print("\n🏷️  Labels assignés :")
for cid, lbl in cluster_to_label.items():
    print(f"   Cluster {cid} → {lbl}")

# ─────────────────────────────────────────────────────────────
# 5. PRÉDIRE LE CLUSTER POUR TOUS LES FIGHTERS
#    (y compris ceux sous le seuil MIN_BOUTS, via le modèle fitté)
# ─────────────────────────────────────────────────────────────
X_all = df[FEATURES].fillna(0)
Xs_all = scaler.transform(X_all)
df['cluster'] = kmeans.predict(Xs_all)
df['style'] = df['cluster'].map(cluster_to_label)

print("\n📈 Répartition finale des styles (tous fighters) :")
counts = df['style'].value_counts()
for style, n in counts.items():
    print(f"   {style:32s} {n:4d} ({n/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 6. EXPORT — fighters.json enrichi avec cluster + style
# ─────────────────────────────────────────────────────────────
for name, row in df.iterrows():
    fighters[name]['cluster'] = int(row['cluster'])
    fighters[name]['style'] = row['style']

with open(FIGHTERS_PATH, "w", encoding="utf-8") as f:
    json.dump(fighters, f, ensure_ascii=False)
print(f"\n✅ {FIGHTERS_PATH} mis à jour avec les champs cluster/style")

# ─────────────────────────────────────────────────────────────
# 7. VISUALISATION — PCA 2D + profil des centroïdes
# ─────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xs_all)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(f"FightVision — Clustering des styles (k={best_k}, silhouette={sil_scores[best_k]:.3f})",
             fontsize=14, fontweight='bold')

palette = ['#d6263c', '#1f6fc4', '#e2a23b', '#3aa564']
for cid in range(best_k):
    mask = df['cluster'] == cid
    axes[0].scatter(coords[mask.values, 0], coords[mask.values, 1],
                     s=14, alpha=0.6, color=palette[cid % len(palette)],
                     label=f"{cluster_to_label[cid]} (n={mask.sum()})")
axes[0].set_title("Projection PCA (2D) des profils de fighters")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)")
axes[0].legend(fontsize=8, loc='best')

centroid_z = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES)
x = np.arange(len(FEATURES))
width = 0.8 / best_k
for cid in range(best_k):
    axes[1].bar(x + cid*width, centroid_z.iloc[cid], width=width,
                color=palette[cid % len(palette)], label=cluster_to_label[cid])
axes[1].set_xticks(x + width*(best_k-1)/2)
axes[1].set_xticklabels(FEATURES, rotation=35, ha='right', fontsize=8)
axes[1].axhline(0, color='#888', linewidth=0.8)
axes[1].set_title("Profil des centroïdes (z-score)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/clustering_results.png", dpi=120, bbox_inches='tight')
print(f"✅ Graphique sauvegardé → {OUTPUT_DIR}/clustering_results.png")
