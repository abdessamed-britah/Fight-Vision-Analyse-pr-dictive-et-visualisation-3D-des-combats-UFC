# =============================================================
#  FightVision — JOUR 1 : Preprocessing + Modèle ML
#  Dataset : data.csv  (6012 combats UFC)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils import shuffle
import seaborn as sns
import joblib
import os

# ── Chemins ──────────────────────────────────────────────────
DATA_PATH   = "data.csv"          # <-- mets ton chemin absolu si besoin
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────
print("📂 Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   {len(df)} combats, {len(df.columns)} colonnes")

# ─────────────────────────────────────────────────────────────
# 2. NETTOYAGE & TARGET
# ─────────────────────────────────────────────────────────────
# On garde uniquement Red / Blue (on enlève Draw : 110 cas)
df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
df['target'] = (df['Winner'] == 'Red').astype(int)   # 1 = Red gagne, 0 = Blue gagne

print(f"   Après suppression des Draw : {len(df)} combats")
print(f"   Red wins : {df['target'].sum()} | Blue wins : {(df['target']==0).sum()}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n⚙️  Feature engineering...")

features = {}

# Différentiels clés R - B
features['reach_diff']         = df['R_Reach_cms']        - df['B_Reach_cms']
features['height_diff']        = df['R_Height_cms']        - df['B_Height_cms']
features['age_diff']           = df['R_age']               - df['B_age']
features['wins_diff']          = df['R_wins']              - df['B_wins']
features['losses_diff']        = df['R_losses']            - df['B_losses']
features['win_streak_diff']    = df['R_current_win_streak'] - df['B_current_win_streak']
features['lose_streak_diff']   = df['R_current_lose_streak']- df['B_current_lose_streak']
features['title_bouts_diff']   = df['R_total_title_bouts'] - df['B_total_title_bouts']
features['KO_wins_diff']       = df['R_win_by_KO/TKO']    - df['B_win_by_KO/TKO']
features['sub_wins_diff']      = df['R_win_by_Submission'] - df['B_win_by_Submission']

# Stats de frappe
features['sig_str_pct_diff']   = df['R_avg_SIG_STR_pct']  - df['B_avg_SIG_STR_pct']
features['td_pct_diff']        = df['R_avg_TD_pct']        - df['B_avg_TD_pct']
features['kd_diff']            = df['R_avg_KD']            - df['B_avg_KD']
features['ctrl_time_diff']     = df['R_avg_CTRL_time(seconds)'] - df['B_avg_CTRL_time(seconds)']
features['sub_att_diff']       = df['R_avg_SUB_ATT']       - df['B_avg_SUB_ATT']

# Strikes par zone (R)
features['R_head_acc']  = df['R_avg_HEAD_landed']  / (df['R_avg_HEAD_att']  + 1)
features['R_body_acc']  = df['R_avg_BODY_landed']  / (df['R_avg_BODY_att']  + 1)
features['R_leg_acc']   = df['R_avg_LEG_landed']   / (df['R_avg_LEG_att']   + 1)

# Strikes par zone (B)
features['B_head_acc']  = df['B_avg_HEAD_landed']  / (df['B_avg_HEAD_att']  + 1)
features['B_body_acc']  = df['B_avg_BODY_landed']  / (df['B_avg_BODY_att']  + 1)
features['B_leg_acc']   = df['B_avg_LEG_landed']   / (df['B_avg_LEG_att']   + 1)

# Style (stance)
le = LabelEncoder()
df['R_Stance_enc'] = le.fit_transform(df['R_Stance'].fillna('Unknown'))
df['B_Stance_enc'] = le.fit_transform(df['B_Stance'].fillna('Unknown'))
features['R_stance'] = df['R_Stance_enc']
features['B_stance'] = df['B_Stance_enc']

# Title bout
features['title_bout'] = df['title_bout'].astype(int)

# Assembler
X = pd.DataFrame(features)
y = df['target']

# Imputation NaN par médiane
X = X.fillna(X.median())

print(f"   Features créées : {X.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 4. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train : {len(X_train)} | Test : {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 4bis. AUGMENTATION PAR INVERSION DE COIN (R ↔ B)
# ─────────────────────────────────────────────────────────────
# Le dataset est déséquilibré (Red gagne ~67% des combats) car le coin Red
# correspond historiquement au favori. Les modèles entraînés tels quels
# apprennent surtout "prédire Red" plutôt qu'une vraie différence de niveau
# (recall Blue Win ~16%). On corrige ça en dupliquant chaque combat
# d'entraînement avec les coins inversés : les features différentielles
# (R - B) sont négées et le label est inversé. Ça rééquilibre parfaitement
# les classes et double les données, sans toucher au jeu de test.
DIFF_COLS = [c for c in X.columns if c.endswith('_diff')]
SWAP_PAIRS = [('R_head_acc', 'B_head_acc'), ('R_body_acc', 'B_body_acc'),
              ('R_leg_acc', 'B_leg_acc'), ('R_stance', 'B_stance')]

def mirror_corners(X_subset):
    Xm = X_subset.copy()
    for c in DIFF_COLS:
        Xm[c] = -Xm[c]
    for a, b in SWAP_PAIRS:
        Xm[a], Xm[b] = X_subset[b].values, X_subset[a].values
    return Xm

X_train_aug = pd.concat([X_train, mirror_corners(X_train)], ignore_index=True)
y_train_aug = pd.concat([y_train, 1 - y_train], ignore_index=True)
X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)

print(f"   Après augmentation miroir : {len(X_train_aug)} combats d'entraînement "
      f"(Red {y_train_aug.sum()} / Blue {(y_train_aug==0).sum()})")

# ─────────────────────────────────────────────────────────────
# 5. MODÈLES
# ─────────────────────────────────────────────────────────────
print("\n🤖 Entraînement des modèles...")

models = {
    "Random Forest"        : RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_aug, y_train_aug)
    preds    = model.predict(X_test)
    acc      = accuracy_score(y_test, preds)
    bal_acc  = balanced_accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    results[name] = {"model": model, "acc": acc, "bal_acc": bal_acc, "macro_f1": macro_f1, "preds": preds}
    print(f"   {name:25s} — Accuracy : {acc:.4f} | Balanced Acc : {bal_acc:.4f} | Macro F1 : {macro_f1:.4f}")

# Meilleur modèle : sélection par balanced accuracy (l'accuracy brute favorise
# artificiellement un modèle qui sur-prédit la classe majoritaire)
best_name  = max(results, key=lambda k: results[k]['bal_acc'])
best_model = results[best_name]['model']
best_preds = results[best_name]['preds']
print(f"\n🏆 Meilleur modèle : {best_name} "
      f"(Accuracy {results[best_name]['acc']*100:.1f}% | Balanced Acc {results[best_name]['bal_acc']*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 6. RAPPORT DE CLASSIFICATION
# ─────────────────────────────────────────────────────────────
print("\n📊 Rapport de classification :")
print(classification_report(y_test, best_preds, target_names=['Blue Win', 'Red Win']))

# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"FightVision — {best_name}", fontsize=14, fontweight='bold')

# (A) Confusion Matrix
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            xticklabels=['Blue Win', 'Red Win'],
            yticklabels=['Blue Win', 'Red Win'])
axes[0].set_title("Matrice de Confusion")
axes[0].set_xlabel("Prédit"); axes[0].set_ylabel("Réel")

# (B) Feature Importance (top 15)
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top15 = importances.nlargest(15)
top15.sort_values().plot(kind='barh', ax=axes[1], color='#c0392b')
axes[1].set_title("Top 15 Features Importantes")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_results.png", dpi=120, bbox_inches='tight')
plt.show()
print(f"\n✅ Graphique sauvegardé → {OUTPUT_DIR}/model_results.png")

# ─────────────────────────────────────────────────────────────
# 8. SAUVEGARDE DU MODÈLE
# ─────────────────────────────────────────────────────────────
joblib.dump(best_model, f"{OUTPUT_DIR}/model.pkl")
X.columns.to_series().to_csv(f"{OUTPUT_DIR}/feature_names.csv", index=False)
print(f"✅ Modèle sauvegardé → {OUTPUT_DIR}/model.pkl")

# ─────────────────────────────────────────────────────────────
# 9. PROFIL D'UN FIGHTER (pour le radar chart — Jour 2)
# ─────────────────────────────────────────────────────────────
def get_fighter_profile(df_raw, fighter_name, corner='R'):
    """Extrait les stats moyennes d'un fighter."""
    mask = df_raw[f'{corner}_fighter'] == fighter_name
    sub  = df_raw[mask]
    if len(sub) == 0:
        return None
    p = corner
    return {
        'name'        : fighter_name,
        'wins'        : sub[f'{p}_wins'].iloc[0],
        'losses'      : sub[f'{p}_losses'].iloc[0],
        'KO_wins'     : sub[f'{p}_win_by_KO/TKO'].iloc[0],
        'sub_wins'    : sub[f'{p}_win_by_Submission'].iloc[0],
        'sig_str_pct' : sub[f'{p}_avg_SIG_STR_pct'].mean(),
        'td_pct'      : sub[f'{p}_avg_TD_pct'].mean(),
        'kd_avg'      : sub[f'{p}_avg_KD'].mean(),
        'ctrl_time'   : sub[f'{p}_avg_CTRL_time(seconds)'].mean(),
        'head_landed' : sub[f'{p}_avg_HEAD_landed'].mean(),
        'body_landed' : sub[f'{p}_avg_BODY_landed'].mean(),
        'leg_landed'  : sub[f'{p}_avg_LEG_landed'].mean(),
        'reach'       : sub[f'{p}_Reach_cms'].iloc[0],
        'height'      : sub[f'{p}_Height_cms'].iloc[0],
        'stance'      : sub[f'{p}_Stance'].iloc[0],
    }

# Test rapide
profile = get_fighter_profile(df, 'Conor McGregor', 'R')
if profile:
    print("\n🥊 Profil Conor McGregor :")
    for k, v in profile.items():
        print(f"   {k:15s}: {v}")

print("\n✅ JOUR 1 TERMINÉ — Modèle prêt, profils extraits.")
print("   → Ouvre index.html (servi en HTTP local) pour l'application.")
