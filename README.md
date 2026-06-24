# ⚡ FightVision UFC — IA + IGRV

> Analyse prédictive et visualisation 3D des combats UFC  
> Projet IGRV — ENP Alger, Master 2 Big Data & Data Mining

---

## 🗂️ Structure du projet

```
ufc_fightvision/
│
├── data.csv              ← Dataset UFC (6012 combats, 144 colonnes)
├── requirements.txt      ← Dépendances Python
│
├── jour1_model.py        ← Preprocessing + ML (Random Forest / GBT)
├── jour2_radar_3d.py     ← Radar chart + Octogone 3D (matplotlib)
├── jour3_app.py          ← Application Streamlit complète
│
└── outputs/
    ├── model.pkl         ← Modèle sauvegardé (généré par jour1)
    ├── model_results.png ← Confusion matrix + feature importance
    └── fight_*.png       ← Figures de combat générées
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation — 3 étapes

### JOUR 1 — Entraîner le modèle ML
```bash
python jour1_model.py
```
- Charge le dataset, fait le feature engineering (25 features)
- Entraîne Random Forest + Gradient Boosting
- Sauvegarde le meilleur modèle dans `outputs/model.pkl`
- Affiche confusion matrix + feature importance

### JOUR 2 — Visualisations standalone
```bash
python jour2_radar_3d.py
```
- Génère un radar chart comparatif (2 fighters)
- Génère l'octogone 3D avec heatmap des zones de frappe
- Modifie `FIGHTER_1` et `FIGHTER_2` dans le script

### JOUR 3 — Application complète
```bash
streamlit run jour3_app.py
```
- Interface web interactive
- Sélection de 2 fighters → prédiction + visualisations
- Fiche détaillée des combattants

---

## 🧠 Modèle ML

| Modèle             | Description                              |
|--------------------|------------------------------------------|
| Random Forest      | 200 arbres, max_depth=8                  |
| Gradient Boosting  | 200 estimateurs, lr=0.05, max_depth=4    |

**Features clés (25 au total) :**
- Différentiels : reach, taille, wins, win_streak, KO/sub wins
- Stats frappe : sig_strike%, takedown%, KD moyen, ctrl_time
- Précision par zone : HEAD, BODY, LEG (pour R et B)
- Style : stance (Orthodox / Southpaw / Switch)

**Performance attendue :** ~65–70% d'accuracy (cohérent avec la littérature)

---

## 🎨 Partie IGRV

### Radar Chart (polar)
Comparaison de 6 attributs normalisés entre les 2 fighters, fond sombre, couleurs UFC (rouge/bleu).

### Octogone 3D (Matplotlib 3D)
- Modélisation géométrique de l'octogone (8 faces + cage)
- Heatmap des zones de frappe projetée sur le sol (cercles proportionnels)
- Silhouettes 3D des 2 fighters positionnées dans l'octogone
- Couleur des fighters = probabilité ML de victoire

---

## 📌 Exemples de fighters dans le dataset

```
Conor McGregor, Khabib Nurmagomedov, Jon Jones, Israel Adesanya,
Dustin Poirier, Leon Edwards, Nate Diaz, Amanda Nunes...
```

---

## 👨‍💻 Auteur

Abdessamed — ENP Alger, M2 Big Data & Data Mining  
Projet : Informatique Graphique et Réalisation Virtuelle (IGRV)
