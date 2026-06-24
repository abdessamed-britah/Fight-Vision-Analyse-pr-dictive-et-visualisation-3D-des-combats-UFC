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
├── fighters.json         ← Profils des fighters exportés pour le front
├── landing.html          ← Page d'accueil marketing
├── index.html            ← Application complète (prédiction + radar + octogone 3D)
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

## 🚀 Utilisation

### Entraîner le modèle ML
```bash
python jour1_model.py
```
- Charge le dataset, fait le feature engineering (25 features)
- Entraîne Random Forest + Gradient Boosting
- Sauvegarde le meilleur modèle dans `outputs/model.pkl`
- Affiche confusion matrix + feature importance

### Lancer l'application web
```bash
npx serve .
```
- Ouvrir `landing.html` (page d'accueil) ou `index.html` (application)
- Sélection de 2 fighters → prédiction + radar Canvas 2D + octogone 3D Three.js
- Servir en HTTP local est nécessaire (le `fetch('fighters.json')` échoue en `file://`)

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

**Performance réelle (après augmentation par inversion de coin R↔B) :**
- Accuracy : ~63% · Balanced accuracy : ~57% · Macro F1 : ~0.58
- Le dataset brut est déséquilibré (Red gagne 67% des combats car le coin Red
  correspond historiquement au favori). Sans correction, un modèle atteint
  68% d'accuracy en prédisant quasi toujours Red (recall Blue ~16%) — un
  score trompeur. L'augmentation par inversion de coin rééquilibre les
  classes à l'entraînement et force le modèle à apprendre une vraie
  différence de niveau plutôt qu'un biais de corner.

---

## 🎨 Partie IGRV

### Radar Chart (Canvas 2D)
Comparaison de 6 attributs normalisés entre les 2 fighters, fond sombre, couleurs UFC (rouge/bleu) — dans `index.html`.

### Octogone 3D (Three.js)
- Modélisation géométrique de l'octogone (8 faces + cage) interactive (drag à la souris)
- Heatmap des zones de frappe projetée sur le sol (sphères proportionnelles)
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
