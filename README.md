---
title: Durée de séjour des patients diabétiques
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Durée de séjour des patients diabétiques : application Flask

Application web qui estime la durée d'hospitalisation d'un patient diabétique (régression).
Elle utilise les deux modèles retenus dans le notebook :

| Scénario | Informations utilisées | Usage |
|---|---|---|
| **A, à l'admission** | connues dès l'arrivée du patient | anticiper la durée |
| **B, en cours de séjour** | A + examens, médicaments, résultats | réévaluer la durée |

## Contenu du projet

```
app.py               application Flask (page web + API JSON)
templates/           page HTML
static/              style et script
models/              modèles exportés depuis le notebook (à remplir, voir ci-dessous)
tests/               tests automatiques
requirements.txt     dépendances
Dockerfile           mise en ligne (Hugging Face Spaces, Render, ...)
```

## 1. Récupérer les modèles (une seule fois)

1. Dans le notebook (Colab), exécutez toutes les cellules jusqu'à la **section 8**.
2. Le fichier `deploiement_modeles.zip` se télécharge.
3. **Décompressez-le à la racine de ce projet** : il remplit `models/` (`model_A.joblib`, `model_B.joblib`, `model_meta.json`) et remplace `requirements.txt` par la version qui fixe la bonne version de scikit-learn.

> Si vous réentraînez le modèle, refaites cette étape : les fichiers `models/` et `requirements.txt` vont ensemble.

## 2. Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # puis ouvrir http://localhost:5000
```

Tests automatiques : `python -m unittest discover -s tests -v`

## 3. Utiliser l'API

```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{
  "scenario": "A",
  "features": {
    "age_num": 65, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2,
    "gender": "Female", "race": "Caucasian", "admission_type_id": "1", "admission_source_id": "7",
    "payer_code": "MC", "medical_specialty": "Unknown", "diag_1_group": "Circulatory"
  }
}'
```

- `GET /api/schema` liste les champs et les valeurs acceptées pour chaque scénario.
- `GET /health` indique si le service et les modèles sont chargés.
- Réponses : `200` prédiction, `400` requête mal formée, `422` valeurs invalides (avec le détail par champ).

## 4. Obtenir un lien public (Hugging Face Spaces, gratuit)

1. Créez un compte sur huggingface.co, puis **New → Space**.
2. Donnez un nom, choisissez **SDK : Docker** (modèle « Blank »), visibilité **Public**, matériel **CPU basic (gratuit)**.
3. Onglet **Files → Add file → Upload files** : glissez **tout le contenu du projet** (y compris `models/` rempli, `README.md` avec son en-tête, `Dockerfile`). Validez avec **Commit**.
4. Le Space se construit en quelques minutes (onglet **Logs**). Quand il affiche « Running », l'application est en ligne.
5. Le lien public est `https://huggingface.co/spaces/<votre-compte>/<nom-du-space>` ; l'application seule est sur `https://<votre-compte>-<nom-du-space>.hf.space`.

Les Spaces gratuits se mettent en veille après une période d'inactivité : **ouvrez le lien quelques minutes avant la présentation** pour le réveiller.

*Alternative : Render (render.com), « New Web Service » depuis un dépôt GitHub contenant ce projet, environnement Docker.*

## En cas de problème

| Symptôme | Cause probable | Solution |
|---|---|---|
| « Fichiers du modèle introuvables » | `models/` vide | refaire l'étape 1 |
| Avertissement de version scikit-learn / erreur au chargement | `requirements.txt` ne correspond pas au modèle | utiliser le `requirements.txt` de l'archive exportée |
| Fichier `.joblib` trop lourd | Random Forest volumineux | l'export allège automatiquement (voir sa sortie) ; sinon baisser `TAILLE_MAX_MO` |

## Limites

Projet pédagogique : ce n'est pas un avis médical. La durée de séjour dépend de facteurs absents des données ; les données (1999-2008) viennent d'hôpitaux américains ; la durée est limitée à 1-14 jours.
