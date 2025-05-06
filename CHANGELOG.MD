# 📦 CHANGELOG – ÉcoConso

## [1.1.0] – 2025-05-06

### ✨ Nouvelles fonctionnalités

- Ajout d’un **système de scoring écologique** basé sur un barème (`SCORE_MAP`)
- Nouvelle fonction `calc_scores()` pour calculer les scores par catégorie
- **Diagramme en barres** avec `st.bar_chart` pour afficher les scores
- **Diagramme radar** (via `matplotlib`) pour visualiser le profil écologique global

### 🧑‍💻 Améliorations de l’interface

- Sélection obligatoire de toutes les questions avec une option par défaut ("Sélectionner une option…")
- Validation du formulaire et message d’avertissement en cas de réponse manquante

### 🧹 Re-factorisation et structure

- Organisation claire par sections : Initialisation Mistral, Appel LLM, Interface utilisateur
- Commentaires nettoyés, lisibilité améliorée

### 📦 Dépendances ajoutées

- `pandas`
- `numpy`
- `matplotlib`

---

## [1.0.0] – Initial

- Version initiale avec formulaire Streamlit
- Génération de conseils personnalisés via Mistral LLM