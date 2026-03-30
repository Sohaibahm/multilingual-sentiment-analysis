# Multilingual Sentiment Analysis — FR/EN Reviews

> Analyse de sentiments sur des avis consommateurs multilingues (français/anglais) — approche hybride combinant analyse lexicale et Deep Learning (LSTM).

---

## Overview

Ce projet implémente un pipeline complet d'analyse de sentiments sur un dataset de 5 000 avis consommateurs multilingues (français/anglais). Il combine deux approches complémentaires : une analyse **lexicale** (VADER + lexique FR custom) et un modèle **Deep Learning LSTM** pour la classification en 3 classes (positive, negative, neutral).

---

## Dataset

| Propriété | Valeur |
|---|---|
| **Source** | `social_media_reviews_dataset.csv` |
| **Taille** | 5 000 reviews |
| **Langues** | Français & Anglais (multilingue) |
| **Classes** | Positive / Negative / Neutral |
| **Colonnes** | `id`, `text`, `sentiment` |

**Exemples de données :**
```
"Franchement, Outstanding! Meilleur achat de l'année 🔥"  → positive
"Hélas... Je regrette cet achat... Totally disappointed"   → negative
"Bon, It's okay. Correspond à la description. We'll see."  → neutral
```

---

## Architecture du pipeline

```
[Dataset CSV]
     ↓
[Preprocessing]
  - Nettoyage (URLs, HTML, espaces)
  - Tokenisation (NLTK)
  - Suppression stopwords FR+EN (conservation des négations)
  - Lemmatisation bilingue (spaCy fr_core_news_sm + en_core_web_sm)
     ↓
[Approche 1 — Analyse Lexicale]
  - VADER (anglais)
  - Lexique FR custom + boosters + gestion négations
  - Classification : positive / negative / neutral
     ↓
[Approche 2 — Deep Learning LSTM]
  - Tokenisation Keras + padding
  - Embedding Layer (dim=64)
  - LSTM (64 units)
  - Dense softmax (3 classes)
  - Entraînement : 10 epochs, batch=32, validation_split=0.2
     ↓
[Évaluation]
  - Accuracy, Classification Report
  - Matrice de confusion
  - Courbes Loss / Accuracy
```

---

## Stack Technique

```
Python 3.x
NLTK                    # Tokenisation, stopwords
spaCy                   # Lemmatisation bilingue (fr + en)
VADER Sentiment         # Analyse lexicale anglais
TextBlob                # Analyse lexicale complémentaire
Scikit-learn            # TF-IDF, Naive Bayes, métriques
TensorFlow / Keras      # Modèle LSTM
WordCloud               # Visualisation
Matplotlib / Seaborn    # Graphiques et matrices de confusion
Pandas / NumPy          # Manipulation des données
```

---

## Structure du projet

```
├── Sentiment.ipynb                     # Notebook principal
├── data/
│   └── social_media_reviews_dataset.csv  # Dataset multilingue
└── README.md
```

---

## Points clés du preprocessing

**Gestion des négations** — les négations sont conservées dans les stopwords car elles portent du sentiment :
```python
NEGATIONS = {"ne", "pas", "jamais", "rien", "plus", "aucun", "not", "no", "never", "n't"}
stop_for_models = stop_all - NEGATIONS
```

**Lexique FR custom avec boosters** :
```python
LEX_FR = {
    "excellent": 2.0, "génial": 1.8, "parfait": 1.8,
    "déçu": -1.5, "mauvais": -2.0, "arnaque": -2.2, ...
}
BOOSTERS_FR = {"très": 1.3, "vraiment": 1.3, "extrêmement": 1.6}
```

**Lemmatisation bilingue** avec spaCy — modèles `fr_core_news_sm` et `en_core_web_sm`

---

## Modèle LSTM

```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_shape=(MAX_LEN,)),
    LSTM(64),
    Dense(3, activation='softmax')  # 3 classes
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

---

## Évaluation

- **Classification report** — précision, rappel, F1-score par classe
- **Matrice de confusion** — visualisation des erreurs de classification
- **Courbes d'entraînement** — loss et accuracy train vs validation

---

## Améliorations envisagées

- Fine-tuning d'un modèle **CamemBERT** (BERT français) pour de meilleures performances
- Ajout d'une détection automatique de la langue (langdetect)
- Gestion des emojis comme indicateurs de sentiment
- Déploiement via une **API FastAPI** pour l'inférence en temps réel
- Interface de démonstration avec **Streamlit**

---

## Auteur

**HASSIB Sohaib** — AI Engineer
