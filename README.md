# NYC Commercial Intelligence

A data-driven decision-support system for exploring and ranking commercial locations in New York City using urban data and machine learning.

---

## Overview

This project integrates NYC Open Data business licensing records and U.S. Census (ACS) data to model neighborhood-level commercial environments.

Users can query neighborhoods using natural language (e.g., *"quiet residential area for boutique retail"*) and receive results ranked by semantic similarity and predicted commercial stability.

The system combines retrieval, clustering, and predictive modeling into a unified pipeline for data-driven location analysis.

---

## Key Components

- **Semantic Retrieval**

    Sentence-transformer embeddings (`all-MiniLM-L6-v2`) with cosine similarity

- **Feature Engineering**

    Aggregated business density, category density per neighborhood, demographics (income, unemployment), foot traffic, text profiles, and persistence labels (`src/feature_engineering.py`)

- **Clustering (From Scratch)**

    K-means implemented in NumPy (`src/kmeans_numpy.py`) 

- **Serialization**

    Loading and saving models, embeddings, and data artifacts (`src/serialization.py`)

- **Supervised Learning**

    Ridge / Random Forest models for persistence prediction (`src/persistence_model.py`)
    Metrics and validation strategies for model evaluation 

- **Ranking Engine**

    Combines similarity and predicted persistence (`src/ranking.py`)


- **Dashboard**

    Streamlit interface (`app.py`)


---

## Project Structure

```
data/        raw and processed datasets
src/         core logic (pipeline, features, models, ranking)
outputs/     saved models, embeddings, figures
tests/       unit tests
app.py       Streamlit entry point
```

---

## Setup

### Recommended (uv)

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
uv pip install -r requirements.txt
```

### Fallback (pip)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

---

## Example Queries

- "quiet residential area for boutique retail"
- "high density food area"
- "stable neighborhood for cafes"

---

## Algorithm Implementation

K-means is implemented from scratch in `src/kmeans_numpy.py`:

- Euclidean distance
- Iterative centroid updates
- Convergence via tolerance
- Random initialization with multiple runs for stability
- Elbow method for optimal cluster selection

---

## Modeling & Ranking

- **Persistence score:** proportion of businesses active beyond a defined time threshold 
- **Models:** Ridge Regression, Random Forest

Final ranking score:

```
score = α · similarity + β · predicted_persistence,  α + β = 1
```

Components are normalized before combination.

---

## Testing

```bash
pytest tests/
```

Includes:

- K-means correctness checks
- Persistence label validation

---

## Notes

- Large datasets are not included in the repository
- Precomputed models and embeddings may be stored in `outputs/`
