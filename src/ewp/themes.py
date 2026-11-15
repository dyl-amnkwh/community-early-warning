"""Zero-shot barrier-theme detection with a sentence transformer (baseline).

Run locally. Do not send interview text to third-party APIs.
"""
import numpy as np

THEME_DESCRIPTIONS = {
    "transportation": "I have trouble getting to school because of the bus, rides, or distance.",
    "housing": "My family has moved, lost housing, or lives in an unstable or crowded place.",
    "caregiving": "I take care of my siblings or family members, which makes it hard to get to school.",
    "health": "I have health problems, mental health struggles, or medical appointments that keep me out.",
    "belonging": "I feel bullied, left out, or unsafe at school.",
    "food": "There isn't always enough food at home.",
    "disengagement": "I feel behind and don't see the point of going to school.",
    "family_conflict": "There is conflict, loss, or serious stress in my family.",
}


def zero_shot_theme_scores(sentences, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Return (themes, scores) where scores has shape (n_sentences, n_themes), cosine similarity."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    themes = list(THEME_DESCRIPTIONS)
    theme_emb = model.encode([THEME_DESCRIPTIONS[t] for t in themes], normalize_embeddings=True)
    sent_emb = model.encode(list(sentences), normalize_embeddings=True)
    return themes, np.asarray(sent_emb) @ np.asarray(theme_emb).T


def aggregate_student_themes(scores: np.ndarray, themes, threshold: float = 0.45) -> dict:
    """Aggregate sentence-level scores into student-level features (count above threshold, max)."""
    out = {}
    for j, t in enumerate(themes):
        col = scores[:, j]
        out[f"theme_{t}"] = int((col >= threshold).sum())
        out[f"theme_{t}_max"] = float(col.max())
    return out
