"""Model factories and nested feature-set helpers."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def nested_feature_sets(groups: dict) -> dict:
    """Build cumulative feature sets A-D from config feature_groups."""
    a = list(groups["attendance"])
    b = a + list(groups["academics"]) + list(groups["discipline"])
    c = b + list(groups["neighborhood"])
    d = c + list(groups["interview_themes"])
    return {"A_attendance": a, "B_plus_academics": b, "C_plus_neighborhood": c, "D_plus_themes": d}


def build_logreg(C: float = 0.5):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=1000),
    )


def build_rf(n_estimators: int = 500, random_state: int = 42):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
        ),
    )


def build_xgb(scale_pos_weight: float = 1.0, random_state: int = 42):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=random_state,
    )
