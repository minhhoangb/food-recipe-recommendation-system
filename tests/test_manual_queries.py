from collections import Counter

import pandas as pd

from src.contracts import PredictRequest
from src.utils import REPO_ROOT

MANUAL = REPO_ROOT / "reports" / "manual_eval"


def test_manual_query_set_is_balanced_and_valid():
    queries = pd.read_csv(MANUAL / "queries.csv")
    assert len(queries) == 40 and queries["query_id"].is_unique
    assert Counter(queries["split"]) == {"dev": 20, "final": 20}
    for _, split in queries.groupby("split"):
        assert Counter(split["difficulty"]) == {"easy": 6, "medium": 7, "hard": 7}
    for row in queries.itertuples():
        PredictRequest(ingredients=row.ingredients.split(";"))  # same validation as the API
        assert row.text.strip() and row.probes.strip()


def test_label_template_has_the_columns_of_the_guide():
    columns = pd.read_csv(MANUAL / "labels_template.csv").columns
    assert {"query_id", "item_id", "recipe_id", "relevance", "annotator", "notes"} <= set(columns)
    assert not any("model" in column for column in columns)  # the pool stays blind
