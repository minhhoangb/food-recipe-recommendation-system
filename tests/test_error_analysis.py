import pandas as pd

from src.error_analysis import BUCKETS, bucket_rows


def test_query_length_table_includes_queries_collapsed_to_one_ingredient():
    frame = pd.DataFrame({"n_ingredients": [1, 2, 6], "rank_FT": [2, 1, None]})
    _, column, labels = BUCKETS[0]
    rows = bucket_rows(frame, ["FT"], column, labels)
    assert sum(row[1] for row in rows[:-1]) == rows[-1][1] == 3
    assert rows[0] == ["1", 1, "0.500"]
