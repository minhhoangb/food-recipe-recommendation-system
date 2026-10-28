import mlflow
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.baseline import fit_retriever, save_bundle
from src.compare import main
from src.config import processed_dir
from src.evaluate import evaluate_retriever, load_split
from src.utils import read_json, read_jsonl


@pytest.fixture(scope="module")
def setup(built_fixture, tmp_path_factory):
    cfg, _, out = built_fixture
    directory = processed_dir(cfg, "main")
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    train_ids = set(read_json(directory / "splits.json")["train"])
    tmp = tmp_path_factory.mktemp("compare")
    main_cfg = OmegaConf.merge(cfg, {"data": {"profile": "main"}})
    bundles = {}
    for label, model in (("E01", "tfidf"), ("E03", "bm25")):
        retriever, info = fit_retriever(model, corpus, train_ids)
        bundles[label] = (save_bundle(retriever, info, main_cfg, tmp / "models"), retriever)
    config = tmp / "fixture.yaml"
    config.write_text(f"paths:\n  processed_dir: {out.as_posix()}\ndata:\n  profile: main\n", encoding="utf-8")
    return directory, bundles, config, tmp


def test_comparison_files_match_the_evaluator(setup):
    directory, bundles, config, tmp = setup
    uri = f"sqlite:///{(tmp / 'mlflow.db').as_posix()}"
    systems = [arg for label, (bundle, _) in bundles.items() for arg in ("--system", f"{label}={bundle}")]
    assert main(["--config", str(config), *systems, "--out", str(tmp), "--tracking-uri", uri]) == 0

    table = pd.read_csv(tmp / "comparison.csv").set_index("experiment")
    queries, qrels = load_split(directory, "validation")
    for label, (_, retriever) in bundles.items():
        expected = evaluate_retriever(retriever, queries, qrels)
        assert table.loc[label, "val_mrr@10"] == pytest.approx(expected["mrr@10"])
        assert table.loc[label, "val_n_queries"] == len(queries)
        run = mlflow.MlflowClient(uri).get_run(table.loc[label, "mlflow_val_run"])
        assert run.data.tags["experiment"] == label and run.info.run_name.startswith(f"{label}-")
    assert table.loc["E03", "cv_note"].startswith("N/A: rank-bm25")  # N/A always carries a reason
    assert table.loc["E01", "cv_note"].startswith("N/A: CV not run")  # no tfidf_cv5.json in this folder

    per_query = pd.read_csv(tmp / "comparison_per_query.csv")
    assert len(per_query) == len(bundles) * len(qrels)
    means = per_query.groupby("experiment")["mrr@10"].mean()
    assert means["E01"] == pytest.approx(table.loc["E01", "val_mrr@10"])
    assert "ranx.compare" in (tmp / "comparison_ranx.md").read_text(encoding="utf-8")
