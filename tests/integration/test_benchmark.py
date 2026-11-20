"""The benchmark runner over the real corpus and dataset, using the offline hashing embedder."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from app.embeddings import HashingEmbedder
from app.evaluation.benchmark import (
    BenchmarkError, PipelineConfig, format_summary, main, run_benchmark, write_results,
)
from app.evaluation.compare import ComparisonError, compare_runs, format_comparison
from tests.conftest import CORPUS_DIR


def run(**overrides) -> dict:
    config = PipelineConfig(label="t", embedding_backend="hashing", embedding_model="hashing", **overrides)
    return run_benchmark(config, embedder=HashingEmbedder())


@pytest.fixture(scope="module")
def baseline() -> dict:
    return run()


def test_result_describes_the_experiment_completely(baseline) -> None:
    assert baseline["schema_version"] == 1
    assert baseline["config"]["embedding_model"] == "hashing-1024" and baseline["config"]["chunker"] == "sentence(max=500,overlap=80)"
    assert baseline["config"]["ks"] == [1, 3, 5] and baseline["config"]["retriever"] == "dense"
    d = baseline["dataset"]
    assert (d["name"], d["version"], d["n_questions"], d["n_answerable"], d["n_must_refuse"]) == ("velmora-qa", "0.1.0", 24, 18, 6)
    assert d["labels_human_verified"] == 0 and d["label_sources"] == {"claude_authored": 24} and len(d["corpus_sha256"]) == 64
    assert baseline["index"]["documents"] == 7 and baseline["index"]["chunks"] > 7
    assert baseline["latency"]["tokens_and_cost"] is None  # retrieval-only: no LLM, so no tokens or cost
    assert baseline["run"]["git"] and baseline["run"]["environment"]["numpy"]


def test_all_metrics_are_valid_probabilities_and_beat_the_random_control(baseline) -> None:
    overall, control = baseline["metrics"]["overall"], baseline["baselines"]["random"]["overall"]
    assert overall["n"] == 18
    for name, value in overall.items():
        if name != "n":
            assert 0.0 <= value <= 1.0, name
    assert overall["mrr"] > control["mrr"] and overall["hit@3"] > control["hit@3"]
    for k in (1, 3, 5):  # cut-offs can only help: hit/recall are monotone in k
        assert overall[f"hit@{k}"] <= overall["hit@5"] and overall[f"recall@{k}"] <= overall["recall@5"]


def test_every_required_quote_is_reachable_with_the_default_chunker(baseline) -> None:
    assert baseline["metrics"]["reachability"]["mean_required_quotes_reachable"] == 1.0


def test_only_answerable_categories_get_retrieval_metrics(baseline) -> None:
    assert set(baseline["metrics"]["by_category"]) == {
        "conflicting_outdated", "direct_factual", "exact_identifier", "multi_document", "multi_hop", "paraphrase"}
    refused = [q for q in baseline["questions"] if not q["answerable"]]
    assert len(refused) == 6 and all(q["metrics"] is None for q in refused)
    assert baseline["metrics"]["top1_score"]["must_refuse"]["n"] == 6


def test_the_exact_identifier_questions_are_found_by_a_lexical_embedder(baseline) -> None:
    assert baseline["metrics"]["by_category"]["exact_identifier"]["mrr"] == 1.0


def test_same_configuration_and_dataset_reproduce_the_same_results(baseline) -> None:
    again = run()
    assert again["results_sha256"] == baseline["results_sha256"]
    assert again["questions"] == baseline["questions"] and again["metrics"] == baseline["metrics"]
    assert again["index"]["chunks"] == baseline["index"]["chunks"]  # no state leaks between runs in one process
    assert again["baselines"] == baseline["baselines"]  # the seeded control is reproducible too


def test_changing_the_pipeline_changes_the_hash(baseline) -> None:
    other = run(chunk_max_chars=200, chunk_overlap_chars=40)
    assert other["results_sha256"] != baseline["results_sha256"]
    assert other["index"]["chunks"] > baseline["index"]["chunks"]


def test_a_dataset_that_no_longer_matches_the_corpus_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS_DIR, corpus)
    (corpus / "expense_policy.txt").write_text("Edited text.\n", encoding="utf-8")
    config = PipelineConfig(label="t", embedding_backend="hashing")
    with pytest.raises(BenchmarkError, match="corpus hash mismatch"):
        run_benchmark(config, corpus_dir=corpus, embedder=HashingEmbedder())


def test_results_are_saved_as_json_and_csv(baseline, tmp_path: Path) -> None:
    json_path, csv_path = write_results(baseline, tmp_path)
    assert json.loads(json_path.read_text("utf-8"))["results_sha256"] == baseline["results_sha256"]
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 24 and {"id", "rr", "ndcg@5", "tie_with_gold", "no_signal", "latency_ms"} <= set(rows[0])
    q008 = next(r for r in rows if r["id"] == "q008")
    assert q008["rr"] == "1.0" and q008["top1_doc"] == "it_security_handbook.txt"
    assert next(r for r in rows if r["id"] == "q020")["rr"] == ""  # must-refuse: metrics are blank, not zero


def test_the_summary_warns_about_unverified_labels(baseline) -> None:
    text = format_summary(baseline)
    assert "random ranking (control)" in text and "0/24 human-verified" in text and "WARNING" in text


def test_cli_runs_end_to_end_and_saves_files(tmp_path: Path, capsys) -> None:
    assert main(["--label", "cli", "--embedder", "hashing", "--output-dir", str(tmp_path)]) == 0
    assert "results_sha256" in capsys.readouterr().out
    assert len(list(tmp_path.glob("cli-*.json"))) == 1 and len(list(tmp_path.glob("cli-*.csv"))) == 1


# ------------------------------------------------------------------ comparing runs
def test_comparing_a_run_with_itself_shows_no_change(baseline) -> None:
    c = compare_runs(baseline, baseline)
    assert c.config_differences == [] and all(r.verdict == "same" and r.wins == r.losses == 0 for r in c.overall)
    assert all(r.ties == 18 for r in c.overall)


def test_comparison_reports_deltas_and_per_question_wins_and_losses(baseline) -> None:
    candidate = run(chunk_max_chars=200, chunk_overlap_chars=40)
    c = compare_runs(baseline, candidate)
    assert {k for k, *_ in c.config_differences} >= {"chunk_max_chars", "chunker"}
    for row in c.overall:
        assert row.wins + row.losses + row.ties == 18
        assert row.delta == pytest.approx(row.b - row.a)
    assert set(c.by_category) == set(baseline["metrics"]["by_category"])
    text = format_comparison(c)
    assert "Caution" in text and "mrr" in text and "better/worse/same" in text


def test_runs_on_different_datasets_are_not_comparable(baseline) -> None:
    other = json.loads(json.dumps(baseline))
    other["dataset"]["corpus_sha256"] = "f" * 64
    with pytest.raises(ComparisonError, match="not comparable"):
        compare_runs(baseline, other)
    assert compare_runs(baseline, other, force=True).n_questions == 18
