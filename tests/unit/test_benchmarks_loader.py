import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
from benchmarks import comparison_rows, deltas, load_runs  # noqa: E402


def test_loads_committed_runs_and_skips_non_results(tmp_path):
    for f in sorted((ROOT / "docs" / "results").glob("m3-*.json")):
        shutil.copy(f, tmp_path)
    (tmp_path / "ragas-x.json").write_text(json.dumps({"summary": {}}))
    (tmp_path / "broken.json").write_text("{not json")
    runs = load_runs([tmp_path, tmp_path / "missing"])
    assert {r["label"] for r in runs} == {"m3-dense-hashing", "m3-bm25-hashing", "m3-hybrid-hashing"}
    assert all(r["stand_in"] for r in runs)  # hashing embedder => flagged as stand-in
    rows = comparison_rows(runs)
    assert rows[0]["run"] and "mrr" in rows[0]
    dense = next(r for r in runs if r["retriever"] == "dense")
    hybrid = next(r for r in runs if r["retriever"] == "hybrid")
    assert deltas(dense, hybrid)["mrr"] == round(hybrid["overall"]["mrr"] - dense["overall"]["mrr"], 4)
