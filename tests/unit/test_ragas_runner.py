import pytest

from app.evaluation.dataset import DEFAULT_CORPUS_DIR, DEFAULT_DATASET_PATH, load_dataset
from app.evaluation.ragas_runner import RagasEvalError, build_ragas_scorers, run_ragas_eval, summarize
from app.generation import ABSTAIN_PHRASE
from tests.fixtures.fakes import FakeLLM


class Result:
    def __init__(self, value):
        self.value = value


class FakeScorer:
    def __init__(self, value=0.5, error=None):
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return Result(self.value)


@pytest.fixture
def loaded(make_service, corpus_files):
    def _build(llm):
        svc = make_service(llm)
        for name, data in corpus_files.items():
            svc.ingest(name, data)
        return svc
    return _build


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DEFAULT_DATASET_PATH)


def test_orchestration_inputs_and_deterministic_abstention(loaded, dataset):
    svc = loaded(FakeLLM("The answer is stated in the policy [1]."))
    scorers = {m: FakeScorer(0.75) for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    rows = run_ragas_eval(svc, dataset, scorers, top_k=3)
    assert len(rows) == 24
    n_answerable = sum(q.answerable for q in dataset.questions)
    assert all(len(s.calls) == n_answerable for s in scorers.values())  # must-refuse questions are not judged
    call = scorers["context_recall"].calls[0]
    assert set(call) == {"user_input", "retrieved_contexts", "reference"} and call["reference"]
    assert set(scorers["faithfulness"].calls[0]) == {"user_input", "response", "retrieved_contexts"}
    s = summarize(rows, list(scorers))
    assert s["ragas"]["faithfulness"] == {"mean": 0.75, "n_scored": n_answerable, "n_errors": 0}
    assert s["abstention"]["must_refuse"] == 6 and s["abstention"]["correctly_abstained"] == 0  # fake LLM never refuses
    assert s["tokens"]["prompt"] > 0


def test_refusals_are_counted_and_not_judged_for_faithfulness(loaded, dataset):
    svc = loaded(FakeLLM(ABSTAIN_PHRASE))
    faith, prec = FakeScorer(1.0), FakeScorer(0.5)
    rows = run_ragas_eval(svc, dataset, {"faithfulness": faith, "context_precision": prec})
    s = summarize(rows, ["faithfulness", "context_precision"])
    assert faith.calls == [] and s["ragas"]["faithfulness"]["n_scored"] == 0
    assert len(prec.calls) == 18 and s["ragas"]["context_precision"]["mean"] == 0.5  # retrieval-based, still judged
    assert s["abstention"] == {"must_refuse": 6, "correctly_abstained": 6, "answerable": 18, "false_abstentions": 18}


def test_scorer_failures_are_recorded_and_total_failure_raises(loaded, dataset):
    svc = loaded(FakeLLM("Fine [1]."))
    rows = run_ragas_eval(svc, dataset, {"context_recall": FakeScorer(error=RuntimeError("judge down"))})
    assert all("RuntimeError: judge down" in r.errors["context_recall"] for r in rows if r.answerable)
    with pytest.raises(RagasEvalError, match="judge down"):
        summarize(rows, ["context_recall"])
    mixed = {"context_recall": FakeScorer(0.4), "faithfulness": FakeScorer(error=ValueError("bad json"))}
    s = summarize(run_ragas_eval(svc, dataset, mixed), ["context_recall", "faithfulness"])
    assert s["ragas"]["context_recall"]["n_scored"] == 18 and s["ragas"]["faithfulness"]["n_errors"] == 18


def test_unknown_metric_and_llm_errors_propagate(loaded, dataset):
    from app.generation.llm import LLMError
    with pytest.raises(ValueError):
        run_ragas_eval(loaded(FakeLLM()), dataset, {"bogus": FakeScorer()})
    with pytest.raises(LLMError):
        run_ragas_eval(loaded(FakeLLM(error=LLMError("down"))), dataset, {"faithfulness": FakeScorer()})


def test_real_ragas_objects_construct_offline_and_validate_config():
    pytest.importorskip("ragas")
    scorers = build_ragas_scorers("judge-model", "http://localhost:11434/v1", None, "embed-model")
    assert set(scorers) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    with pytest.raises(RagasEvalError, match="embedding"):
        build_ragas_scorers("judge-model", "http://localhost:11434/v1", None, None)
