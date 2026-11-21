"""Streamlit UI. Run: uv run streamlit run frontend/app.py  (API must be running)."""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from api_client import ApiClient, ApiError
from benchmarks import comparison_rows, deltas, load_runs

st.set_page_config(page_title="Document QA", page_icon=None, layout="wide")
st.title("Evaluated Document QA")
st.caption("Answers are generated from retrieved passages. Always inspect the sources: citations do not guarantee correctness.")

api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000", key="api_url")
token = st.sidebar.text_input("API token (only if the server requires one)", type="password", key="api_token")
client = ApiClient(api_url, token=token or None)

# ----------------------------------------------------------------- sidebar: status + upload
with st.sidebar:
    st.subheader("Index status")
    try:
        health = client.health()
        st.success("API connected")
        st.write(f"**{health['documents']}** documents, **{health['indexed_chunks']}** chunks")
        st.caption(f"Embedding model: {health['embedding_model']}")
        default_k = int(health["default_top_k"])
    except ApiError as exc:
        st.error(str(exc))
        default_k = 4
        health = None

    st.subheader("Add documents")
    files = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    groups = st.text_input("Restrict to groups (comma-separated, blank = public)", key="upload_groups")
    if st.button("Ingest", disabled=not files or health is None):
        messages: list[tuple[str, str]] = []
        for f in files:
            try:
                report = client.upload(f.name, f.getvalue(), groups or None)
                note = {"unchanged": " (unchanged: skipped)", "updated": f" (new version {report['ingest_version']})"}.get(report["status"], "")
                messages.append(("success", f"{report['name']}: {report['chunks']} chunks in {report['duration_ms']:.0f} ms{note}"))
                messages.extend(("warning", f"{report['name']}: {w}") for w in report["warnings"])
            except ApiError as exc:
                messages.append(("error", f"{f.name}: {exc}"))
        st.session_state["ingest_messages"] = messages
        st.rerun()  # refresh the status counts above; messages survive in session state
    for level, text in st.session_state.get("ingest_messages", []):
        getattr(st, level)(text)

tab_ask, tab_docs, tab_bench = st.tabs(["Ask", "Documents", "Benchmarks"])

# ----------------------------------------------------------------- ask tab
with tab_ask:
    with st.form("ask_form"):
        question = st.text_input("Question", key="question")
        col1, col2 = st.columns([1, 2])
        top_k = col1.slider("Passages to retrieve", 1, 10, default_k)
        retrieve_only = col2.checkbox("Retrieve only (no LLM). Useful for inspecting evidence")
        submitted = st.form_submit_button("Ask")

    if submitted:
        try:
            with st.spinner("Working..."):
                result = client.retrieve(question, top_k) if retrieve_only else client.ask(question, top_k)
            st.session_state["result"] = result
            st.session_state["retrieve_only"] = retrieve_only
            st.session_state.pop("error", None)
        except ApiError as exc:
            st.session_state["error"] = str(exc)
            st.session_state.pop("result", None)

    if "error" in st.session_state:
        st.error(st.session_state["error"])

    result = st.session_state.get("result")
    if result:
        t = result["timings"]
        if not st.session_state.get("retrieve_only"):
            status = result.get("status", "answered")
            if status == "abstained_no_evidence":
                st.warning("Not answered: the retrieved passages share too little vocabulary with the question (no LLM call was made).")
            elif result["abstained"]:
                st.warning("The model says the documents do not contain enough information to answer.")
            for conflict in result.get("version_conflicts", []):
                st.warning(f"Several versions of one document were retrieved: {conflict}. Check which version the answer uses.")
            if result.get("injection_flags"):
                st.error("Some retrieved passages contain text that looks like instructions to the model (possible prompt injection). Inspect them below.")
            st.markdown("### Answer")
            st.write(result["answer"])
            if status == "withheld":
                st.error("The answer failed citation validation and was withheld by the server.")
            if result["invalid_citations"]:
                st.error(f"Answer cites sources that do not exist: {result['invalid_citations']}. Treat it with suspicion.")
            elif not result["valid_citations"] and not result["abstained"]:
                st.warning("The answer contains no citations. It may not be grounded in the sources.")
            elif result.get("uncited_sentences"):
                st.warning(f"{len(result['uncited_sentences'])} sentence(s) in the answer have no citation: " + " | ".join(result["uncited_sentences"][:3]))
            st.caption(f"Status: {status}. Citations are checked structurally (do the numbers exist?), not for whether the source supports the claim.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieval", f"{t['retrieval_ms']:.0f} ms")
            m2.metric("Generation", f"{t['generation_ms']:.0f} ms" if t["generation_ms"] is not None else "not called")
            u = result["usage"]
            m3.metric("Prompt tokens", u["prompt_tokens"] if u["prompt_tokens"] is not None else "n/a")
            m4.metric("Completion tokens", u["completion_tokens"] if u["completion_tokens"] is not None else "n/a")
        else:
            st.info(f"Retrieval only: {t['retrieval_ms']:.0f} ms")

        st.markdown("### Retrieved passages")
        st.caption("Score depends on the retriever: cosine similarity (dense), RRF score (hybrid) or reranker score (reranked). Compare only within one result list.")
        for s in result["sources"]:
            where = f"p.{s['page']}" if s["page"] is not None else "no page"
            tag = ("cited" if s["cited"] else "not cited") if not st.session_state.get("retrieve_only") else ""
            with st.expander(f"[{s['source_number']}] {s['doc_name']} ({where}) | score {s['score']:.3f} {tag}".strip(),
                             expanded=s["cited"]):
                if s["heading"]:
                    st.markdown(f"**Section:** {s['heading']}")
                st.text(s["text"])
                st.caption(f"chunk {s['chunk_id']} | position {s['ordinal']} | rank {s['rank']}")
                if result.get("injection_flags", {}).get(s["chunk_id"]):
                    st.error("Flagged: " + ", ".join(result["injection_flags"][s["chunk_id"]]))

# ----------------------------------------------------------------- documents tab
with tab_docs:
    try:
        docs = client.list_documents()
        if docs:
            st.dataframe(docs, width="stretch", hide_index=True)
        else:
            st.info("No documents yet. Upload some in the sidebar.")
    except ApiError as exc:
        st.error(str(exc))


# ----------------------------------------------------------------- benchmarks tab
with tab_bench:
    st.caption("Saved retrieval benchmark runs (docs/results and results/). Metrics are on answerable questions only.")
    runs = load_runs([Path("docs/results"), Path("results")])
    if not runs:
        st.info("No benchmark results found. Run: uv run python -m app.evaluation.benchmark --label my-run --embedder hashing")
    else:
        if any(r["stand_in"] for r in runs):
            st.warning("Runs marked stand-in used the offline hashing embedder or the lexical stand-in reranker. They test the "
                       "pipeline, not real model quality, and must not be quoted as such.")
        if all((r.get("n_human_verified") or 0) == 0 for r in runs):
            st.warning("No labels in these runs are human-verified (Claude-authored, 24 questions). Treat small differences as noise.")
        metric = st.selectbox("Metric to chart", ["mrr", "hit@1", "hit@5", "recall@5", "ndcg@5"], key="bench_metric")
        st.bar_chart({r["label"]: r["overall"].get(metric) for r in runs})
        st.dataframe(comparison_rows(runs), width="stretch", hide_index=True)
        labels = [r["file"] for r in runs]
        a, b = st.columns(2)
        base_file = a.selectbox("Baseline run", labels, key="bench_a")
        other_file = b.selectbox("Compare with", labels, index=len(labels) - 1, key="bench_b")
        base = next(r for r in runs if r["file"] == base_file)
        other = next(r for r in runs if r["file"] == other_file)
        st.write("Change (candidate minus baseline; positive = higher):")
        st.dataframe([deltas(base, other)], hide_index=True, width="stretch")
        st.caption("A random-ranking control for the same index is stored in each result file (`baselines.random`).")
