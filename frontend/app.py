"""Streamlit UI. Run: uv run streamlit run frontend/app.py  (API must be running)."""

from __future__ import annotations

import streamlit as st
from api_client import ApiClient, ApiError

st.set_page_config(page_title="Document QA", page_icon=None, layout="wide")
st.title("Evaluated Document QA")
st.caption("Answers are generated from retrieved passages. Always inspect the sources: citations do not guarantee correctness.")

api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000", key="api_url")
client = ApiClient(api_url)

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
    if st.button("Ingest", disabled=not files or health is None):
        messages: list[tuple[str, str]] = []
        for f in files:
            try:
                report = client.upload(f.name, f.getvalue())
                note = " (replaced previous version)" if report["replaced_existing"] else ""
                messages.append(("success", f"{report['name']}: {report['chunks']} chunks in {report['duration_ms']:.0f} ms{note}"))
                messages.extend(("warning", f"{report['name']}: {w}") for w in report["warnings"])
            except ApiError as exc:
                messages.append(("error", f"{f.name}: {exc}"))
        st.session_state["ingest_messages"] = messages
        st.rerun()  # refresh the status counts above; messages survive in session state
    for level, text in st.session_state.get("ingest_messages", []):
        getattr(st, level)(text)

tab_ask, tab_docs = st.tabs(["Ask", "Documents"])

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
            if result["abstained"]:
                st.warning("The model says the documents do not contain enough information to answer.")
            st.markdown("### Answer")
            st.write(result["answer"])
            if result["invalid_citations"]:
                st.error(f"Answer cites sources that do not exist: {result['invalid_citations']}. Treat it with suspicion.")
            elif not result["valid_citations"] and not result["abstained"]:
                st.warning("The answer contains no citations. It may not be grounded in the sources.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieval", f"{t['retrieval_ms']:.0f} ms")
            m2.metric("Generation", f"{t['generation_ms']:.0f} ms")
            u = result["usage"]
            m3.metric("Prompt tokens", u["prompt_tokens"] if u["prompt_tokens"] is not None else "n/a")
            m4.metric("Completion tokens", u["completion_tokens"] if u["completion_tokens"] is not None else "n/a")
        else:
            st.info(f"Retrieval only: {t['retrieval_ms']:.0f} ms")

        st.markdown("### Retrieved passages")
        st.caption("Score = cosine similarity between the question and the passage (higher = closer).")
        for s in result["sources"]:
            where = f"p.{s['page']}" if s["page"] is not None else "no page"
            tag = ("cited" if s["cited"] else "not cited") if not st.session_state.get("retrieve_only") else ""
            with st.expander(f"[{s['source_number']}] {s['doc_name']} ({where}) | score {s['score']:.3f} {tag}".strip(),
                             expanded=s["cited"]):
                if s["heading"]:
                    st.markdown(f"**Section:** {s['heading']}")
                st.text(s["text"])
                st.caption(f"chunk {s['chunk_id']} | position {s['ordinal']} | rank {s['rank']}")

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
