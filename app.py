import logging
from itertools import chain
from pathlib import Path

import streamlit as st

from local_rag.config import Settings
from local_rag.engine import RagEngine
from local_rag.errors import LocalRagError, NoSelectionError
from local_rag.ingest import load_chunks
from local_rag.retrievers import RETRIEVERS
from local_rag.store import DocumentStore

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

UPLOAD_DIR = "./uploads"

AVATARS = {
    "user": ":material/person:",
    "assistant": ":material/find_in_page:",
}


def source_columns(mode: str) -> dict:
    if mode == "semantic":
        score = st.column_config.ProgressColumn(
            "Relevance", min_value=0.0, max_value=1.0, format="%.2f"
        )
    else:
        score = st.column_config.NumberColumn("Score", format="%.2f")
    return {
        "Source": st.column_config.TextColumn("Source"),
        "Page": st.column_config.TextColumn("Page"),
        "Relevance": score,
    }


SCORE_EXPLANATIONS = {
    "semantic": (
        "Relevance: cosine similarity between question and chunk "
        "(0-1, higher is more similar)"
    ),
    "lexical": (
        "Score: BM25 keyword-match score (unbounded, higher means more matching terms)"
    ),
    "hybrid": (
        "Score: reciprocal rank fusion (RRF) combining the semantic and "
        "lexical rankings (unbounded, higher means it ranked well in both)"
    ),
    "hybrid_rerank": (
        "Score: cross-encoder relevance score "
        "(unbounded, can be negative; higher is more relevant)"
    ),
}


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


@st.cache_resource
def get_store(_settings: Settings) -> DocumentStore:
    return DocumentStore(_settings)


@st.cache_resource
def get_engine(_settings: Settings, _store: DocumentStore) -> RagEngine:
    return RagEngine(_settings, store=_store)


settings = get_settings()
store = get_store(settings)
engine = get_engine(settings, store)


def save_uploads(files) -> list[str]:
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for file in files:
        if file is not None:
            target = upload_dir / Path(file.name).name
            if not target.exists():
                target.write_bytes(file.getbuffer())
            paths.append(str(target))
    return paths


def handle_uploads(files):
    paths = save_uploads(files)
    count = 0
    for path in paths:
        try:
            filename = Path(path).name
            if not store.contains(path):
                store.add_chunks(chunks=load_chunks(settings, path))
                count += 1
            else:
                st.toast(
                    f"File: {filename} already exists in the vector store", icon="⚠️"
                )
        except LocalRagError as e:
            st.toast(f"{e!s}. File: {filename} didn't upload", icon="⚠️")
            store.delete_source(path)
            Path(path).unlink(missing_ok=True)
    st.toast(f"successfully embedded: {count} new file(s)")


st.set_page_config(
    page_title="local-RAG",
    page_icon="",
    layout="centered",
    initial_sidebar_state="locked",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "upload_round" not in st.session_state:
    st.session_state.upload_round = 0


with st.sidebar:
    st.subheader(":primary[local·RAG]")

    with st.container(border=True, gap=5):
        st.caption(":blue-background[**Vector Store**]")
        docs_col, chunks_col = st.columns(2)
        docs_col.metric(":gray[Documents]", store.count_sources())
        chunks_col.metric(":gray[Chunks]", store.count_chunks())

    with st.expander("Documents", expanded=True):
        staged = st.file_uploader(
            "Add documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploads_{st.session_state.upload_round}",
        )
        if st.button(
            "Embed staged files",
            icon=":material/library_add:",
            type="primary",
            disabled=not staged,
            help="Embeds the staged files and stores them in Chroma",
        ):
            handle_uploads(staged)
            st.session_state.upload_round += 1
            st.rerun()

        st.caption("All files stay on this machine")

    with st.expander("Mode"):
        st.radio(
            "Retrieval mode",
            list(RETRIEVERS),
            index=list(RETRIEVERS).index("hybrid"),
            label_visibility="collapsed",
            key="retrieval_mode",
        )

    with st.expander("Scope"), st.container(gap=None):
        sources = store.list_sources()
        files = [{"name": Path(source).name, "path": source} for source in sources]
        for file in files:
            st.session_state.setdefault(f"scope_{file['name']}", True)

        if files:
            all_selected = all(
                st.session_state[f"scope_{file['name']}"] for file in files
            )
            btn_label = "deselect all" if all_selected else "select all"
            if st.button(f":gray[{btn_label}]", type="tertiary"):
                for file in files:
                    st.session_state[f"scope_{file['name']}"] = not all_selected
                st.rerun()

            for file in files:
                pick_col, drop_col = st.columns(
                    [6, 1], vertical_alignment="center", gap=None
                )
                pick_col.checkbox(file["name"], key=f"scope_{file['name']}")
                if drop_col.button(
                    ":material/close:",
                    key=f"drop_{file['name']}",
                    type="tertiary",
                    help=f"Remove {file['name']} from the vector store",
                ):
                    try:
                        store.delete_source(file["path"])
                        Path(file["path"]).unlink(missing_ok=True)
                        st.toast(f"successfully deleted file: {file['name']}")
                        st.rerun()
                    except LocalRagError as e:
                        st.toast(str(e), icon="⚠️")

            if st.button(
                "Delete all",
                icon=":material/delete:",
                type="tertiary",
                help="Removes every embedded chunk from the vector store",
            ):
                try:
                    store.delete_all()
                    for file in files:
                        Path(file["path"]).unlink(missing_ok=True)
                    st.toast("cleared vector store")
                    st.rerun()
                except LocalRagError as e:
                    st.toast(str(e), icon="⚠️")
        else:
            st.caption("vector store is empty")

    with st.expander("System"):
        with st.container(gap=None):
            st.caption(":blue-background[**Models**]")
            st.caption(f":small[chat: {settings.ollama_chat_model}]")
            st.caption(f":small[embeddings: {settings.ollama_embeddings_model}]")
            st.caption(f":small[running at: {settings.ollama_base_url}]")

        with st.container(gap=None):
            st.caption(":blue-background[**Vector store**]")
            st.caption(":small[engine: Chroma]")
            st.caption(f":small[persist path: {settings.persist_dir}]")

        with st.container(gap=None):
            st.caption(":blue-background[**Chunking**]")
            st.caption(f":small[size: {settings.chunk_size} characters]")
            st.caption(f":small[overlap: {settings.chunk_overlap} characters]")


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            with st.expander("Sources"):
                st.dataframe(
                    message["sources"],
                    hide_index=True,
                    column_config=source_columns(message["mode"]),
                )
                st.caption(SCORE_EXPLANATIONS[message["mode"]])

if question := st.chat_input("Ask about your documents"):
    try:
        with st.chat_message(name="user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("thinking...", show_time=True):
                selected_sources = []
                for file in files:
                    if st.session_state[f"scope_{file['name']}"]:
                        selected_sources.append(file["path"])
                if st.session_state.retrieval_mode is None:
                    raise NoSelectionError("no retrieval mode selected")

                retrieved_chunks = engine.retrieve(
                    question, selected_sources, st.session_state.retrieval_mode
                )
                source_rows = []
                for chunk, score in retrieved_chunks:
                    page = chunk.metadata.get("page")
                    source_rows.append(
                        {
                            "Source": Path(chunk.metadata["source"]).name,
                            "Page": str(page)
                            if page is not None
                            else "not available for this file type",
                            "Relevance": score,
                        }
                    )
                answer = engine.stream_answer(question, retrieved_chunks)
                first_token = next((token for token in answer if token), "")

            response = st.write_stream(chain([first_token], answer))

        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": response,
                "sources": source_rows,
                "mode": st.session_state.retrieval_mode,
            }
        )
    except LocalRagError as e:
        st.toast(str(e), icon="⚠️")
    st.rerun()

if not st.session_state.messages:
    st.subheader("Ask your documents a question.")
    st.caption("Add files from the sidebar, then ask in your own words")
