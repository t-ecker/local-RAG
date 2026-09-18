from itertools import chain
from pathlib import Path

import streamlit as st

from local_rag.config import Settings
from local_rag.errors import LocalRagError
from local_rag.ingest import load_chunks
from local_rag.query import RagEngine
from local_rag.store import DocumentStore

UPLOAD_DIR = "./uploads"

AVATARS = {
    "user": ":material/person:",
    "assistant": ":material/find_in_page:",
}

SOURCE_COLUMNS = {
    "Source": st.column_config.TextColumn("Source"),
    "Page": st.column_config.TextColumn("Page"),
    "Relevance": st.column_config.ProgressColumn(
        "Relevance", min_value=0.0, max_value=1.0, format="%.2f"
    ),
}


@st.cache_resource
def build_settings() -> Settings:
    return Settings()


@st.cache_resource
def build_vector_store(_settings: Settings) -> DocumentStore:
    return DocumentStore(_settings)


@st.cache_resource
def build_rag_engine(_settings: Settings, _store: DocumentStore) -> RagEngine:
    return RagEngine(_settings, document_store=_store)


settings = build_settings()
vector_store = build_vector_store(settings)
rag_engine = build_rag_engine(settings, vector_store)


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
            if not vector_store.is_present(path):
                vector_store.add_in_batches(chunks=load_chunks(settings, path))
                count += 1
            else:
                st.toast(
                    f"File: {filename} already exists in the vector store", icon="⚠️"
                )
        except LocalRagError as e:
            st.toast(f"{e!s}. File: {filename} didnt upload", icon="⚠️")
    st.toast(f"successfully embedded: {count} new file(s)")


def clear_vector_store() -> None:
    vector_store.delete()
    st.toast("cleared vector store")


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
        docs_col.metric(":gray[Documents]", vector_store.get_stored_document_amount())
        chunks_col.metric(":gray[Chunks]", vector_store.get_stored_chunk_amount())

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

        if st.button(
            "Delete all Documents",
            icon=":material/delete:",
            type="tertiary",
            help="Removes every embedded chunk from the vector store",
        ):
            try:
                clear_vector_store()
                st.rerun()
            except LocalRagError as e:
                st.toast(str(e), icon="⚠️")

    with st.expander("System"):
        with st.container(gap=None):
            st.caption(":blue-background[**Models**]")
            st.caption(f":small[chat: {settings.ollama_model}]")
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
                    message["metadata"],
                    hide_index=True,
                    column_config=SOURCE_COLUMNS,
                )

if question := st.chat_input("Ask about your documents"):
    try:
        with st.chat_message(name="user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("thinking...", show_time=True):
                retrieved_chunks = rag_engine.retrieve(question)
                chunks_metadata = []
                for chunk, score in retrieved_chunks:
                    page = chunk.metadata.get("page")
                    chunks_metadata.append(
                        {
                            "Source": Path(chunk.metadata["source"]).name,
                            "Page": str(page)
                            if page is not None
                            else "not available for this file type",
                            "Relevance": score,
                        }
                    )
                answer = rag_engine.stream_answer(question, retrieved_chunks)
                first_token = next((token for token in answer if token), "")

            response = st.write_stream(chain([first_token], answer))

        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": response,
                "metadata": chunks_metadata,
            }
        )
    except LocalRagError as e:
        st.toast(str(e), icon="⚠️")
    st.rerun()

if not st.session_state.messages:
    st.subheader("Ask your documents a question.")
    st.caption("Add files from the sidebar, then ask in your own words")
