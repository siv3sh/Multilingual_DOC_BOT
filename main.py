import base64
import json
import os
import re
import time
from datetime import date, datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from llm_handler import GroqHandler
from rag_pipeline import RAGPipeline
from utils import (
    detect_language,
    extract_text,
    extract_glossary_terms,
    get_language_distribution,
    get_language_name,
)

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:  # pragma: no cover - optional dependency
    sanscript = None
    transliterate = None

# Load environment variables from .env (if present)
load_dotenv()

ANSWER_MODE_LABELS: Dict[str, str] = {
    "Concise": "concise",
    "Detailed with citations": "detailed_with_citations",
    "Bullet summary": "bullet_summary",
    "Step-by-step": "step_by_step",
}

SUPPORTED_FILE_TYPES = [
    "pdf",
    "docx",
    "txt",
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "gif",
    "tiff",
    "tif",
    "webp",
]
SECRET_KEYS = {"GROQ_API_KEY", "GROQ_MODEL", "QDRANT_URL", "QDRANT_COLLECTION_NAME"}
LANGUAGE_TO_SANSCRIPT = {
    "ta": getattr(sanscript, "TAMIL", None),
    "ml": getattr(sanscript, "MALAYALAM", None),
    "te": getattr(sanscript, "TELUGU", None),
    "kn": getattr(sanscript, "KANNADA", None),
    "hi": getattr(sanscript, "DEVANAGARI", None),
}


def process_uploaded_batch(uploaded_files: List[Any], pipeline: RAGPipeline) -> None:
    """Process a batch of uploaded files with progress indication."""
    total_files = len(uploaded_files)
    if total_files == 0:
        return

    progress_bar = st.progress(0.0)
    status_container = st.empty()

    new_documents: List[Dict[str, Any]] = []

    for idx, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state["processed_files"]:
            status_container.info(f"⏭️ Skipping `{uploaded_file.name}` (already ingested)")
            progress_bar.progress((idx + 1) / total_files)
            continue

        result = process_uploaded_document(
            uploaded_file=uploaded_file,
            pipeline=pipeline,
            progress_bar=progress_bar,
            status_container=status_container,
            index=idx,
            total=total_files,
        )

        if result:
            new_documents.append(result)
            st.session_state["processed_files"].append(result["filename"])

    # Merge document metadata
    existing_docs = {doc["filename"]: doc for doc in st.session_state["documents"]}
    for doc in new_documents:
        existing_docs[doc["filename"]] = doc

    st.session_state["documents"] = list(existing_docs.values())
    st.session_state["document_uploaded"] = len(st.session_state["documents"]) > 0

    available_langs = sorted({doc["language"] for doc in st.session_state["documents"]})
    if st.session_state["selected_languages"]:
        current = set(st.session_state["selected_languages"])
        updated = sorted(current.union(available_langs))
        st.session_state["selected_languages"] = updated
    else:
        st.session_state["selected_languages"] = available_langs

    available_sources = sorted({doc["filename"] for doc in st.session_state["documents"]})
    if st.session_state["selected_sources"]:
        current_sources = set(st.session_state["selected_sources"])
        st.session_state["selected_sources"] = sorted(current_sources.union(available_sources))
    else:
        st.session_state["selected_sources"] = available_sources

    if st.session_state["documents"]:
        upload_dates = [
            datetime.fromtimestamp(doc["uploaded_at"]).date()
            for doc in st.session_state["documents"]
        ]
        st.session_state["date_range"] = (min(upload_dates), max(upload_dates))

    if new_documents:
        glossary_index = {
            (entry["term"], entry["source"]): entry for entry in st.session_state["glossary"]
        }
        for doc in new_documents:
            for term_entry in doc.get("glossary_terms", []):
                key = (term_entry["term"], doc["filename"])
                glossary_index[key] = {
                    "term": term_entry["term"],
                    "definition": term_entry["definition"],
                    "language": doc["language"],
                    "source": doc["filename"],
                }
        st.session_state["glossary"] = list(glossary_index.values())

    if new_documents:
        status_container.success(f"📚 Imported {len(new_documents)} new document(s).")
    else:
        status_container.info("ℹ️ No new documents were added.")

    progress_bar.empty()


def generate_query_variants(
    query: str,
    language_filters: Optional[List[str]],
) -> Tuple[List[str], bool]:
    """Return query variants including transliterations for code-mixed input."""
    cleaned_query = re.sub(r'\s+', ' ', query.strip())
    variants = [cleaned_query] if cleaned_query else [query]
    transliteration_applied = False

    if not cleaned_query:
        return variants, transliteration_applied

    ascii_ratio = sum(1 for c in cleaned_query if ord(c) < 128) / len(cleaned_query)
    if (
        transliterate
        and language_filters
        and ascii_ratio > 0.6
    ):
        for lang in language_filters:
            target_scheme = LANGUAGE_TO_SANSCRIPT.get(lang)
            if not target_scheme:
                continue
            try:
                transliterated = transliterate(cleaned_query, sanscript.ITRANS, target_scheme)
            except Exception:
                continue
            if transliterated and transliterated not in variants:
                variants.append(transliterated)
                transliteration_applied = True

    return variants, transliteration_applied


def merge_context_results(
    existing: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
    max_items: int,
) -> List[Dict[str, Any]]:
    """Merge and deduplicate context results while keeping highest scoring entries."""
    context_map = {chunk['text']: chunk for chunk in existing}

    for chunk in new_results:
        key = chunk['text']
        if key in context_map:
            if chunk.get('score', 0) > context_map[key].get('score', 0):
                context_map[key] = chunk
        else:
            context_map[key] = chunk

    sorted_chunks = sorted(
        context_map.values(),
        key=lambda item: item.get('score', 0),
        reverse=True,
    )
    return sorted_chunks[:max_items]


def gather_context_chunks(
    pipeline: RAGPipeline,
    query_variants: List[str],
    top_k: int,
    language_filters: Optional[List[str]],
    source_filters: Optional[List[str]],
    date_range_ts: Optional[Tuple[float, float]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Retrieve contexts for multiple query variants and merge the results."""
    aggregated: List[Dict[str, Any]] = []
    transliteration_used = False

    for idx, variant in enumerate(query_variants):
        results = pipeline.retrieve_context(
            query=variant,
            top_k=top_k,
            language=language_filters,
            filenames=source_filters,
            date_range=date_range_ts,
        )
        if idx > 0 and results:
            transliteration_used = True
        if results:
            aggregated = merge_context_results(aggregated, results, max_items=top_k)
        if len(aggregated) >= top_k:
            break

    return aggregated, transliteration_used


def build_markdown_export(exchanges: List[Dict[str, Any]]) -> bytes:
    """Create a Markdown export for the selected exchanges."""
    lines: List[str] = ["# Document QA Export", ""]
    for idx, exchange in enumerate(exchanges, 1):
        lines.append(f"## Q{idx}: {exchange['question']}")
        lines.append("")
        lines.append(exchange['answer'])
        lines.append("")
        if exchange.get("contexts"):
            lines.append("**Sources**")
            for chunk in exchange["contexts"]:
                source = chunk.get("filename", "unknown")
                chunk_idx = chunk.get("chunk_index", "-")
                lines.append(f"- {source} (chunk #{chunk_idx})")
            lines.append("")
    content = "\n".join(lines)
    return content.encode("utf-8")


def sync_streamlit_secrets() -> None:
    """Promote Streamlit secrets into environment variables for compatibility."""
    if not hasattr(st, "secrets"):
        return

    try:
        secrets = st.secrets  # type: ignore[attr-defined]
        for key in SECRET_KEYS:
            value = secrets.get(key)  # type: ignore[attr-defined]
            if value and not os.getenv(key):
                os.environ[key] = str(value)
    except (StreamlitSecretNotFoundError, KeyError):
        return

    except Exception:
        # Silently ignore other issues to avoid crashes on startup
        return


def ensure_session_defaults() -> None:
    """Initialise session state fields used throughout the UI."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("document_language", None)
    st.session_state.setdefault("document_uploaded", False)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("processed_files", [])
    st.session_state.setdefault("selected_languages", [])
    st.session_state.setdefault("selected_sources", [])
    st.session_state.setdefault("date_range", None)
    st.session_state.setdefault("answer_mode", "Detailed with citations")
    st.session_state.setdefault("exchanges", [])
    st.session_state.setdefault("glossary", [])
    st.session_state.setdefault("share_mode", False)
    st.session_state.setdefault("share_applied", False)
    st.session_state.setdefault("translated_outputs", {})
    default_model = os.getenv("GROQ_MODEL", GroqHandler.get_available_models()[0])
    st.session_state.setdefault("groq_model", default_model)


def get_active_api_key() -> Optional[str]:
    """Return the API key from session or environment."""
    api_key = st.session_state.get("api_key") or os.getenv("GROQ_API_KEY")
    if api_key:
        api_key = api_key.strip()
    return api_key or None


def apply_api_configuration(api_key: str) -> None:
    """Persist API key changes and refresh cached resources."""
    clean_key = api_key.strip()
    st.session_state["api_key"] = clean_key
    os.environ["GROQ_API_KEY"] = clean_key
    get_cached_groq_handler.clear()


def create_share_token(payload: Dict[str, Any]) -> str:
    """Encode payload into a URL-safe share token."""
    json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(json_bytes).decode("utf-8")


def decode_share_token(token: str) -> Dict[str, Any]:
    """Decode a share token into a payload dictionary."""
    data = base64.urlsafe_b64decode(token.encode("utf-8"))
    return json.loads(data.decode("utf-8"))


def apply_share_view() -> None:
    """Initialise the session with shared content if a token is provided."""
    if st.session_state.get("share_applied"):
        return

    query_params = st.query_params
    token = query_params.get("share")
    if isinstance(token, list):
        token = token[0] if token else None
    if not token:
        return

    try:
        payload = decode_share_token(token)
    except Exception:
        st.warning("⚠️ Unable to open shared conversation. The link may be invalid.")
        st.session_state["share_applied"] = True
        return

    st.session_state["messages"] = payload.get("messages", [])
    st.session_state["exchanges"] = payload.get("exchanges", [])
    st.session_state["documents"] = payload.get("documents", [])
    st.session_state["glossary"] = payload.get("glossary", [])
    st.session_state["processed_files"] = [doc["filename"] for doc in st.session_state["documents"]]
    st.session_state["share_mode"] = True
    st.session_state["share_applied"] = True

    if st.session_state["documents"]:
        st.session_state["document_uploaded"] = True
        st.session_state["selected_languages"] = sorted({doc["language"] for doc in st.session_state["documents"]})
        st.session_state["selected_sources"] = [doc["filename"] for doc in st.session_state["documents"]]
        upload_dates = [
            datetime.fromtimestamp(doc["uploaded_at"]).date()
            for doc in st.session_state["documents"]
            if doc.get("uploaded_at")
        ]
        if upload_dates:
            st.session_state["date_range"] = (min(upload_dates), max(upload_dates))


@st.cache_resource(show_spinner=False)
def get_cached_rag_pipeline(qdrant_url: Optional[str], collection_name: str) -> RAGPipeline:
    """Return a cached RAG pipeline instance."""
    return RAGPipeline(qdrant_url=qdrant_url, collection_name=collection_name)


@st.cache_resource(show_spinner=False)
def get_cached_groq_handler(api_key: str, model: Optional[str]) -> GroqHandler:
    """Return a cached Groq handler bound to the provided credentials."""
    return GroqHandler(api_key=api_key, model=model)


def determine_document_language(
    raw_text: str, lang_counts: Dict[str, int]
) -> Tuple[str, Optional[str]]:
    """Resolve a language code and present a selection to the user when needed."""
    detected_code = detect_language(raw_text)
    selection_label = None

    if lang_counts and len(lang_counts) > 1:
        st.info("Multiple languages detected in the document. Please choose a language for answers.")
        options: List[Tuple[str, str]] = [
            (code, f"{get_language_name(code)} ({count} pages)")
            for code, count in sorted(lang_counts.items(), key=lambda item: item[1], reverse=True)
        ]
        labels = [label for _, label in options]
        default_index = next(
            (idx for idx, (code, _) in enumerate(options) if code == detected_code),
            0,
        )
        selected_label = st.selectbox("Select document language", labels, index=default_index)
        for code, label in options:
            if label == selected_label:
                detected_code = code
                selection_label = label
                break

    return detected_code, selection_label


def process_uploaded_document(
    uploaded_file,
    pipeline: RAGPipeline,
    progress_bar,
    status_container,
    index: int,
    total: int,
) -> Optional[Dict[str, Any]]:
    """Extract text, detect language, and index content into the vector store."""
    try:
        status_container.info(f"📄 Processing `{uploaded_file.name}`...")
        lang_counts = get_language_distribution(uploaded_file)
        uploaded_file.seek(0)
        text = extract_text(uploaded_file)

        if not text or not text.strip():
            st.warning(f"⚠️ No text extracted from `{uploaded_file.name}`. Skipping.")
            progress_bar.progress((index + 1) / total)
            return None

        lang_code, selection_label = determine_document_language(text, lang_counts)
        lang_name = get_language_name(lang_code)
        uploaded_ts = time.time()

        with st.spinner(f"Embedding `{uploaded_file.name}`..."):
            success = pipeline.process_document(
                        text=text,
                        filename=uploaded_file.name,
                language=lang_code,
                uploaded_at=uploaded_ts,
                extra_metadata={"original_filename": uploaded_file.name},
            )

        if not success:
            st.error(f"❌ Document processing failed for `{uploaded_file.name}`.")
            progress_bar.progress((index + 1) / total)
            return None

        status_container.success(
            f"✅ `{uploaded_file.name}` processed ({lang_name}, {len(text)} characters)"
        )
        progress_bar.progress((index + 1) / total)

        document_metadata = {
            "filename": uploaded_file.name,
            "language": lang_code,
            "language_label": lang_name,
            "uploaded_at": uploaded_ts,
            "text_length": len(text),
        }
        if selection_label:
            document_metadata["language_selection"] = selection_label

        glossary_terms = extract_glossary_terms(text, lang_code)
        if glossary_terms:
            document_metadata["glossary_terms"] = glossary_terms

        return document_metadata
    except Exception as exc:
        st.error(f"❌ Error processing `{uploaded_file.name}`: {exc}")
        progress_bar.progress((index + 1) / total)
        return None


def render_sidebar(read_only: bool = False) -> Dict[str, Any]:
    """Render sidebar controls and return configuration/state selections."""
    with st.sidebar:
        st.header("🔐 API Settings")
        api_key_input = st.text_input(
            "Groq API key",
            value=st.session_state.get("api_key", ""),
            type="password",
            help="The key is stored only in your current session.",
        )
        apply_clicked = st.button("Apply API key", use_container_width=True, disabled=read_only)

        if apply_clicked and not read_only:
            if api_key_input.strip():
                apply_api_configuration(api_key_input)
                st.success("API key applied successfully.")
            else:
                st.warning("Please enter a valid API key.")

        api_key = get_active_api_key()
        if api_key:
            st.caption("🔒 API key configured.")
        else:
            st.info("Provide a Groq API key to enable answer generation.")

        st.markdown("---")
        st.subheader("🤖 Model")

        model_options = GroqHandler.get_available_models()
        default_model = (
            st.session_state.get("groq_model")
            if st.session_state.get("groq_model") in model_options
            else os.getenv("GROQ_MODEL", model_options[0])
        )
        if default_model not in model_options:
            default_model = model_options[0]

        selected_model = st.selectbox(
            "Groq model",
            model_options,
            index=model_options.index(default_model),
            disabled=api_key is None,
        )

        if selected_model != st.session_state.get("groq_model"):
            st.session_state["groq_model"] = selected_model
            os.environ["GROQ_MODEL"] = selected_model
            if api_key:
                get_cached_groq_handler.clear()

        st.markdown("---")
        st.subheader("📄 Document Upload")
        uploaded_files = []
        if not read_only:
            uploaded_files = st.file_uploader(
                "Upload one or more documents (drag & drop files or folders)",
                type=SUPPORTED_FILE_TYPES,
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT, and common image types.",
            )
        else:
            st.info("Read-only mode: uploads disabled for shared view.")
        process_clicked = st.button(
            "Process selected documents",
            disabled=(not uploaded_files) or read_only,
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🧠 Answer style")
        answer_mode_label = st.selectbox(
            "Choose how answers should be formatted",
            list(ANSWER_MODE_LABELS.keys()),
            index=list(ANSWER_MODE_LABELS.keys()).index(st.session_state["answer_mode"]),
            disabled=read_only,
        )
        st.session_state["answer_mode"] = answer_mode_label

    st.markdown("---")
    st.subheader("🔍 Retrieval filters")
    documents = st.session_state["documents"]
    available_languages = sorted({doc["language"] for doc in documents})
    available_sources = sorted({doc["filename"] for doc in documents})

    if available_languages:
        default_langs = (
            st.session_state["selected_languages"]
            if st.session_state["selected_languages"]
            else available_languages
        )
        selected_languages = st.multiselect(
            "Languages",
            options=available_languages,
            default=default_langs,
            help="Filter retrieved chunks by language",
        )
    else:
        selected_languages = []

    if available_sources:
        default_sources = (
            st.session_state["selected_sources"]
            if st.session_state["selected_sources"]
            else available_sources
        )
        selected_sources = st.multiselect(
            "Sources",
            options=available_sources,
            default=default_sources,
            help="Restrict retrieval to specific documents",
        )
    else:
        selected_sources = []

    date_range = None
    if documents:
        upload_dates = [
            datetime.fromtimestamp(doc["uploaded_at"]).date() for doc in documents
        ]
        min_date = min(upload_dates)
        max_date = max(upload_dates)
        stored_range = st.session_state.get("date_range")
        default_range = (
            stored_range
            if stored_range
            else (min_date, max_date)
        )
        selected_date_range = st.date_input(
            "Upload date range",
            value=default_range,
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
            date_range = selected_date_range
        else:
            date_range = (selected_date_range, selected_date_range)
    else:
        selected_date_range = None

    st.session_state["selected_languages"] = selected_languages
    st.session_state["selected_sources"] = selected_sources
    st.session_state["date_range"] = date_range

    if st.button("🗑️ Clear chat history", use_container_width=True, disabled=read_only):
        st.session_state["messages"] = []
        st.session_state["exchanges"] = []
        st.experimental_rerun()

    return {
        "api_key": api_key,
        "selected_model": selected_model,
        "uploaded_files": uploaded_files or [],
        "process_clicked": process_clicked,
        "selected_languages": selected_languages,
        "selected_sources": selected_sources,
        "selected_date_range": date_range,
        "answer_mode": ANSWER_MODE_LABELS[answer_mode_label],
    }


def render_chat_interface(
    pipeline: Optional[RAGPipeline],
    handler: Optional[GroqHandler],
    language_filters: Optional[List[str]],
    source_filters: Optional[List[str]],
    date_range: Optional[Tuple[date, date]],
    answer_mode: str,
    read_only: bool,
) -> None:
    """Display chat history and handle new prompts."""
    exchanges = st.session_state.get("exchanges", [])
    translated_outputs = st.session_state.get("translated_outputs", {})
    assistant_msg_count = 0
    
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
            # Show source verification and translation for assistant messages
            if message["role"] == "assistant":
                # Find corresponding exchange
                if assistant_msg_count < len(exchanges):
                    exchange = exchanges[assistant_msg_count]
                    context_chunks = exchange.get("contexts", [])
                    response_language = exchange.get("language", "en")
                    
                    # Show source verification
                    if context_chunks:
                        st.markdown("---")
                        with st.expander("📄 View Source Documents & Context", expanded=False):
                            st.markdown("### Source Verification")
                            
                            # Group chunks by document
                            doc_sources = {}
                            for chunk in context_chunks:
                                filename = chunk.get("filename", "Unknown")
                                if filename not in doc_sources:
                                    doc_sources[filename] = []
                                doc_sources[filename].append(chunk)
                            
                            # Show document sources
                            st.markdown(f"**📚 Documents Used:** {len(doc_sources)} document(s)")
                            for idx, (filename, chunks) in enumerate(doc_sources.items(), 1):
                                st.markdown(f"**{idx}. {filename}**")
                                st.markdown(f"   - Chunks used: {len(chunks)}")
                                for chunk in chunks:
                                    chunk_idx = chunk.get("chunk_index", "?")
                                    score = chunk.get("score", 0)
                                    st.markdown(f"   - Chunk #{chunk_idx} (relevance: {score:.3f})")
                            
                            st.markdown("---")
                            st.markdown("### Retrieved Context Chunks")
                            
                            # Show actual context text
                            for idx, chunk in enumerate(context_chunks, 1):
                                filename = chunk.get("filename", "Unknown")
                                chunk_idx = chunk.get("chunk_index", "?")
                                chunk_text = chunk.get("text", "")
                                score = chunk.get("score", 0)
                                
                                with st.expander(f"Chunk {idx}: {filename} (Chunk #{chunk_idx}, Score: {score:.3f})", expanded=False):
                                    st.markdown(f"**Source:** `{filename}` | **Chunk Index:** `{chunk_idx}` | **Relevance Score:** `{score:.3f}`")
                                    st.markdown("---")
                                    st.markdown("**Context Text:**")
                                    st.text_area(
                                        "",
                                        value=chunk_text,
                                        height=150,
                                        disabled=True,
                                        key=f"context_hist_{assistant_msg_count}_{idx}",
                                        label_visibility="collapsed",
                                    )
                    else:
                        st.warning("⚠️ **No source documents found.** This answer may not be based on your uploaded documents.")
                    
                    # Show translation button for assistant messages that are not in English
                    if handler and not read_only and response_language != "en":
                        translation_key = f"translation_{assistant_msg_count}"
                        cached_translation = translated_outputs.get(translation_key)
                        
                        if cached_translation:
                            st.markdown("---")
                            st.markdown("**English Translation:**")
                            st.markdown(cached_translation)
                            if st.button("🔄 Hide translation", key=f"hide_trans_{translation_key}_{assistant_msg_count}"):
                                translated_outputs.pop(translation_key, None)
                                st.session_state["translated_outputs"] = translated_outputs
                                st.rerun()
                        else:
                            if st.button("🌐 Translate to English", key=f"translate_{translation_key}_{assistant_msg_count}"):
                                with st.spinner("Translating to English..."):
                                    translated_text = handler.translate_text(
                                        message["content"],
                                        target_language="English",
                                    )
                                    if translated_text:
                                        translated_outputs[translation_key] = translated_text
                                        st.session_state["translated_outputs"] = translated_outputs
                                        st.rerun()
                                    else:
                                        st.error("Translation failed. Please try again.")
                assistant_msg_count += 1
    
    if read_only:
        st.info("📄 This is a read-only shared view. Chat input is disabled.")
        return

    if pipeline is None or handler is None:
        st.error("❌ Retrieval pipeline is not available.")
        return

    prompt = st.chat_input("Ask a question about the document...")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                date_range_ts = None
                if date_range:
                    start_dt = datetime.combine(date_range[0], datetime.min.time())
                    end_dt = datetime.combine(date_range[1], datetime.max.time())
                    date_range_ts = (start_dt.timestamp(), end_dt.timestamp())

                # Auto-detect query language if no language filter is selected
                effective_language_filters = language_filters
                if not effective_language_filters:
                    detected_query_lang = detect_language(prompt)
                    if detected_query_lang and detected_query_lang != "en":
                        # Use detected language as filter to prevent cross-language retrieval
                        effective_language_filters = [detected_query_lang]
                        st.caption(f"🔍 Auto-detected query language: {get_language_name(detected_query_lang)}. Filtering results to this language.")
                
                query_variants, transliteration_applied = generate_query_variants(
                    prompt,
                    effective_language_filters or language_filters,
                )

                context_chunks, transliteration_used = gather_context_chunks(
                    pipeline=pipeline,
                    query_variants=query_variants,
                    top_k=3,
                    language_filters=effective_language_filters or language_filters,
                    source_filters=source_filters,
                    date_range_ts=date_range_ts,
                )

                # Post-retrieval filter: Remove chunks that don't match the query language
                if effective_language_filters and context_chunks:
                    filtered_chunks = [
                        chunk for chunk in context_chunks
                        if chunk.get('language') in effective_language_filters
                    ]
                    if filtered_chunks:
                        context_chunks = filtered_chunks
                    elif context_chunks:
                        # If all chunks were filtered out but we have chunks, warn but keep them
                        st.warning("⚠️ Some retrieved chunks don't match the query language. Showing all results.")
                
                if not context_chunks:
                    st.warning("⚠️ No relevant context found. The answer may be less accurate.")
                elif transliteration_used and transliteration_applied:
                    st.caption("🔁 Used transliterated query variant to improve retrieval.")

                response_language = (
                    (effective_language_filters or language_filters)[0]
                    if (effective_language_filters or language_filters)
                    else (context_chunks[0]['language'] if context_chunks else "en")
                )

                answer = handler.generate_answer(
                    query=prompt,
                    context_chunks=context_chunks,
                    language=response_language,
                    answer_mode=answer_mode,
                )

                st.markdown(answer)
                    
                # Show source verification
                if context_chunks:
                    st.markdown("---")
                    with st.expander("📄 View Source Documents & Context", expanded=False):
                        st.markdown("### Source Verification")
                        
                        # Group chunks by document
                        doc_sources = {}
                        for chunk in context_chunks:
                            filename = chunk.get("filename", "Unknown")
                            if filename not in doc_sources:
                                doc_sources[filename] = []
                            doc_sources[filename].append(chunk)
                        
                        # Show document sources
                        st.markdown(f"**📚 Documents Used:** {len(doc_sources)} document(s)")
                        for idx, (filename, chunks) in enumerate(doc_sources.items(), 1):
                            st.markdown(f"**{idx}. {filename}**")
                            st.markdown(f"   - Chunks used: {len(chunks)}")
                            for chunk in chunks:
                                chunk_idx = chunk.get("chunk_index", "?")
                                score = chunk.get("score", 0)
                                st.markdown(f"   - Chunk #{chunk_idx} (relevance: {score:.3f})")
                        
                        st.markdown("---")
                        st.markdown("### Retrieved Context Chunks")
                        
                        # Show actual context text
                        for idx, chunk in enumerate(context_chunks, 1):
                            filename = chunk.get("filename", "Unknown")
                            chunk_idx = chunk.get("chunk_index", "?")
                            chunk_text = chunk.get("text", "")
                            score = chunk.get("score", 0)
                            
                            with st.expander(f"Chunk {idx}: {filename} (Chunk #{chunk_idx}, Score: {score:.3f})", expanded=False):
                                st.markdown(f"**Source:** `{filename}` | **Chunk Index:** `{chunk_idx}` | **Relevance Score:** `{score:.3f}`")
                                st.markdown("---")
                                st.markdown("**Context Text:**")
                                st.text_area(
                                    "",
                                    value=chunk_text,
                                    height=150,
                                    disabled=True,
                                    key=f"context_{len(st.session_state.get('exchanges', []))}_{idx}",
                                    label_visibility="collapsed",
                                )
                else:
                    st.warning("⚠️ **No source documents found.** This answer may not be based on your uploaded documents.")
                
                # Store exchange first to get the correct index
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                exchange = {
                    "question": prompt,
                    "answer": answer,
                    "contexts": context_chunks,
                    "language": response_language,
                    "timestamp": time.time(),
                }
                st.session_state["exchanges"].append(exchange)
                
                # Add "Translate to English" button if answer is not in English
                if response_language != "en" and handler:
                    exchange_idx = len(st.session_state.get("exchanges", [])) - 1
                    translation_key = f"translation_{exchange_idx}"
                    translated_outputs = st.session_state.setdefault("translated_outputs", {})
                    cached_translation = translated_outputs.get(translation_key)
                    
                    if cached_translation:
                        st.markdown("---")
                        st.markdown("**English Translation:**")
                        st.markdown(cached_translation)
                        if st.button("🔄 Hide translation", key=f"hide_trans_{translation_key}"):
                            translated_outputs.pop(translation_key, None)
                            st.session_state["translated_outputs"] = translated_outputs
                            st.rerun()
                    else:
                        if st.button("🌐 Translate to English", key=f"translate_{translation_key}"):
                            with st.spinner("Translating to English..."):
                                translated_text = handler.translate_text(
                                    answer,
                                    target_language="English",
                                )
                                if translated_text:
                                    translated_outputs[translation_key] = translated_text
                                    st.session_state["translated_outputs"] = translated_outputs
                                    st.rerun()
                                else:
                                    st.error("Translation failed. Please try again.")
            except Exception as exc:
                error_msg = f"❌ Error generating response: {exc}"
                st.error(error_msg)
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})


def render_export_controls(read_only: bool) -> None:
    """Provide export/share utilities."""
    st.subheader("📤 Export & Share")
    exchanges = st.session_state.get("exchanges", [])
    glossary = st.session_state.get("glossary", [])

    if not exchanges:
        st.info("Run at least one query to enable exports and sharing.")
        return

    options = {
        f"Q{idx + 1}: {exchange['question'][:60]}": idx
        for idx, exchange in enumerate(exchanges)
    }
    default_selection = list(options.keys())[-1:]
    selected_labels = st.multiselect(
        "Select question/answer pairs to export",
        list(options.keys()),
        default=default_selection,
    )

    if selected_labels:
        selected_indices = [options[label] for label in selected_labels]
        selected_exchanges = [exchanges[i] for i in selected_indices]
        markdown_bytes = build_markdown_export(selected_exchanges)
        st.download_button(
            "Download Markdown",
            data=markdown_bytes,
            file_name="doc_llm_export.md",
            mime="text/markdown",
        )

    share_payload = {
        "messages": st.session_state["messages"],
        "exchanges": exchanges,
        "documents": st.session_state["documents"],
        "glossary": glossary,
    }
    share_token = create_share_token(share_payload)
    share_url = f"?share={share_token}"
    st.text_input(
        "Shareable link",
        value=share_url,
        help="Copy and share this link for a read-only view.",
        disabled=True,
    )
    if read_only:
        st.caption("You are viewing a shared conversation. Interactive features are disabled.")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="South Indian Multilingual QA Chatbot",
        page_icon="🌏",
        layout="wide",
    )

    sync_streamlit_secrets()
    ensure_session_defaults()
    apply_share_view()

    st.title("🌏 South Indian Multilingual Document QA Chatbot")
    st.markdown("Upload documents in **Malayalam, Tamil, Telugu, Kannada, or Tulu** and ask questions!")

    read_only = st.session_state.get("share_mode", False)
    sidebar_state = render_sidebar(read_only=read_only)
    api_key = sidebar_state["api_key"]
    selected_model = sidebar_state["selected_model"]

    pipeline: Optional[RAGPipeline] = None
    if not read_only:
        try:
            pipeline = get_cached_rag_pipeline(
                os.getenv("QDRANT_URL"),
                os.getenv("QDRANT_COLLECTION_NAME", "multilingual_docs"),
            )
        except Exception as exc:
            st.error(f"❌ Failed to initialise processing pipeline: {exc}")
            return

        if sidebar_state["process_clicked"]:
            if not api_key:
                st.error("A Groq API key is required before processing documents.")
            elif sidebar_state["uploaded_files"]:
                process_uploaded_batch(sidebar_state["uploaded_files"], pipeline)

    st.markdown("---")
    st.subheader("📊 Workspace status")
    documents = sorted(
        st.session_state["documents"],
        key=lambda d: d["uploaded_at"],
        reverse=True,
    )

    if documents:
        st.write(f"✅ {len(documents)} document(s) indexed")
        summary_rows = []
        for doc in documents:
            summary_rows.append(
                {
                    "Document": doc["filename"],
                    "Language": get_language_name(doc["language"]),
                    "Uploaded": datetime.fromtimestamp(doc["uploaded_at"]).strftime("%Y-%m-%d %H:%M"),
                    "Characters": doc["text_length"],
                }
            )
        st.dataframe(summary_rows, use_container_width=True)
    else:
        st.write("⏳ No documents indexed yet.")

    with st.expander("📘 Workspace glossary", expanded=False):
        glossary = st.session_state.get("glossary", [])
        if not glossary:
            st.write("No glossary entries yet.")
        else:
            term_search = st.text_input(
                "Search glossary terms",
                value="",
                key="glossary_search",
            ).strip().lower()
            filtered_glossary = [
                entry for entry in glossary
                if not term_search or term_search in entry["term"].lower()
            ]
            st.table(filtered_glossary)

    if not documents:
        st.info("👆 Upload documents from the sidebar to begin!")
        return

    if read_only:
        render_chat_interface(
            pipeline=pipeline,
            handler=None,  # type: ignore[arg-type]
            language_filters=sidebar_state["selected_languages"] or None,
            source_filters=sidebar_state["selected_sources"] or None,
            date_range=sidebar_state["selected_date_range"],
            answer_mode=sidebar_state["answer_mode"],
            read_only=True,
        )
        render_export_controls(read_only=True)
        st.markdown("---")
        st.markdown(
            "Built with ❤️ using Streamlit, Qdrant, and Groq | Supports Malayalam, Tamil, Telugu, Kannada, and Tulu",
        )
        return

    if not api_key:
        st.warning("Provide a valid Groq API key to ask questions about your documents.")
        return

    try:
        handler = get_cached_groq_handler(api_key, selected_model)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        return
    except Exception as exc:
        st.error(f"❌ Failed to initialise Groq handler: {exc}")
        return

    render_chat_interface(
        pipeline=pipeline,
        handler=handler,
        language_filters=sidebar_state["selected_languages"] or None,
        source_filters=sidebar_state["selected_sources"] or None,
        date_range=sidebar_state["selected_date_range"],
        answer_mode=sidebar_state["answer_mode"],
        read_only=False,
    )

    render_export_controls(read_only=False)

st.markdown("---")
st.markdown(
        "Built with ❤️ using Streamlit, Qdrant, and Groq | Supports Malayalam, Tamil, Telugu, Kannada, and Tulu",
)


if __name__ == "__main__":
    main()
