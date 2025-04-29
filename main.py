# streamlit_app.py
"""
Self-service internal-link recommender for email campaigns
=========================================================
Upload a spreadsheet (CSV or XLSX) containing:
    • URL
    • Title
    • Meta Description
    • Top-Level Subfolder

The app builds OpenAI *text-embedding-3-large* vectors for each row,
then ranks pages by cosine similarity to a user-supplied topic/e-mail
snippet.  It supports thousands of pages via asyncio concurrency,
sub-folder filtering, and one-click CSV download of the top matches.
"""

import os
import io
import asyncio
from typing import List

import streamlit as st
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAIError


# ──────────────────────────────────────────────────────────────────────────
# UI – sidebar controls
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Internal-Link Recommender", layout="wide")

st.title("🔗 Internal-Link Recommender")

with st.sidebar:
    st.header("🔧 Settings")

    # API key
    openai_key = st.text_input("🔑 OpenAI API key", type="password")

    # concurrency limit
    max_concurrency = st.slider(
        "⏱️ Parallel requests", min_value=1, max_value=50, value=10, step=1
    )

    # top-N slider (5–10 typical, but make it flexible)
    top_n = st.slider("🔝 Results to return", min_value=1, max_value=50, value=10)

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📄 Upload URL spreadsheet (CSV or XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )

    # text query for e-mail topic / snippet
    query_text = st.text_area(
        "✏️ Paste the e-mail topic / body snippet you’re writing about",
        height=120,
    )

    run_button = st.button("🚀 Find relevant pages")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
MODEL_NAME = "text-embedding-3-large"


def combine_fields(url: str, title: str, description: str) -> str:
    """Text fed to the embedding model."""
    return f"{url.strip()} — {title.strip()} — {description.strip()}"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def embed_texts(
    client: AsyncOpenAI, texts: List[str], concurrency: int
) -> List[List[float]]:
    """
    Concurrently embed *texts* with an AsyncOpenAI client, respecting *concurrency*.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _embed(text: str):
        async with semaphore:
            for attempt in range(5):  # simple exponential-backoff retry
                try:
                    resp = await client.embeddings.create(
                        model=MODEL_NAME, input=[text]
                    )
                    return resp.data[0].embedding
                except OpenAIError as e:
                    await asyncio.sleep(2**attempt)
            raise RuntimeError(f"Repeated OpenAI error: {e}")

    tasks = [_embed(t) for t in texts]
    return await asyncio.gather(*tasks)


def prepare_dataframe(file: io.BytesIO | io.StringIO) -> pd.DataFrame:
    """
    Load CSV or XLSX and verify mandatory columns.
    """
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    required_cols = ["URL", "Title", "Meta Description", "Top-Level Subfolder"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing column(s): {', '.join(missing)}")
        st.stop()

    # Drop rows with any nulls in critical fields
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Combined text column for embedding
    df["combined"] = df.apply(
        lambda r: combine_fields(r["URL"], r["Title"], r["Meta Description"]), axis=1
    )

    return df


# ──────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────
if run_button:
    if not all([uploaded_file, query_text, openai_key]):
        st.warning("⬅️ Please provide an API key, a spreadsheet, and a query text.")
        st.stop()

    df = prepare_dataframe(uploaded_file)

    # Allow sub-folder filter *after* reading the file
    subfolders = df["Top-Level Subfolder"].unique().tolist()
    chosen_folders = st.multiselect(
        "📂 Restrict to sub-folders (optional)",
        options=subfolders,
        default=subfolders,
    )

    client = AsyncOpenAI(api_key=openai_key)

    # ── Embedding – dataset
    if "page_embeddings" not in st.session_state or st.session_state.get(
        "source_file"
    ) != uploaded_file.name:
        with st.status("Creating embeddings for uploaded pages…", expanded=False):
            embeddings = asyncio.run(
                embed_texts(client, df["combined"].tolist(), max_concurrency)
            )
        st.session_state["page_embeddings"] = np.array(embeddings, dtype=np.float32)
        st.session_state["source_file"] = uploaded_file.name

    page_embeddings = st.session_state["page_embeddings"]

    # ── Embedding – query
    with st.spinner("Embedding your query…"):
        q_emb = asyncio.run(
            embed_texts(client, [query_text], concurrency=1)
        )[0]  # single embedding

    q_emb = np.array(q_emb, dtype=np.float32)

    # ── Similarity search
    df["similarity"] = np.dot(page_embeddings, q_emb) / (
        np.linalg.norm(page_embeddings, axis=1) * np.linalg.norm(q_emb)
    )

    # Filter by chosen folders
    df_filtered = df[df["Top-Level Subfolder"].isin(chosen_folders)].copy()

    results = (
        df_filtered.sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    st.subheader("🎯 Top matches")
    st.dataframe(
        results[
            ["URL", "Title", "Meta Description", "Top-Level Subfolder", "similarity"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv_bytes = results.to_csv(index=False).encode()
    st.download_button(
        label="💾 Download CSV",
        data=csv_bytes,
        file_name="recommended_links.csv",
        mime="text/csv",
    )

    st.success("Done! (cosine similarity used for ranking.)")
