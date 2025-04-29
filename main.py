# internal_link_recommender.py  – v2.1
"""
Smart Internal‑Link Finder
=========================
Upload a spreadsheet of site pages, supply:

1. **Core concept / product keyword** – e.g. *Pella Impervia*
2. **Email blurb / topic** – free‑form text

…and get the most relevant URLs ranked by semantic similarity.

Fixes in **v2.1**
-----------------
* **Bug‑fix:** embeddings cache now keyed by *row‑count* as well – prevents the
  length‑mismatch ValueError when the concept filter shrinks the DataFrame.
* **Bug‑fix:** `str(row["URL"])` typo corrected.
* If cached embeddings length ≠ current DataFrame length, they are
  regenerated automatically.
* Code tidied but behaviour unchanged.

Run:
```bash
streamlit run internal_link_recommender.py
```
"""

import asyncio
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "text-embedding-3-large"
BASE_COLS = ["URL", "Title", "Meta Description", "Top-Level Subfolder"]
QUERY_COLS = [
    "Query #1",
    "Query #1 Clicks",
    "Query #2",
    "Query #2 Clicks",
    "Query #3",
    "Query #3 Clicks",
]

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔗 Smart Internal‑Link Finder", layout="wide")
st.title("🔗 Smart Internal‑Link Finder")

with st.sidebar:
    st.header("⚙️  Settings")
    api_key = st.text_input("OpenAI API key", type="password")
    concurrency = st.slider("Parallel requests", 1, 50, 10)

    # Query inputs
    concept = st.text_input("🎯 Core concept / product", help="e.g. ‘Pella Impervia’")
    blurb = st.text_area("✏️ Email blurb / topic", height=120)

    top_n = st.slider("Results to return", 1, 50, 10)
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.60, 0.01)
    include_queries = st.checkbox("Add top search queries to embedding", value=True)
    must_contain_concept = st.checkbox("Only show pages containing concept", value=True)

    uploaded_file = st.file_uploader("Upload CSV or XLSX", ["csv", "xlsx"])
    run_btn = st.button("🚀 Find links")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def combine_fields(row: pd.Series, enrich: bool = True) -> str:
    parts: List[str] = [
        str(row["URL"]),
        str(row["Title"]),
        str(row["Meta Description"]),
    ]

    if enrich:
        total_clicks: int = (
            row.get("Query #1 Clicks", 0)
            + row.get("Query #2 Clicks", 0)
            + row.get("Query #3 Clicks", 0)
        ) or 1
        for q, c in [
            (row.get("Query #1", ""), row.get("Query #1 Clicks", 0)),
            (row.get("Query #2", ""), row.get("Query #2 Clicks", 0)),
            (row.get("Query #3", ""), row.get("Query #3 Clicks", 0)),
        ]:
            if isinstance(q, float) or pd.isna(q):
                q = ""
            q = str(q).strip()
            if q:
                repeats = 1 + round(4 * c / total_clicks)  # 1–5 repeats
                parts.extend([q] * repeats)

    return " | ".join(filter(None, parts))


def load_dataframe(file) -> pd.DataFrame:
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()
    for c in QUERY_COLS:
        if c not in df.columns:
            df[c] = 0 if "Clicks" in c else ""
    return df.dropna(subset=BASE_COLS).reset_index(drop=True)


def cosine_similarity_matrix(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return (mat @ vec) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(vec))


async def embed_texts(client: AsyncOpenAI, texts: List[str], parallel: int) -> List[List[float]]:
    sem = asyncio.Semaphore(parallel)

    async def _embed(txt: str):
        async with sem:
            for attempt in range(5):
                try:
                    r = await client.embeddings.create(model=MODEL_NAME, input=[txt])
                    return r.data[0].embedding
                except OpenAIError as e:
                    if attempt == 4:
                        raise e
                    await asyncio.sleep(2**attempt)

    return await asyncio.gather(*(_embed(t) for t in texts))


def get_cache_key(filename: str, enrich: bool, df_len: int) -> Tuple[str, bool, int]:
    return (filename, enrich, df_len)

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Please enter your OpenAI API key.")
        st.stop()
    if uploaded_file is None:
        st.error("Please upload the spreadsheet first.")
        st.stop()
    if not (concept or blurb):
        st.error("Enter a concept, a blurb, or both.")
        st.stop()

    # Compose query
    query_text = f"{concept} {blurb}".strip()

    # Read and prep DataFrame
    df = load_dataframe(uploaded_file)

    # Sub‑folder filter
    folders = df["Top-Level Subfolder"].unique().tolist()
    chosen_folders = st.multiselect("Restrict to sub‑folders (optional)", folders, default=folders)

    df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)

    # Concept containment (pre‑embed)
    if concept and must_contain_concept:
        mask = df["combined"].str.contains(concept, case=False, na=False)
        df = df[mask]
        if df.empty:
            st.warning("No pages contained the concept keyword – initial filter disabled.")
            df = load_dataframe(uploaded_file)
            df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)

    # Embedding stage with cache keyed by len(df)
    cache_key = get_cache_key(uploaded_file.name, include_queries, len(df))
    if st.session_state.get("embed_cache_key") != cache_key:
        with st.status("Embedding site pages…", expanded=False):
            client = AsyncOpenAI(api_key=api_key)
            vectors = asyncio.run(embed_texts(client, df["combined"].tolist(), concurrency))
        st.session_state["page_embeddings"] = np.array(vectors, dtype=np.float32)
        st.session_state["embed_cache_key"] = cache_key

    embeddings: np.ndarray = st.session_state["page_embeddings"]

    # Safety check for length mismatch (edge‑case)
    if len(embeddings) != len(df):
        client = AsyncOpenAI(api_key=api_key)
        embeddings = np.array(asyncio.run(embed_texts(client, df["combined"].tolist(), concurrency)), dtype=np.float32)
        st.session_state["page_embeddings"] = embeddings

    # Embed query
    client = AsyncOpenAI(api_key=api_key)
    q_vec = np.array(asyncio.run(embed_texts(client, [query_text], parallel=1))[0], dtype=np.float32)

    df["similarity"] = cosine_similarity_matrix(embeddings, q_vec)

    # Post‑filters
    df = df[df["Top-Level Subfolder"].isin(chosen_folders) & (df["similarity"] >= min_sim)]

    results = df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)

    # Output
    st.subheader("Top Matches")
    st.dataframe(
        results[["URL", "Title", "Meta Description", "Top-Level Subfolder", "similarity"]],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = results.to_csv(index=False).encode()
    st.download_button("💾 Download CSV", csv_bytes, "recommended_links.csv", "text/csv")

    st.success("Done! Matches filtered and ranked by similarity.")
