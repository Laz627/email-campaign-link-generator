# internal_link_recommender.py
"""
Smart Internal‑Link Finder (v2)
------------------------------
Upload a spreadsheet of your site’s pages, give the tool **two inputs**:

1. **Core concept / product keyword** (e.g. *"Pella Impervia"*)
2. **Email blurb / topic** (free‑form)

…and it returns the 5‑50 most relevant pages to link to, ranked by semantic
similarity and filtered so that every result actually contains the concept word
if you want it to.

### What changed in v2
* Stronger weighting for high‑click queries (1‑‑5 repeats instead of 1‑‑3).
* **Min‑similarity slider** – hide low‑confidence matches.
* **Concept filter** – checkbox forces the concept to appear in page text.
* Separate sidebar fields for *Concept* and *Email blurb* (concatenated for the
  query embedding), so users don’t forget to include the product keyword.

### Supported columns (case‑sensitive)
* URL
* Title
* Meta Description
* Top‑Level Subfolder
* (optional) Query #1/2/3 and their *_Clicks* companions

Run with:
```bash
streamlit run internal_link_recommender.py
```
"""

import io
import asyncio
from typing import List

import streamlit as st
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "text-embedding-3-large"
BASE_COLS = [
    "URL",
    "Title",
    "Meta Description",
    "Top-Level Subfolder",
]
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
    st.header("⚙️ Settings")

    # API & performance
    api_key = st.text_input("OpenAI API key", type="password")
    concurrency = st.slider("Parallel requests", 1, 50, 10)

    # Query inputs
    concept = st.text_input("🎯 Core concept / product", help="e.g. ‘Pella Impervia’")
    blurb = st.text_area("✏️ Email blurb / topic", height=120)

    # Options
    top_n = st.slider("Results to return", 1, 50, 10)
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.60, step=0.01)
    include_queries = st.checkbox("Add top search queries to embedding", value=True)
    must_contain_concept = st.checkbox("Only show pages containing concept", value=True)

    # Data upload
    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX of pages", type=["csv", "xlsx"], accept_multiple_files=False
    )

    run_btn = st.button("🚀 Find links")

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def combine_fields(row: pd.Series, enrich: bool = True) -> str:
    """Build the text we embed for each page."""
    parts: List[str] = [str(row["URL"],), str(row["Title"]), str(row["Meta Description"])]

    if enrich:
        total_clicks = (
            row.get("Query #1 Clicks", 0)
            + row.get("Query #2 Clicks", 0)
            + row.get("Query #3 Clicks", 0)
        ) or 1
        for q, c in [
            (row.get("Query #1", ""), row.get("Query #1 Clicks", 0)),
            (row.get("Query #2", ""), row.get("Query #2 Clicks", 0)),
            (row.get("Query #3", ""), row.get("Query #3 Clicks", 0)),
        ]:
            q = "" if (pd.isna(q) or not q) else str(q)
            if q:
                repeats = 1 + round(4 * c / total_clicks)  # stronger weighting (1‑5)
                parts.extend([q] * repeats)

    parts = [p for p in parts if isinstance(p, str) and p.strip()]
    return " | ".join(parts)


def load_dataframe(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        st.stop()
    for c in QUERY_COLS:
        if c not in df.columns:
            df[c] = "" if "Clicks" not in c else 0
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
                    await asyncio.sleep(2 ** attempt)

    return await asyncio.gather(*(_embed(t) for t in texts))


# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    # Basic validation
    if not api_key:
        st.error("Please paste your OpenAI API key.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload the pages spreadsheet.")
        st.stop()
    if not (concept or blurb):
        st.error("Enter at least a concept, a blurb, or both.")
        st.stop()

    # Build composite query text
    query_text = f"{concept.strip()} {blurb.strip()}".strip()

    df = load_dataframe(uploaded_file)

    # Folder filter (optional)
    folders = df["Top-Level Subfolder"].unique().tolist()
    chosen_folders = st.multiselect("Restrict to sub‑folders (optional)", folders, default=folders)

    df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)

    # Apply concept containment filter before ranking if user requests it
    if concept and must_contain_concept:
        mask = df["combined"].str.contains(concept, case=False, na=False)
        df = df[mask]
        if df.empty:
            st.warning("No pages contained the concept keyword – showing full set instead.")
            df = load_dataframe(uploaded_file)  # reset
            df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)

    # (Re)‑embed only if spreadsheet or enrich flag changed
    cache_key = (uploaded_file.name, include_queries)
    if st.session_state.get("embed_cache_key") != cache_key:
        with st.status("Embedding site pages…", expanded=False):
            client = AsyncOpenAI(api_key=api_key)
            embeds = asyncio.run(embed_texts(client, df["combined"].tolist(), concurrency))
        st.session_state["page_embeddings"] = np.array(embeds, dtype=np.float32)
        st.session_state["embed_cache_key"] = cache_key

    embeddings = st.session_state["page_embeddings"]

    # Embed the query
    with st.spinner("Embedding your query…"):
        client = AsyncOpenAI(api_key=api_key)
        q_vec = np.array(asyncio.run(embed_texts(client, [query_text], parallel=1))[0], dtype=np.float32)

    # Similarity & filtering
    df["similarity"] = cosine_similarity_matrix(embeddings, q_vec)
    df = df[df["Top-Level Subfolder"].isin(chosen_folders) & (df["similarity"] >= min_sim)]

    results = (
        df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    )

    # Display
    st.subheader("Top Matches")
    st.dataframe(
        results[[
            "URL",
            "Title",
            "Meta Description",
            "Top-Level Subfolder",
            "similarity",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # Download CSV
    csv_bytes = results.to_csv(index=False).encode()
    st.download_button("💾 Download CSV", csv_bytes, file_name="recommended_links.csv", mime="text/csv")

    st.success("Done! Matches filtered by concept, folder, and similarity threshold.")
