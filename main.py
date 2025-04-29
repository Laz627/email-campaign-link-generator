# internal_link_recommender.py
"""
Smart Internal‑Link Finder
---------------------------------
Upload a spreadsheet of your site’s pages, paste an e‑mail topic (or any blurb),
and instantly get the 5‑10 most relevant URLs to link to.  

* Supports CSV **or** XLSX with these columns (case‑sensitive):
  - URL
  - Title
  - Meta Description
  - Top-Level Subfolder
  - (optional) Query #1, Query #1 Clicks, Query #2, Query #2 Clicks, Query #3, Query #3 Clicks
* Uses **text‑embedding‑3‑large** for high‑quality semantic matching
* Async concurrency slider (1–50) — 10–20 is usually safe under default rate limits
* Optional checkbox to enrich each page embedding with its top 3 search queries
* Cosine‑similarity ranking (vectorised NumPy)
* Download CSV of results

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
    api_key = st.text_input("OpenAI API key", type="password")
    concurrency = st.slider("Parallel requests", 1, 50, 10)
    top_n = st.slider("Results to return", 1, 50, 10)
    include_queries = st.checkbox("Add top search queries to embedding", value=True)
    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX of pages", type=["csv", "xlsx"], accept_multiple_files=False
    )
    query_text = st.text_area("Email topic / blurb", height=120)
    run_btn = st.button("🚀 Find links")

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def combine_fields(row: pd.Series, enrich: bool = True) -> str:
    """Create the text we send to the embedding model.
    Cast every element to *str* to avoid TypeErrors when NaN / floats appear.
    """
    parts: List[str] = [str(row["URL"]), str(row["Title"]), str(row["Meta Description"])]

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
                repeats = 1 + round(2 * c / total_clicks)  # 1–3 repeats
                parts.extend([q] * repeats)

    # keep only non-empty strings, ensure every element is str
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
    # ensure optional query columns exist
    for c in QUERY_COLS:
        if c not in df.columns:
            df[c] = "" if "Clicks" not in c else 0
    df = df.dropna(subset=BASE_COLS).reset_index(drop=True)
    return df


def cosine_similarity_matrix(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return (mat @ vec) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(vec))


async def embed_texts(client: AsyncOpenAI, texts: List[str], parallel: int) -> List[List[float]]:
    sem = asyncio.Semaphore(parallel)

    async def _embed(txt: str):
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.embeddings.create(model=MODEL_NAME, input=[txt])
                    return resp.data[0].embedding
                except OpenAIError as e:
                    if attempt == 4:
                        raise e
                    await asyncio.sleep(2 ** attempt)

    tasks = [_embed(t) for t in texts]
    return await asyncio.gather(*tasks)


# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not all([api_key, uploaded_file, query_text.strip()]):
        st.warning("Please provide an API key, upload a file, and enter a topic.")
        st.stop()

    df = load_dataframe(uploaded_file)

    # optional folder filter after load
    folders = df["Top-Level Subfolder"].unique().tolist()
    chosen_folders = st.multiselect(
        "Restrict to sub‑folders (optional)", options=folders, default=folders
    )

    df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)

    client = AsyncOpenAI(api_key=api_key)

    # cache embeddings per file + settings combo in session_state
    cache_key = (uploaded_file.name, include_queries)
    if st.session_state.get("embed_cache_key") != cache_key:
        with st.status("Embedding site pages…"):
            df_embeddings = asyncio.run(
                embed_texts(client, df["combined"].tolist(), concurrency)
            )
        st.session_state["page_embeddings"] = np.array(df_embeddings, dtype=np.float32)
        st.session_state["embed_cache_key"] = cache_key

    page_embeddings = st.session_state["page_embeddings"]

    # embed query
    with st.spinner("Embedding your topic…"):
        query_vec = np.array(
            asyncio.run(embed_texts(client, [query_text], parallel=1))[0], dtype=np.float32
        )

    # similarity search
    df["similarity"] = cosine_similarity_matrix(page_embeddings, query_vec)
    df_filtered = df[df["Top-Level Subfolder"].isin(chosen_folders)]
    results = (
        df_filtered.sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # display
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

    # download csv
    csv_bytes = results.to_csv(index=False).encode()
    st.download_button("💾 Download CSV", data=csv_bytes, mime="text/csv", file_name="recommended_links.csv")

    st.success("Done! Relevance ranked by cosine similarity in embedding space.")
