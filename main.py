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


def get_relevance_indicator(score: float) -> str:
    if score >= 0.85:
        return "🟢 Excellent match"
    elif score >= 0.75:
        return "🟡 Good match"
    else:
        return "🟠 Relevant"

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Content Recommendation Engine", layout="wide")

# Header and description
st.title("📚 Content Recommendation Engine")
st.markdown("""
Discover relevant content based on topics you're interested in. This AI-powered tool analyzes your content 
library and recommends the most semantically similar items to your specified theme.
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API key", type="password", help="Required for semantic analysis")
    
    st.subheader("Search Parameters")
    min_sim = st.slider("Minimum relevance score", 0.0, 1.0, 0.65, 0.01, 
                      help="Higher = more relevant but potentially fewer results")
    include_queries = st.checkbox("Include search behavior data", value=True, 
                               help="Considers popular search queries when matching content")
    must_contain_topic = st.checkbox("Content must contain topic keyword", value=False)
    
    with st.expander("Advanced Settings"):
        concurrency = st.slider("Parallel API requests", 1, 50, 10, 
                              help="Higher values may process faster but use more API credits")
        top_n = st.slider("Maximum results to show", 5, 50, 15)

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.header("1️⃣ Upload Content Library")
    st.markdown("""
    Upload a spreadsheet with your content. Required columns:
    - **URL**: Link to the content
    - **Title**: Content title
    - **Meta Description**: Brief description
    - **Top-Level Subfolder**: Category or section
    
    Optional search data columns are also supported.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV or XLSX", ["csv", "xlsx"])
    
    if uploaded_file:
        st.success("✅ Content library uploaded!")
        
        st.header("2️⃣ Specify Your Interest")
        topic = st.text_input("What topic are you interested in?", 
                            placeholder="e.g., sustainable gardening, home renovation")
        
        details = st.text_area("Additional details (optional)", 
                             placeholder="Provide more specific information about what you're looking for...",
                             height=100)
        
        if topic:
            find_btn = st.button("🔍 Find Recommendations", type="primary", use_container_width=True)
        else:
            st.info("Please enter a topic to continue")
            find_btn = False

with col2:
    if not uploaded_file:
        st.header("How It Works")
        
        st.markdown("""
        ### 1. Upload Your Content Library
        Start by uploading a spreadsheet containing your content data. The recommendation engine needs 
        information about your content to find the best matches.
        
        ### 2. Specify Your Topic of Interest
        Tell the system what you're interested in. The more specific you are, the better your 
        recommendations will be.
        
        ### 3. Review Personalized Recommendations
        The AI will analyze your content and find the most relevant pieces based on your 
        specified topic, ranked by semantic similarity.
        """)
        
        # Example content format
        with st.expander("Sample Spreadsheet Format", expanded=True):
            example_data = {
                "URL": ["https://example.com/page1", "https://example.com/page2"],
                "Title": ["Beginner's Guide to Gardening", "Energy-Efficient Home Improvements"],
                "Meta Description": ["Learn the basics of sustainable gardening", "Cost-effective ways to improve home energy efficiency"],
                "Top-Level Subfolder": ["Gardening", "Home Improvement"],
                "Query #1": ["gardening tips", "energy saving"],
                "Query #1 Clicks": [120, 89]
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file and 'find_btn' in locals() and find_btn:
    # Input validation
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    # Display the recommendations section
    st.header(f"📊 Recommendations for: {topic}")
    if details:
        st.caption(f"Additional context: {details}")
    
    # Read and prep DataFrame
    df = load_dataframe(uploaded_file)
    
    # Sub‑folder filter
    folders = df["Top-Level Subfolder"].unique().tolist()
    chosen_folders = st.multiselect("Filter by category (optional)", folders, default=folders)
    
    with st.status("Finding relevant content...", expanded=True) as status:
        # Compose query
        query_text = f"{topic} {details}".strip()
        
        status.update(label="Processing content data...")
        df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)
        
        # Topic keyword filter (if enabled)
        original_size = len(df)
        if topic and must_contain_topic:
            mask = df["combined"].str.contains(topic, case=False, na=False)
            df = df[mask]
            filtered_size = len(df)
            if df.empty:
                status.update(label="No exact matches found, showing semantic matches instead...")
                df = load_dataframe(uploaded_file)
                df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)
            elif filtered_size < original_size:
                status.update(label=f"Found {filtered_size} pages containing '{topic}'. Analyzing...")
        
        # Embedding stage
        status.update(label="Creating content embeddings (this may take a moment)...")
        cache_key = get_cache_key(uploaded_file.name, include_queries, len(df))
        if st.session_state.get("embed_cache_key") != cache_key:
            client = AsyncOpenAI(api_key=api_key)
            vectors = asyncio.run(embed_texts(client, df["combined"].tolist(), concurrency))
            st.session_state["page_embeddings"] = np.array(vectors, dtype=np.float32)
            st.session_state["embed_cache_key"] = cache_key
        
        embeddings: np.ndarray = st.session_state["page_embeddings"]
        
        # Safety check for length mismatch
        if len(embeddings) != len(df):
            client = AsyncOpenAI(api_key=api_key)
            embeddings = np.array(asyncio.run(embed_texts(client, df["combined"].tolist(), concurrency)), dtype=np.float32)
            st.session_state["page_embeddings"] = embeddings
        
        # Embed query and find matching content
        status.update(label="Finding the best content matches for your topic...")
        client = AsyncOpenAI(api_key=api_key)
        q_vec = np.array(asyncio.run(embed_texts(client, [query_text], parallel=1))[0], dtype=np.float32)
        
        df["similarity"] = cosine_similarity_matrix(embeddings, q_vec)
        
        # Post‑filters
        df = df[df["Top-Level Subfolder"].isin(chosen_folders) & (df["similarity"] >= min_sim)]
        
        results = df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
        status.update(label="✅ Recommendations ready!", state="complete")
    
    # Display results
    if len(results) == 0:
        st.warning("No matching content found. Try broadening your topic or lowering the minimum relevance threshold.")
    else:
        st.success(f"Found {len(results)} recommendations related to your topic!")
        
        # Format for display
        display_results = results.copy()
        display_results["Score"] = (display_results["similarity"] * 100).round(1).astype(str) + '%'
        display_results["Relevance"] = display_results["similarity"].apply(get_relevance_indicator)
        
        # Results summary table
        st.subheader("Summary")
        col_order = ["Title", "Top-Level Subfolder", "Score", "Relevance"]
        st.dataframe(display_results[col_order], use_container_width=True, hide_index=True)
        
        # Detailed recommendations
        st.subheader("Detailed Recommendations")
        
        # Determine if we should group by category
        if display_results["Top-Level Subfolder"].nunique() > 1:
            # Show tabs for each category
            tabs = st.tabs([f"{cat} ({len(display_results[display_results['Top-Level Subfolder']==cat])})" 
                           for cat in sorted(display_results["Top-Level Subfolder"].unique())])
            
            for i, category in enumerate(sorted(display_results["Top-Level Subfolder"].unique())):
                with tabs[i]:
                    category_results = display_results[display_results["Top-Level Subfolder"] == category]
                    
                    for _, row in category_results.iterrows():
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.subheader(row["Title"])
                            with col2:
                                st.markdown(f"<div style='text-align: right'>{row['Relevance']} ({row['Score']})</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"**Description:** {row['Meta Description']}")
                            st.markdown(f"**Link:** [{row['URL']}]({row['URL']})")
                            
                            # Show search queries if available
                            queries = []
                            for q_num in range(1, 4):
                                q = row.get(f"Query #{q_num}", "")
                                clicks = row.get(f"Query #{q_num} Clicks", 0)
                                if q and not pd.isna(q) and clicks > 0:
                                    queries.append(f"- {q} ({clicks} clicks)")
                            
                            if queries:
                                with st.expander("Popular searches"):
                                    st.markdown("\n".join(queries))
                            
                            st.divider()
        else:
            # Single category - show as expandable items
            for i, row in display_results.iterrows():
                with st.expander(f"{row['Title']} - {row['Relevance']} ({row['Score']})", expanded=i==0):
                    st.markdown(f"**Description:** {row['Meta Description']}")
                    st.markdown(f"**Category:** {row['Top-Level Subfolder']}")
                    st.markdown(f"**Link:** [{row['URL']}]({row['URL']})")
                    
                    # Show search queries if available
                    queries = []
                    for q_num in range(1, 4):
                        q = row.get(f"Query #{q_num}", "")
                        clicks = row.get(f"Query #{q_num} Clicks", 0)
                        if q and not pd.isna(q) and clicks > 0:
                            queries.append(f"- {q} ({clicks} clicks)")
                    
                    if queries:
                        st.markdown("**Popular searches:**")
                        st.markdown("\n".join(queries))
        
        # Download option
        csv_bytes = results.to_csv(index=False).encode()
        st.download_button("💾 Download Recommendations", csv_bytes, "content_recommendations.csv", "text/csv")
