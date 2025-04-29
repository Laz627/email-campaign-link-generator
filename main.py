import asyncio
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "text-embedding-3-small"
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
        str(row["Title"]),
        str(row["Meta Description"]),
        str(row["URL"]),
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
    try:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        
        # Add debug info
        st.session_state['debug_info'] = f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns"
        
        # More flexible column checking
        missing = []
        for c in BASE_COLS:
            found = False
            for col in df.columns:
                if c.lower() in col.lower():
                    # Rename to standard name if found with different case/format
                    if col != c:
                        df[c] = df[col]
                    found = True
                    break
            if not found:
                missing.append(c)
                
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.info("Your spreadsheet must contain: URL, Title, Meta Description, and Top-Level Subfolder columns")
            st.stop()
            
        for c in QUERY_COLS:
            if c not in df.columns:
                df[c] = 0 if "Clicks" in c else ""
                
        # Ensure data types
        for col in ["Title", "Meta Description", "URL", "Top-Level Subfolder"]:
            df[col] = df[col].astype(str)
            
        # Remove NaN or empty values in critical columns
        df = df.dropna(subset=BASE_COLS).reset_index(drop=True)
        
        # Add debug info
        st.session_state['debug_info'] += f"\nAfter cleanup: {len(df)} rows remain"
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


def cosine_similarity_matrix(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    # Safe version with error handling
    try:
        # Check if inputs are valid
        if len(mat) == 0 or len(vec) == 0:
            st.error("Empty embeddings detected")
            return np.zeros(len(mat))
            
        # Normalize to prevent division by zero
        norm_mat = np.linalg.norm(mat, axis=1)
        norm_vec = np.linalg.norm(vec)
        
        # Handle zero norms
        if norm_vec == 0:
            return np.zeros(len(mat))
            
        zero_norms = norm_mat == 0
        if np.any(zero_norms):
            norm_mat[zero_norms] = 1.0  # Prevent division by zero
            
        return (mat @ vec) / (norm_mat * norm_vec)
        
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return np.zeros(len(mat))


async def embed_texts(client: AsyncOpenAI, texts: List[str], parallel: int) -> List[List[float]]:
    sem = asyncio.Semaphore(parallel)
    results = []

    async def _embed(txt: str, idx: int):
        async with sem:
            for attempt in range(5):
                try:
                    # Add a debug message
                    if idx % 10 == 0:
                        st.session_state['embed_status'] = f"Processing item {idx+1}/{len(texts)}"
                        
                    # Ensure text isn't empty
                    if not txt.strip():
                        txt = "Empty content"
                        
                    r = await client.embeddings.create(model=MODEL_NAME, input=[txt])
                    return r.data[0].embedding
                except OpenAIError as e:
                    if attempt == 4:
                        st.error(f"Embedding error after 5 attempts: {str(e)}")
                        # Return a zero embedding as fallback
                        return [0.0] * 1536  # Standard embedding size
                    await asyncio.sleep(2**attempt)

    # Use enumerate to track progress
    tasks = [_embed(t, i) for i, t in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    return results


def get_cache_key(filename: str, enrich: bool, df_len: int) -> Tuple[str, bool, int]:
    return (filename, enrich, df_len)


def boost_exact_match_scores(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """Boost similarity scores for content that exactly matches the topic"""
    if not topic:
        return df
        
    # Check if topic appears in title or description
    topic_lower = topic.lower()
    
    for idx, row in df.iterrows():
        title_match = topic_lower in row['Title'].lower()
        desc_match = topic_lower in row['Meta Description'].lower()
        
        # Apply boost based on where the match is found
        if title_match:
            df.at[idx, 'similarity'] = min(1.0, df.at[idx, 'similarity'] * 1.2)  # 20% boost for title matches
        elif desc_match:
            df.at[idx, 'similarity'] = min(1.0, df.at[idx, 'similarity'] * 1.1)  # 10% boost for description matches
            
    return df


def get_relevance_indicator(score: float) -> str:
    """Return a relevance indicator based on similarity score"""
    # Recalibrated thresholds based on real-world performance
    if score >= 0.65:
        return "🟢 Excellent match"
    elif score >= 0.55:
        return "🟡 Strong match"
    else:
        return "🔵 Relevant"


def format_score_for_display(score: float) -> str:
    """Format the similarity score for display, with color coding"""
    score_pct = int(score * 100)
    
    if score >= 0.65:
        return f"<span style='color:green;font-weight:bold'>{score_pct}%</span>"
    elif score >= 0.55:
        return f"<span style='color:#D4AC0D;font-weight:bold'>{score_pct}%</span>"
    else:
        return f"<span style='color:blue'>{score_pct}%</span>"


def display_similarity_gauge(score: float) -> None:
    """Display a visual gauge representing similarity strength"""
    # Calculate value for the gauge (0-100)
    gauge_value = min(100, max(0, int(score * 100)))
    
    # Create a visual representation with custom colors
    colors = {
        "poor": "#CDCDCD",
        "low": "#ADD8E6",
        "medium": "#D4AC0D",
        "high": "#5CB85C"
    }
    
    # Determine color based on score
    if score >= 0.65:
        color = colors["high"]
    elif score >= 0.55:
        color = colors["medium"]
    else:
        color = colors["low"]
    
    st.progress(gauge_value/100, text=f"Relevance: {gauge_value}%")
    
    # Add explanation text
    if score >= 0.65:
        st.caption("This content is highly relevant to your topic")
    elif score >= 0.55:
        st.caption("This content is strongly related to your topic")
    else:
        st.caption("This content has relevant information for your topic")

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Content Recommendation Engine", layout="wide")

# Initialize session state variables if they don't exist
if 'debug_info' not in st.session_state:
    st.session_state['debug_info'] = ""
if 'embed_status' not in st.session_state:
    st.session_state['embed_status'] = ""

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
    # Lowered default threshold to ensure matches
    min_sim = st.slider("Minimum relevance score", 0.0, 1.0, 0.45, 0.01, 
                      help="Higher = more relevant but potentially fewer results")
    
    # Explanation of scoring
    st.info("""
    **About relevance scores:**
    - 65%+ = Excellent match 🟢
    - 55%+ = Strong match 🟡
    - 45%+ = Relevant 🔵
    
    Scores reflect semantic similarity - even "Relevant" content can be quite useful!
    """)
    
    include_queries = st.checkbox("Include search behavior data", value=True, 
                               help="Considers popular search queries when matching content")
    boost_exact_matches = st.checkbox("Boost exact keyword matches", value=True,
                                  help="Increase scores for content containing your exact keywords")
    
    with st.expander("Advanced Settings"):
        concurrency = st.slider("Parallel API requests", 1, 50, 10, 
                              help="Higher values may process faster but use more API credits")
        top_n = st.slider("Maximum results to show", 5, 50, 15)
        
        # Debug mode toggle
        debug_mode = st.checkbox("Debug mode", value=False, 
                              help="Show detailed processing information")

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
                            placeholder="e.g., Pella Impervia, sustainable gardening")
        
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

# Show debug info if enabled
if 'debug_mode' in locals() and debug_mode and st.session_state['debug_info']:
    st.sidebar.subheader("Debug Information")
    st.sidebar.code(st.session_state['debug_info'])
    
    if st.session_state['embed_status']:
        st.sidebar.text(st.session_state['embed_status'])

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file and 'find_btn' in locals() and find_btn:
    # Input validation
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    # Reset debug info
    st.session_state['debug_info'] = ""
    st.session_state['embed_status'] = ""
    
    # Display the recommendations section
    st.header(f"📊 Recommendations for: {topic}")
    if details:
        st.caption(f"Additional context: {details}")
    
    # Read and prep DataFrame
    df = load_dataframe(uploaded_file)
    
    # Safety check for empty dataframe
    if len(df) == 0:
        st.error("No valid content found in the uploaded file.")
        st.stop()
    
    # Sub‑folder filter
    folders = df["Top-Level Subfolder"].unique().tolist()
    chosen_folders = st.multiselect("Filter by category (optional)", folders, default=folders)
    
    # Make sure at least one folder is selected
    if not chosen_folders:
        st.warning("Please select at least one category to search in.")
        st.stop()
    
    with st.status("Finding relevant content...", expanded=True) as status:
        # Compose query
        query_text = f"{topic} {details}".strip()
        
        status.update(label="Processing content data...")
        # Add debug info
        st.session_state['debug_info'] += f"\nProcessing query: '{query_text}'"
        
        # Create combined field for matching
        df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)
        
        # Apply folder filter
        df = df[df["Top-Level Subfolder"].isin(chosen_folders)]
        original_size = len(df)
        st.session_state['debug_info'] += f"\nFiltered by folders: {len(df)} rows after folder filtering"
        
        if df.empty:
            status.update(label="No content found in selected categories.", state="error")
            st.error("No content matched your category filters. Please select different categories.")
            st.stop()
        
        # Embedding stage
        status.update(label="Creating content embeddings (this may take a moment)...")
        client = AsyncOpenAI(api_key=api_key)
        
        # Always regenerate embeddings to ensure they're fresh
        try:
            vectors = asyncio.run(embed_texts(client, df["combined"].tolist(), concurrency))
            embeddings = np.array(vectors, dtype=np.float32)
            st.session_state['debug_info'] += f"\nGenerated {len(embeddings)} embeddings"
        except Exception as e:
            status.update(label=f"Error generating embeddings: {str(e)}", state="error")
            st.error(f"Failed to create embeddings: {str(e)}")
            st.stop()
        
        # Embed query and find matching content
        status.update(label="Finding the best content matches for your topic...")
        try:
            q_vec = np.array(asyncio.run(embed_texts(client, [query_text], parallel=1))[0], dtype=np.float32)
            st.session_state['debug_info'] += f"\nGenerated query embedding successfully"
            
            # Calculate similarity
            df["similarity"] = cosine_similarity_matrix(embeddings, q_vec)
            
            # Boost exact match scores if enabled
            if boost_exact_matches:
                df = boost_exact_match_scores(df, topic)
                st.session_state['debug_info'] += f"\nBoost applied for exact keyword matches"
            
            st.session_state['debug_info'] += f"\nCalculated similarity scores. Range: {df['similarity'].min():.4f} to {df['similarity'].max():.4f}"
            
            # Post‑filters
            filtered_df = df[df["similarity"] >= min_sim]
            st.session_state['debug_info'] += f"\nAfter similarity threshold ({min_sim}): {len(filtered_df)}/{len(df)} rows remain"
            
            if len(filtered_df) == 0:
                # If no results meet threshold, just take top N
                status.update(label="No results met similarity threshold, showing best matches instead", state="warning")
                results = df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
                st.session_state['debug_info'] += f"\nTaking best {len(results)} results regardless of threshold"
            else:
                # Normal case - use filtered results
                results = filtered_df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
                st.session_state['debug_info'] += f"\nFinal results: {len(results)} items"
            
            status.update(label="✅ Recommendations ready!", state="complete")
            
        except Exception as e:
            status.update(label=f"Error processing recommendations: {str(e)}", state="error")
            st.error(f"Failed to process recommendations: {str(e)}")
            st.stop()
    
    # Display results
    if len(results) == 0:
        st.warning("No matching content found. Try broadening your topic or lowering the minimum relevance threshold.")
    else:
        st.success(f"Found {len(results)} recommendations related to your topic!")
        
        # Format for display
        display_results = results.copy()
        display_results["Score"] = display_results["similarity"].apply(lambda s: format_score_for_display(s))
        display_results["Relevance"] = display_results["similarity"].apply(get_relevance_indicator)
        
        # Results summary table
        st.subheader("Summary")
        # Use HTML for better formatting
        st.markdown("""
        <style>
        .score-column { text-align: center; }
        </style>
        """, unsafe_allow_html=True)
        
        # Display summary dataframe with HTML-formatted scores
        col_order = ["Title", "Top-Level Subfolder", "Score", "Relevance"]
        st.write(
            display_results[col_order].to_html(
                escape=False, 
                index=False,
                classes=['dataframe']
            ), 
            unsafe_allow_html=True
        )
        
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
                                st.markdown(f"<div style='text-align: right'>{row['Relevance']}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align: right'>{row['Score']}</div>", unsafe_allow_html=True)
                            
                            # Display similarity gauge
                            display_similarity_gauge(row["similarity"])
                            
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
                    # Display similarity gauge
                    display_similarity_gauge(row["similarity"])
                    
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
