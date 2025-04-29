import asyncio
from typing import List, Tuple
import base64
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AsyncOpenAI, OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
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
                        
                    r = await client.embeddings.create(model=EMBEDDING_MODEL, input=[txt])
                    return r.data[0].embedding
                except OpenAIError as e:
                    if attempt == 4:
                        st.error(f"Embedding error after 5 attempts: {str(e)}")
                        # Return a zero embedding as fallback
                        return [0.0] * 3072  # Dimension for large embedding model
                    await asyncio.sleep(2**attempt)

    # Use enumerate to track progress
    tasks = [_embed(t, i) for i, t in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    return results


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


def create_distribution_chart(results_df: pd.DataFrame) -> str:
    """Create a distribution chart of similarity scores"""
    plt.figure(figsize=(10, 4))
    sns.histplot(results_df['similarity'], bins=10, kde=True)
    plt.title('Distribution of Relevance Scores')
    plt.xlabel('Relevance Score')
    plt.ylabel('Count')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Save to buffer and convert to base64 for embedding
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def create_category_chart(results_df: pd.DataFrame) -> str:
    """Create a chart showing scores by category"""
    if results_df["Top-Level Subfolder"].nunique() <= 1:
        return None
        
    # Get average similarity by category
    cat_avg = results_df.groupby("Top-Level Subfolder")["similarity"].mean().reset_index()
    cat_avg = cat_avg.sort_values("similarity", ascending=False)
    
    plt.figure(figsize=(10, 4))
    bars = plt.barh(cat_avg["Top-Level Subfolder"], cat_avg["similarity"])
    
    # Color bars by score
    for i, bar in enumerate(bars):
        score = cat_avg["similarity"].iloc[i]
        if score >= 0.65:
            bar.set_color("#5CB85C")  # green
        elif score >= 0.55:
            bar.set_color("#D4AC0D")  # yellow
        else:
            bar.set_color("#ADD8E6")  # blue
            
    plt.title('Average Relevance by Category')
    plt.xlabel('Average Relevance Score')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Save to buffer and convert to base64 for embedding
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"
    

def generate_csv_template() -> BytesIO:
    """Generate a template CSV file"""
    template_data = {
        "URL": ["https://example.com/page1", "https://example.com/page2"],
        "Title": ["Example Title 1", "Example Title 2"],
        "Meta Description": ["This is a description for page 1", "This is a description for page 2"],
        "Top-Level Subfolder": ["Category A", "Category B"],
        "Query #1": ["search term 1", "search term 3"],
        "Query #1 Clicks": [100, 150],
        "Query #2": ["search term 2", "search term 4"],
        "Query #2 Clicks": [50, 75],
        "Query #3": ["", "search term 5"],
        "Query #3 Clicks": [0, 25]
    }
    
    df = pd.DataFrame(template_data)
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def highlight_keyword(text: str, keyword: str) -> str:
    """Highlight a keyword in text for display"""
    if not keyword or not text:
        return text
        
    keyword_lower = keyword.lower()
    if keyword_lower not in text.lower():
        return text
        
    # Find all occurrences (case insensitive)
    import re
    pattern = re.compile(re.escape(keyword_lower), re.IGNORECASE)
    
    # Replace with highlighted version
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Content Recommendation Engine", layout="wide")

# Initialize session state variables if they don't exist
if 'debug_info' not in st.session_state:
    st.session_state['debug_info'] = ""
if 'embed_status' not in st.session_state:
    st.session_state['embed_status'] = ""
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# Apply dark mode if enabled
if st.session_state['dark_mode']:
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p {
        color: #FFFFFF !important;
    }
    .stDataFrame {
        background-color: #1E1E1E;
    }
    </style>
    """, unsafe_allow_html=True)

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
    highlight_keywords = st.checkbox("Highlight keywords in results", value=True,
                                 help="Visually highlight your topic in the results")
    
    with st.expander("Advanced Settings"):
        concurrency = st.slider("Parallel API requests", 1, 50, 10, 
                              help="Higher values may process faster but use more API credits")
        top_n = st.slider("Maximum results to show", 5, 50, 15)
        
        show_visualizations = st.checkbox("Show data visualizations", value=True,
                                      help="Display charts and graphs of the recommendation data")
                                      
        # Theme selection
        dark_mode = st.checkbox("Dark mode", value=st.session_state['dark_mode'])
        if dark_mode != st.session_state['dark_mode']:
            st.session_state['dark_mode'] = dark_mode
            st.rerun()
        
        # Debug mode toggle
        debug_mode = st.checkbox("Debug mode", value=False, 
                              help="Show detailed processing information")
                              
    # Template download
    st.markdown("### Templates")
    template_buffer = generate_csv_template()
    st.download_button("📄 Download CSV Template", 
                    template_buffer, 
                    "content_template.csv", 
                    "text/csv",
                    help="Download a sample CSV template for your content library")

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
        
        # Multiple topics mode
        enable_batch = st.checkbox("Process multiple topics at once", value=False)
        if enable_batch:
            topic_list = st.text_area("Enter one topic per line", 
                                    height=100, 
                                    help="Each line will be processed as a separate topic")
        
        if (topic or (enable_batch and topic_list)):
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
def process_topic(topic_text, details_text, df, client, all_results=None):
    """Process a single topic and return results"""
    # Compose query
    query_text = f"{topic_text} {details_text}".strip()
    
    # Add debug info
    st.session_state['debug_info'] += f"\nProcessing query: '{query_text}'"
    
    try:
        # Create combined field for matching if not already done
        if "combined" not in df.columns:
            df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)
        
        # Get embeddings for query
        q_vec = np.array(asyncio.run(embed_texts(client, [query_text], parallel=1))[0], dtype=np.float32)
        st.session_state['debug_info'] += f"\nGenerated query embedding successfully"
        
        # Calculate similarity
        df["similarity"] = cosine_similarity_matrix(embeddings, q_vec)
        
        # Boost exact match scores if enabled
        if boost_exact_matches:
            df = boost_exact_match_scores(df, topic_text)
            st.session_state['debug_info'] += f"\nBoost applied for exact keyword matches"
        
        st.session_state['debug_info'] += f"\nCalculated similarity scores. Range: {df['similarity'].min():.4f} to {df['similarity'].max():.4f}"
        
        # Apply threshold filter
        filtered_df = df[df["similarity"] >= min_sim]
        st.session_state['debug_info'] += f"\nAfter similarity threshold ({min_sim}): {len(filtered_df)}/{len(df)} rows remain"
        
        if len(filtered_df) == 0:
            # If no results meet threshold, just take top N
            results = df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
            st.session_state['debug_info'] += f"\nTaking best {len(results)} results regardless of threshold"
        else:
            # Normal case - use filtered results
            results = filtered_df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
            st.session_state['debug_info'] += f"\nFinal results: {len(results)} items"
        
        # Add topic info to results
        results["topic"] = topic_text
        
        # Add to all results if in batch mode
        if all_results is not None:
            all_results.append(results)
            
        return results
            
    except Exception as e:
        st.error(f"Error processing topic '{topic_text}': {str(e)}")
        return pd.DataFrame()

if uploaded_file and 'find_btn' in locals() and find_btn:
    # Input validation
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    # Reset debug info
    st.session_state['debug_info'] = ""
    st.session_state['embed_status'] = ""
    
    # Check if we're in batch mode
    batch_mode = enable_batch and topic_list and len(topic_list.strip()) > 0
    
    if batch_mode:
        topics = [t.strip() for t in topic_list.splitlines() if t.strip()]
        st.header(f"📊 Recommendations for {len(topics)} Topics")
    else:
        topics = [topic]
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
    
    with st.status("Finding relevant content...", expanded=True) as status:
        # Create combined field for matching
        df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)
        
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
        
        # Process each topic
        all_results = []
        
        if batch_mode:
            for i, current_topic in enumerate(topics):
                status.update(label=f"Processing topic {i+1}/{len(topics)}: {current_topic}")
                process_topic(current_topic, details, df, client, all_results)
                
            # Combine all results
            if all_results:
                results = pd.concat(all_results, ignore_index=True)
                status.update(label=f"✅ Found recommendations for {len(topics)} topics!", state="complete")
            else:
                status.update(label="No matching content found for any topics", state="error")
                st.warning("No matching content found. Try broadening your topics or lowering the minimum relevance threshold.")
                st.stop()
        else:
            # Single topic mode
            status.update(label=f"Finding the best content matches for your topic...")
            results = process_topic(topic, details, df, client)
            
            if len(results) > 0:
                status.update(label="✅ Recommendations ready!", state="complete")
            else:
                status.update(label="No matching content found", state="error")
                st.warning("No matching content found. Try broadening your topic or lowering the minimum relevance threshold.")
                st.stop()
    
    # Display results
    if len(results) == 0:
        st.warning("No matching content found. Try broadening your topic or lowering the minimum relevance threshold.")
    else:
        if batch_mode:
            st.success(f"Found recommendations for {len(topics)} topics!")
        else:
            st.success(f"Found {len(results)} recommendations related to your topic!")
        
        # Format for display
        display_results = results.copy()
        display_results["Score"] = display_results["similarity"].apply(lambda s: format_score_for_display(s))
        display_results["Relevance"] = display_results["similarity"].apply(get_relevance_indicator)
        
        # Data Visualizations
        if show_visualizations and not batch_mode:
            st.subheader("Visualization")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                dist_chart = create_distribution_chart(results)
                st.markdown(f"<img src='{dist_chart}' width='100%'>", unsafe_allow_html=True)
                
            with viz_col2:
                if results["Top-Level Subfolder"].nunique() > 1:
                    cat_chart = create_category_chart(results)
                    if cat_chart:
                        st.markdown(f"<img src='{cat_chart}' width='100%'>", unsafe_allow_html=True)
        
        # Results summary table
        st.subheader("Summary")
        # Use HTML for better formatting
        st.markdown("""
        <style>
        .score-column { text-align: center; }
        </style>
        """, unsafe_allow_html=True)
        
        # Display summary dataframe with HTML-formatted scores
        if batch_mode:
            col_order = ["topic", "Title", "Top-Level Subfolder", "Score", "Relevance"]
        else:
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
        
        # If batch mode, create tabs for each topic
        if batch_mode:
            topic_tabs = st.tabs(topics)
            
            for i, current_topic in enumerate(topics):
                with topic_tabs[i]:
                    topic_results = display_results[display_results["topic"] == current_topic]
                    
                    if len(topic_results) == 0:
                        st.info(f"No matches found for '{current_topic}'")
                        continue
                        
                    # Determine if we should group by category for this topic
                    if topic_results["Top-Level Subfolder"].nunique() > 1:
                        # Show tabs for each category
                        cats = sorted(topic_results["Top-Level Subfolder"].unique())
                        cat_tabs = st.tabs([f"{cat} ({len(topic_results[topic_results['Top-Level Subfolder']==cat])})" 
                                        for cat in cats])
                        
                        for j, category in enumerate(cats):
                            with cat_tabs[j]:
                                cat_results = topic_results[topic_results["Top-Level Subfolder"] == category]
                                
                                for _, row in cat_results.iterrows():
                                    display_result_item(row, current_topic if highlight_keywords else None)
                    else:
                        # Single category for this topic
                        for _, row in topic_results.iterrows():
                            display_result_item(row, current_topic if highlight_keywords else None)
        else:
            # Single topic mode - group by category if needed
            if display_results["Top-Level Subfolder"].nunique() > 1:
                # Show tabs for each category
                cats = sorted(display_results["Top-Level Subfolder"].unique())
                cat_tabs = st.tabs([f"{cat} ({len(display_results[display_results['Top-Level Subfolder']==cat])})" 
                                   for cat in cats])
                
                for j, category in enumerate(cats):
                    with cat_tabs[j]:
                        cat_results = display_results[display_results["Top-Level Subfolder"] == category]
                        
                        for _, row in cat_results.iterrows():
                            display_result_item(row, topic if highlight_keywords else None)
            else:
                # Single category
                for _, row in display_results.iterrows():
                    display_result_item(row, topic if highlight_keywords else None)
        
        # Download options
        st.subheader("Export Results")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            csv_bytes = results.to_csv(index=False).encode()
            st.download_button("💾 Download CSV", 
                            csv_bytes, 
                            "content_recommendations.csv", 
                            "text/csv")
                            
        with export_col2:
            excel_buffer = BytesIO()
            results.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button("📊 Download Excel", 
                            excel_buffer, 
                            "content_recommendations.xlsx", 
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            
        with export_col3:
            # Generate HTML report
            html_buffer = BytesIO()
            html_content = f"""
            <html>
            <head>
                <title>Content Recommendations for {topic}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2C3E50; }}
                    .score-high {{ color: green; font-weight: bold; }}
                    .score-medium {{ color: #D4AC0D; font-weight: bold; }}
                    .score-low {{ color: blue; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Content Recommendations</h1>
                <p>Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                <h2>Summary</h2>
                {results.to_html(index=False)}
            </body>
            </html>
            """
            html_buffer.write(html_content.encode())
            html_buffer.seek(0)
            
            st.download_button("📄 Download HTML Report", 
                            html_buffer, 
                            "content_recommendations.html", 
                            "text/html")

# Define the result item display function
def display_result_item(row, highlight_topic=None):
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            title_text = row["Title"]
            if highlight_topic and highlight_keywords:
                title_text = highlight_keyword(title_text, highlight_topic)
            st.markdown(f"### {title_text}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='text-align: right'>{row['Relevance']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: right'>{row['Score']}</div>", unsafe_allow_html=True)
        
        # Display similarity gauge
        display_similarity_gauge(row["similarity"])
        
        # Display highlighted content if enabled
        desc_text = row["Meta Description"]
        if highlight_topic and highlight_keywords:
            desc_text = highlight_keyword(desc_text, highlight_topic)
            
        st.markdown(f"**Description:** {desc_text}", unsafe_allow_html=True)
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
