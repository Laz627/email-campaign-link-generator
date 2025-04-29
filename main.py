import asyncio
import re
from typing import List, Tuple, Set, Optional
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"  # Using large model for better accuracy
EXPLANATION_MODEL = "gpt-4o-mini"  # For generating explanations
BASE_COLS = ["URL", "Title", "Meta Description", "Top-Level Subfolder"]
QUERY_COLS = [
    "Query #1",
    "Query #1 Clicks",
    "Query #2",
    "Query #2 Clicks",
    "Query #3",
    "Query #3 Clicks",
]
# Common English stop words to filter out when extracting keywords
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "when",
    "where", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "to", "from", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "why",
    "who", "whom", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "for", "of", "at", "by"
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_keywords(text: str) -> Set[str]:
    """Extract important keywords from a text by removing stop words and keeping substantive terms"""
    if not text:
        return set()
    
    # Convert to lowercase and split on non-alphanumeric characters
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    # Filter out stop words and short words (length < 3)
    keywords = {word for word in words if word not in STOP_WORDS and len(word) >= 3}
    
    return keywords


def find_traffic_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most appropriate traffic column in the dataframe"""
    # Try common traffic column names
    candidates = [
        "Monthly Clicks", 
        "Monthly Traffic", 
        "Traffic", 
        "Pageviews", 
        "Monthly Pageviews",
        "Visits",
        "Monthly Visits",
        "Total Clicks"
    ]
    
    for col in candidates:
        if col in df.columns:
            return col
            
    # Look for columns with traffic-related terms
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ["traffic", "click", "visit", "view", "impression"]):
            # Verify it contains numeric data
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].astype(str).str.isnumeric().any():
                return col
    
    return None


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


async def generate_explanations(client: AsyncOpenAI, results_df: pd.DataFrame, topic: str, details: str) -> List[str]:
    """Generate short explanations for why each URL was selected"""
    full_topic = f"{topic} {details}".strip()
    explanations = []
    
    async def _get_explanation(row_idx: int):
        title = results_df.iloc[row_idx]["Title"]
        description = results_df.iloc[row_idx]["Meta Description"]
        score = results_df.iloc[row_idx]["similarity"]
        
        prompt = f"""Please explain in 1-2 short sentences why this content is relevant to the topic "{full_topic}".
        
Content Title: {title}
Content Description: {description}
Relevance Score: {score:.2f}

Your explanation should be brief, specific, and highlight the connection between the content and the topic.
"""
        
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=EXPLANATION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    return f"This content appears relevant to your search for {topic}."
                await asyncio.sleep(1)
    
    # Process explanations in batches to avoid rate limits
    batch_size = 5
    for i in range(0, len(results_df), batch_size):
        batch_indices = range(i, min(i + batch_size, len(results_df)))
        batch_tasks = [_get_explanation(idx) for idx in batch_indices]
        batch_results = await asyncio.gather(*batch_tasks)
        explanations.extend(batch_results)
        
    return explanations


def boost_by_traffic(df: pd.DataFrame, max_boost: float = 1.25) -> pd.DataFrame:
    """
    Boost scores based on page traffic data.
    
    Args:
        df: DataFrame with content items
        max_boost: Maximum boost factor to apply (1.0 = no boost)
        
    Returns:
        DataFrame with adjusted similarity scores
    """
    # Find appropriate traffic column
    traffic_col = find_traffic_column(df)
    
    # If no suitable column found, return unchanged
    if not traffic_col:
        if 'debug_info' in st.session_state:
            st.session_state['debug_info'] += "\nNo traffic column found for boosting"
        return df
    
    # Convert to numeric and handle NaN values
    df[traffic_col] = pd.to_numeric(df[traffic_col], errors='coerce').fillna(0)
    
    # Get non-zero traffic values
    traffic_values = df[traffic_col][df[traffic_col] > 0]
    
    # If no valid traffic data, return unchanged
    if len(traffic_values) == 0:
        if 'debug_info' in st.session_state:
            st.session_state['debug_info'] += f"\nNo valid traffic data in column '{traffic_col}'"
        return df
    
    # Calculate percentiles for normalization (using 95th to handle outliers)
    p95 = traffic_values.quantile(0.95)
    p50 = traffic_values.quantile(0.50)
    
    # Add debug info
    if 'debug_info' in st.session_state:
        st.session_state['debug_info'] += f"\nTraffic stats from '{traffic_col}': median={p50:.1f}, 95th percentile={p95:.1f}"
    
    # Apply boost based on traffic
    boosts_applied = 0
    
    for idx, row in df.iterrows():
        traffic = row.get(traffic_col, 0)
        if traffic and not pd.isna(traffic) and traffic > 0:
            # Normalize traffic (0 to 1 scale)
            norm_traffic = min(traffic, p95) / p95
            
            # Calculate boost factor (between 1.0 and max_boost)
            # Higher boosts for traffic above median
            if traffic > p50:
                boost = 1.0 + (max_boost - 1.0) * norm_traffic
            else:
                # Smaller boost for below-median traffic
                boost = 1.0 + (max_boost - 1.0) * norm_traffic * 0.5
            
            # Apply boost
            df.at[idx, 'similarity'] = min(1.0, df.at[idx, 'similarity'] * boost)
            boosts_applied += 1
            
            if 'debug_info' in st.session_state and norm_traffic > 0.5:  # Only log significant boosts
                st.session_state['debug_info'] += f"\nBoosted item {idx} by {boost:.2f}x based on {traffic} {traffic_col}"
    
    if 'debug_info' in st.session_state:
        st.session_state['debug_info'] += f"\nTraffic boosts applied to {boosts_applied} items"
    
    return df


def adjust_scores_by_keywords(df: pd.DataFrame, topic: str, demotion_factor: float = 0.8) -> pd.DataFrame:
    """
    Apply a penalty to pages that don't contain important keywords from the topic.
    
    Args:
        df: DataFrame with content items
        topic: The user's query topic
        demotion_factor: How much to reduce scores for pages without keywords (0-1)
        
    Returns:
        DataFrame with adjusted similarity scores
    """
    if not topic:
        return df
    
    # Extract keywords from the topic
    topic_keywords = extract_keywords(topic)
    
    if not topic_keywords:
        return df  # No valid keywords to check
    
    # Count keyword occurrences in each item
    for idx, row in df.iterrows():
        content_text = f"{row['Title']} {row['Meta Description']}"
        content_keywords = extract_keywords(content_text)
        
        # Check keyword overlap
        matching_keywords = topic_keywords.intersection(content_keywords)
        
        # Apply demotion if no keyword matches found
        if not matching_keywords:
            df.at[idx, 'similarity'] = df.at[idx, 'similarity'] * demotion_factor
            if 'debug_info' in st.session_state:
                st.session_state['debug_info'] += f"\nDemoted item {idx} (no keyword matches)"
    
    return df


def boost_by_query_clicks(df: pd.DataFrame, topic: str, max_boost: float = 1.3) -> pd.DataFrame:
    """
    Boost scores based on search query clicks that are related to the topic.
    
    Args:
        df: DataFrame with content items
        topic: The user's query topic
        max_boost: Maximum boost factor to apply (1.0 = no boost)
        
    Returns:
        DataFrame with adjusted similarity scores
    """
    if not topic:
        return df
    
    # Extract keywords from the topic
    topic_keywords = extract_keywords(topic)
    
    if not topic_keywords:
        return df  # No valid keywords to check
    
    # Process each content item
    for idx, row in df.iterrows():
        boost_applied = False
        
        for q_num in range(1, 4):
            query = row.get(f"Query #{q_num}", "")
            clicks = row.get(f"Query #{q_num} Clicks", 0)
            
            if not isinstance(query, str) or pd.isna(query) or not clicks:
                continue
                
            # Extract keywords from the query
            query_keywords = extract_keywords(query)
            
            # Calculate keyword overlap ratio with topic
            if query_keywords and topic_keywords:
                overlap = len(topic_keywords.intersection(query_keywords)) / len(topic_keywords)
                
                # Apply boost based on overlap and click volume
                if overlap > 0:
                    # Normalize clicks (assuming most clicks are < 1000)
                    norm_clicks = min(clicks, 1000) / 1000
                    
                    # Calculate boost factor (between 1.0 and max_boost)
                    boost = 1.0 + (max_boost - 1.0) * overlap * norm_clicks
                    
                    # Apply boost
                    df.at[idx, 'similarity'] = min(1.0, df.at[idx, 'similarity'] * boost)
                    boost_applied = True
        
        if boost_applied and 'debug_info' in st.session_state:
            st.session_state['debug_info'] += f"\nBoosted item {idx} based on related query clicks"
    
    return df


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
    
    st.subheader("Advanced Scoring")
    boost_exact_matches = st.checkbox("Boost exact keyword matches", value=True,
                                  help="Increase scores for content containing your exact keywords")
    
    demote_non_keyword_matches = st.checkbox("Demote pages without keywords", value=True,
                                         help="Reduce scores for pages that don't contain important keywords")
    
    boost_by_clicks = st.checkbox("Boost by query relevance & clicks", value=True,
                               help="Increase scores for pages with high-traffic queries related to your topic")
                               
    boost_traffic = st.checkbox("Boost high-traffic pages", value=True,
                             help="Increase scores for pages with high overall traffic")
    
    generate_explanations_toggle = st.checkbox("Generate AI explanations", value=True,
                                          help="Add brief explanations of why content was recommended")
    
    with st.expander("Advanced Settings"):
        concurrency = st.slider("Parallel API requests", 1, 50, 10, 
                              help="Higher values may process faster but use more API credits")
        top_n = st.slider("Maximum results to show", 5, 50, 15)
        
        demotion_factor = st.slider("Keyword demotion factor", 0.5, 1.0, 0.8, 0.05,
                                  help="How much to reduce scores for pages without keywords (lower = stronger penalty)")
        
        max_click_boost = st.slider("Maximum click boost", 1.0, 2.0, 1.3, 0.05,
                                  help="Maximum boost factor for high-traffic relevant queries")
                                  
        max_traffic_boost = st.slider("Maximum traffic boost", 1.0, 2.0, 1.25, 0.05,
                                   help="Maximum boost factor for high-traffic pages")
        
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
    
    Optional columns that improve recommendations:
    - Search queries and their clicks
    - Monthly traffic/pageviews data
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
        specified topic, ranked by semantic similarity and enhanced by:
        
        - ✓ Semantic relevance to your topic
        - ✓ Presence of important keywords
        - ✓ Traffic and engagement data
        - ✓ Search query relevance
        """)
        
        # Example content format
        with st.expander("Sample Spreadsheet Format", expanded=True):
            example_data = {
                "URL": ["https://example.com/page1", "https://example.com/page2"],
                "Title": ["Beginner's Guide to Gardening", "Energy-Efficient Home Improvements"],
                "Meta Description": ["Learn the basics of sustainable gardening", "Cost-effective ways to improve home energy efficiency"],
                "Top-Level Subfolder": ["Gardening", "Home Improvement"],
                "Query #1": ["gardening tips", "energy saving"],
                "Query #1 Clicks": [120, 89],
                "Monthly Traffic": [1450, 980]
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
    
    with st.status("Finding relevant content...", expanded=True) as status:
        # Compose query
        query_text = f"{topic} {details}".strip()
        
        status.update(label="Processing content data...")
        # Add debug info
        st.session_state['debug_info'] += f"\nProcessing query: '{query_text}'"
        
        # Create combined field for matching
        df["combined"] = df.apply(lambda r: combine_fields(r, include_queries), axis=1)
        
        # Check for traffic column
        traffic_col = find_traffic_column(df)
        if traffic_col:
            st.session_state['debug_info'] += f"\nFound traffic data in column: '{traffic_col}'"
        
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
            st.session_state['debug_info'] += f"\nInitial similarity scores: range {df['similarity'].min():.4f} to {df['similarity'].max():.4f}"
            
            # Apply scoring adjustments
            if boost_exact_matches:
                df = boost_exact_match_scores(df, topic)
                st.session_state['debug_info'] += f"\nBoost applied for exact keyword matches"
            
            if demote_non_keyword_matches:
                df = adjust_scores_by_keywords(df, topic, demotion_factor)
                st.session_state['debug_info'] += f"\nDemotion applied for pages without important keywords"
            
            if boost_by_clicks:
                df = boost_by_query_clicks(df, topic, max_click_boost)
                st.session_state['debug_info'] += f"\nBoost applied based on query relevance and clicks"
                
            if boost_traffic:
                df = boost_by_traffic(df, max_traffic_boost)
                st.session_state['debug_info'] += f"\nBoost applied based on page traffic"
            
            st.session_state['debug_info'] += f"\nFinal similarity scores: range {df['similarity'].min():.4f} to {df['similarity'].max():.4f}"
            
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
            
            # Generate explanations if enabled
            if generate_explanations_toggle and len(results) > 0:
                status.update(label="Generating explanations for recommendations...")
                explanations = asyncio.run(generate_explanations(client, results, topic, details))
                results["explanation"] = explanations
                st.session_state['debug_info'] += f"\nGenerated {len(explanations)} recommendation explanations"
            
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
                            
                            # Show explanation if available
                            if "explanation" in row and row["explanation"]:
                                st.markdown(f"**Why this is relevant:** {row['explanation']}")
                            
                            st.markdown(f"**Description:** {row['Meta Description']}")
                            st.markdown(f"**Link:** [{row['URL']}]({row['URL']})")
                            
                            # Show traffic if available
                            if traffic_col and traffic_col in row and row[traffic_col] > 0:
                                st.markdown(f"**{traffic_col}:** {int(row[traffic_col]):,}")
                            
                            # Show search queries if available
                            queries = []
                            for q_num in range(1, 4):
                                q = row.get(f"Query #{q_num}", "")
                                clicks = row.get(f"Query #{q_num} Clicks", 0)
                                if q and not pd.isna(q) and clicks > 0:
                                    queries.append(f"- {q} ({clicks:,} clicks)")
                            
                            if queries:
                                with st.expander("Popular searches"):
                                    st.markdown("\n".join(queries))
                            
                            st.divider()
        else:
            # Single category view
            for i, row in display_results.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.subheader(row["Title"])
                    with col2:
                        st.markdown(f"<div style='text-align: right'>{row['Relevance']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align: right'>{row['Score']}</div>", unsafe_allow_html=True)
                    
                    # Display similarity gauge
                    display_similarity_gauge(row["similarity"])
                    
                    # Show explanation if available
                    if "explanation" in row and row["explanation"]:
                        st.markdown(f"**Why this is relevant:** {row['explanation']}")
                    
                    st.markdown(f"**Description:** {row['Meta Description']}")
                    st.markdown(f"**Link:** [{row['URL']}]({row['URL']})")
                    
                    # Show traffic if available
                    if traffic_col and traffic_col in row and row[traffic_col] > 0:
                        st.markdown(f"**{traffic_col}:** {int(row[traffic_col]):,}")
                    
                    # Show search queries if available
                    queries = []
                    for q_num in range(1, 4):
                        q = row.get(f"Query #{q_num}", "")
                        clicks = row.get(f"Query #{q_num} Clicks", 0)
                        if q and not pd.isna(q) and clicks > 0:
                            queries.append(f"- {q} ({clicks:,} clicks)")
                    
                    if queries:
                        with st.expander("Popular searches"):
                            st.markdown("\n".join(queries))
                    
                    st.divider()
        
        # Download option
        csv_bytes = results.to_csv(index=False).encode()
        st.download_button("💾 Download Recommendations", csv_bytes, "content_recommendations.csv", "text/csv")
