# Medical FAQ RAG System - Streamlit User Interface
# Features: Chat interface, retrieval method selection, evidence display, chat history

import streamlit as st
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline


# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical FAQ Assistant",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Color Variables ─────────────────────────────────────────── */
    :root {
        --purple-50: #faf5ff;
        --purple-100: #f3e8ff;
        --purple-200: #e9d5ff;
        --purple-300: #d8b4fe;
        --purple-400: #c084fc;
        --purple-500: #a855f7;
        --purple-600: #9333ea;
        --purple-700: #7e22ce;
        --purple-800: #6b21a8;
        --purple-900: #581c87;
        --text-primary: #1e1b2e;
        --text-secondary: #4a4458;
        --text-muted: #6b6380;
        --bg-surface: #ffffff;
        --bg-surface-alt: #f8f5ff;
        --border-color: #e9e0f7;
        --shadow-sm: 0 1px 3px rgba(88, 28, 135, 0.08);
        --shadow-md: 0 4px 12px rgba(88, 28, 135, 0.10);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f0e6ff;
            --text-secondary: #d4c4ef;
            --text-muted: #a897c4;
            --bg-surface: #1e1529;
            --bg-surface-alt: #261d33;
            --border-color: #3d2e54;
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.35);
        }
    }

    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #7e22ce 0%, #a855f7 50%, #c084fc 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 6px 20px rgba(126, 34, 206, 0.30);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.92;
        font-size: 1rem;
        color: #f3e8ff;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Evidence card styling */
    .evidence-card {
        background: var(--bg-surface-alt);
        border-left: 4px solid var(--purple-600);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .evidence-card .source-tag {
        display: inline-block;
        background: var(--purple-600);
        color: #ffffff;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .evidence-card .score-tag {
        display: inline-block;
        background: var(--purple-100);
        color: var(--purple-800);
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .evidence-card .score-tag {
            background: var(--purple-900);
            color: var(--purple-200);
        }
    }
    .evidence-card p {
        color: var(--text-secondary) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-surface-alt) 100%);
        border-right: 1px solid var(--border-color);
    }

    /* Status indicators */
    .status-ready {
        color: #16a34a;
        font-weight: 600;
    }
    .status-loading {
        color: #d97706;
        font-weight: 600;
    }
    .status-error {
        color: #dc2626;
        font-weight: 600;
    }

    /* Metrics cards */
    .metric-card {
        background: var(--bg-surface);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
        text-align: center;
        color: var(--text-primary);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: var(--text-muted);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }

    /* Streamlit overrides for dark/light compat */
    .stMarkdown, .stText {
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ────────────────────────────────────────────────
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


# ── Helper Functions ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and initialize the RAG pipeline (cached)."""
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


def format_evidence(chunks: list) -> str:
    """Format retrieved evidence chunks as HTML cards."""
    html_parts = []
    for chunk in chunks:
        source = chunk.get('source', 'Unknown')
        score = chunk.get('score', 0)
        text = chunk.get('text', '')
        rank = chunk.get('rank', '-')
        method = chunk.get('method', 'Unknown')
        
        # Truncate text for display
        display_text = text[:300] + '...' if len(text) > 300 else text
        
        html_parts.append(f"""
        <div class="evidence-card">
            <span class="source-tag">{source}</span>
            <span class="score-tag">Score: {score:.4f}</span>
            <span class="score-tag">Rank #{rank}</span>
            <span class="score-tag">{method}</span>
            <p style="margin-top: 0.5rem;">{display_text}</p>
        </div>
        """)
    
    return ''.join(html_parts)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    
    # Retrieval method selection
    retrieval_method = st.selectbox(
        "Retrieval Method",
        options=['hybrid', 'dense', 'tfidf'],
        format_func=lambda x: {
            'hybrid': 'Hybrid (TF-IDF + Dense)',
            'dense': 'Dense Embeddings',
            'tfidf': 'TF-IDF'
        }[x],
        index=0,
        help="Choose the retrieval strategy for finding relevant context."
    )
    
    # Top-K slider
    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of relevant chunks to retrieve for each query."
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("## System Status")
    if st.session_state.initialized:
        st.markdown('<p class="status-ready">System Ready</p>', unsafe_allow_html=True)
        if st.session_state.pipeline and st.session_state.pipeline.chunks:
            st.metric("Documents Indexed", len(set(c['source'] for c in st.session_state.pipeline.chunks)))
            st.metric("Total Chunks", len(st.session_state.pipeline.chunks))
    else:
        st.markdown('<p class="status-loading">Initializing...</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat history info
    st.markdown("## Chat History")
    st.metric("Messages", len(st.session_state.messages))
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # About section
    st.markdown("## About")
    st.markdown("""
    **Medical FAQ RAG System**
    
    A Retrieval-Augmented Generation system for answering medical questions using:
    - 10 medical FAQ documents
    - TF-IDF and Dense retrieval
    - HuggingFace LLM generation
    - Chat history memory
    
    *By Muqaddas Khan and Ruhaan Ahmad*
    """)


# ── Main Content ────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>Medical FAQ Assistant</h1>
    <p>Ask any health-related question and get answers powered by RAG technology</p>
</div>
""", unsafe_allow_html=True)

# Initialize pipeline
if not st.session_state.initialized:
    with st.spinner("Initializing RAG pipeline... (first load takes a moment to download models)"):
        try:
            st.session_state.pipeline = load_pipeline()
            st.session_state.initialized = True
            st.rerun()
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {str(e)}")
            st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show evidence for assistant messages
        if message["role"] == "assistant" and "evidence" in message:
            with st.expander("View Retrieved Evidence", expanded=False):
                st.markdown(format_evidence(message["evidence"]), unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a medical question... (e.g., 'What are the symptoms of diabetes?')"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                # Build chat history for context (excluding current message)
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]  # Exclude current message
                ]
                
                # Query the RAG pipeline
                result = st.session_state.pipeline.query(
                    question=prompt,
                    method=retrieval_method,
                    top_k=top_k,
                    chat_history=chat_history if chat_history else None
                )
                
                answer = result['answer']
                evidence = result['retrieved_chunks']
                
                # Display answer
                st.markdown(answer)
                
                # Display evidence
                with st.expander("View Retrieved Evidence", expanded=False):
                    st.markdown(format_evidence(evidence), unsafe_allow_html=True)
                
                # Store in chat history with evidence
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence
                })
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Method: {result['retrieval_method']}")
                with col2:
                    st.caption(f"Model: {result['model']}")
                with col3:
                    st.caption(f"Chunks: {len(evidence)}")
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("""
<div class="footer">
    <p>Medical FAQ RAG System | Built with Streamlit, Sentence Transformers and HuggingFace | 
    Muqaddas Khan (22F-3214) and Ruhaan Ahmad (22F-4949)</p>
    <p>This system is for educational purposes only. Always consult a healthcare professional for medical advice.</p>
</div>
""", unsafe_allow_html=True)
