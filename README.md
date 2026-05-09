# 🏥 Medical FAQ RAG System

A **Retrieval-Augmented Generation (RAG)** system for answering medical questions using an external knowledge base of medical FAQ documents.

## 👥 Authors
- **Muqaddas Khan** (22F-3214)
- **Ruhaan Ahmad** (22F-4949)

## 📋 Project Overview

This system demonstrates a complete RAG pipeline that:
1. Accepts user medical queries
2. Retrieves relevant document chunks from a curated medical FAQ knowledge base
3. Uses retrieved context to generate accurate answers via an LLM
4. Evaluates system performance across multiple medical domains

## 🏗️ System Architecture

```
User Query → Preprocessing → Retrieval (TF-IDF + Dense) → Context Selection → LLM Generation → Answer
```

### Components
- **Corpus**: 10 medical FAQ documents covering Diabetes, Hypertension, Asthma, COVID-19, Heart Disease, Mental Health, Nutrition, Allergies, Pregnancy, Vaccination, Kidney Disease, and Infectious Diseases
- **Retrieval**: Dual retrieval using TF-IDF (scikit-learn) and Dense Embeddings (Sentence Transformers + FAISS)
- **Generation**: HuggingFace Inference API (Mistral-7B-Instruct)
- **UI**: Streamlit-based chat interface with chat history memory
- **Evaluation**: Automated evaluation on 10 test queries with comparison metrics

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- HuggingFace API token (free)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Token
1. Get a free token at https://huggingface.co/settings/tokens
2. Edit the `.env` file:
```
HF_API_TOKEN=hf_your_actual_token_here
```

### Step 3: Run Preprocessing (Optional - app does this automatically)
```bash
python -m src.preprocessing
```

### Step 4: Launch the Application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
medical-rag-system/
├── data/
│   ├── raw/                        # 10 raw medical FAQ documents
│   │   ├── diabetes_faq.txt
│   │   ├── hypertension_faq.txt
│   │   ├── asthma_faq.txt
│   │   ├── covid19_faq.txt
│   │   ├── heart_disease_faq.txt
│   │   ├── mental_health_faq.txt
│   │   ├── nutrition_faq.txt
│   │   ├── allergies_faq.txt
│   │   ├── pregnancy_faq.txt
│   │   ├── vaccination_faq.txt
│   │   ├── kidney_disease_faq.txt
│   │   └── infectious_diseases_faq.txt
│   └── processed/                  # Preprocessed chunks (auto-generated)
│       ├── chunks.json
│       └── preprocessing_summary.json
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Document loading, cleaning, chunking
│   ├── retrieval.py                # TF-IDF + Dense retrieval + FAISS
│   ├── generation.py               # HuggingFace LLM generation
│   ├── rag_pipeline.py             # Full RAG pipeline orchestrator
│   └── evaluation.py               # Evaluation with 10 test queries
├── app.py                          # Streamlit UI with chat history
├── requirements.txt                # Python dependencies
├── .env                            # API token (not committed)
├── .env.example                    # API token template
├── .gitignore
├── evaluation_results/             # Auto-generated evaluation outputs
│   ├── evaluation_results.json
│   └── retrieval_comparison.png
└── README.md
```

## 🔍 Retrieval Methods

### TF-IDF Retrieval
- Uses scikit-learn's `TfidfVectorizer` with unigrams and bigrams
- Cosine similarity for ranking
- Good for keyword-matching queries

### Dense Embedding Retrieval
- Uses Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS index for fast similarity search
- Better for semantic/conceptual queries

### Hybrid Retrieval
- Combines both methods using Reciprocal Rank Fusion (RRF)
- Provides the best overall performance

## 🧪 Evaluation

Run the evaluation module to test the system on 10 predefined medical queries:
```bash
python -m src.evaluation
```

This generates:
- `evaluation_results/evaluation_results.json` — Detailed results
- `evaluation_results/retrieval_comparison.png` — Comparison charts

## 💡 Bonus Feature: Chat History Memory
The Streamlit interface maintains conversation history across messages, allowing the LLM to consider previous Q&A exchanges for follow-up questions.

## ⚠️ Disclaimer
This system is for **educational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

## 🛠️ Technologies Used
| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| TF-IDF | scikit-learn |
| Dense Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| LLM | HuggingFace Inference API |
| UI | Streamlit |
| Visualization | Matplotlib |
