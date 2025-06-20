import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Part 1: Gensim Doc2Vec Model Training ---
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# --- Part 2: BERTopic Clustering and Topic Modeling ---
from bertopic import BERTopic
from umap import UMAP

print("--- SCRIPT START ---")
print(f"Current Time: {pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')}")

# --- 1. Load and Prepare Data ---
print("\nStep 1: Loading and preparing data...")
try:
    df = pd.read_csv("microstrategy_reports.csv")
except FileNotFoundError:
    print("Error: 'microstrategy_reports.csv' not found.")
    print("Creating a dummy DataFrame for demonstration purposes.")
    data = {
        "id": [f"ID{i}" for i in range(100)],
        "name": [
            "Q1 Sales Analysis",
            "Q2 Sales Analysis",
            "Regional Sales Performance",
            "YoY Sales Growth",
            "Inventory Stock Status",
            "Warehouse Inventory Levels",
            "Product Reorder Report",
            "Inventory Turnover Rate",
            "Marketing Campaign ROI",
            "Social Media Engagement",
            "Website Traffic Analytics",
            "Lead Generation Funnel",
            "Financial Health Summary",
            "Quarterly P&L Statement",
            "Accounts Receivable Aging",
            "Cash Flow Forecast",
        ]
        * 6
        + ["Extra Report 1", "Extra Report 2", "Extra Report 3", "Extra Report 4"],
        "description": [
            "Detailed analysis of first quarter sales figures by product line.",
            "Comprehensive report on second quarter sales performance.",
            "A dashboard showing sales performance across different geographical regions.",
            "Year-over-year comparison of sales data to identify growth trends.",
            "Current stock levels for all products in the main warehouse.",
            "Real-time view of inventory quantities at all warehouse locations.",
            "A list of products that have fallen below the reorder threshold.",
            "Calculation of the inventory turnover ratio for the last fiscal year.",
            "Return on investment for recent marketing campaigns.",
            "Report on user engagement metrics from various social media platforms.",
            "Analysis of visitor traffic and behavior on the company website.",
            "Overview of the lead generation and conversion funnel.",
            "A high-level summary of the company's financial health.",
            "The profit and loss statement for the last quarter.",
            "A report detailing the aging of accounts receivable.",
            "Projection of cash flow for the next six months.",
        ]
        * 6
        + ["Misc 1", "Misc 2", "Misc 3", "Misc 4"],
    }
    df = pd.DataFrame(data)

# Combine name and description into a single text field
df["description"] = df["description"].fillna("")
df["documents"] = df["name"] + ". " + df["description"]
# Keep the original documents for BERTopic
documents = df["documents"].tolist()

# --- 2. Preprocess Text and Create Tagged Corpus for Doc2Vec ---
print("\nStep 2: Preprocessing text for Doc2Vec...")
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text, re.I | re.A)
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 2]


# Create a tagged corpus: each document needs a unique integer tag
tagged_corpus = [
    TaggedDocument(preprocess(doc), [i]) for i, doc in enumerate(documents)
]

print(f"Created a tagged corpus with {len(tagged_corpus)} documents.")


# --- 3. Train Gensim Doc2Vec Model ---
print("\nStep 3: Training Doc2Vec model...")
# Parameters:
# - vector_size: Dimensionality of the document vectors.
# - min_count: Ignores all words with a total frequency lower than this.
# - epochs: Number of training passes over the corpus.
# - dm=1: Use the 'distributed memory' (PV-DM) algorithm, often performs better.
doc2vec_model = Doc2Vec(vector_size=150, min_count=2, epochs=40, dm=1, workers=4)

doc2vec_model.build_vocab(tagged_corpus)
doc2vec_model.train(
    tagged_corpus,
    total_examples=doc2vec_model.corpus_count,
    epochs=doc2vec_model.epochs,
)
print("Doc2Vec model trained successfully.")

# Extract the document vectors for all documents in the training corpus
# The vector for the document with tag 'i' is at index 'i' in model.dv
doc_vectors = doc2vec_model.dv.vectors
print(f"Extracted document vectors with shape: {doc_vectors.shape}")


# --- 4. Cluster with BERTopic using Pre-computed Embeddings ---
print("\nStep 4: Initializing and running BERTopic with Doc2Vec embeddings...")

# To ensure reproducibility, we can define our own UMAP and HDBSCAN models
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)

# Instantiate BERTopic
# We pass an empty "embedding_model" because we are providing our own pre-computed embeddings.
# This tells BERTopic to skip the embedding step.
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",  # Still needed for some internal steps
    umap_model=umap_model,
    min_topic_size=5,
    verbose=True,
)

# Use fit_transform with both the original documents and our custom embeddings
# BERTopic uses the embeddings for clustering and the documents for topic representation.
topics, probabilities = topic_model.fit_transform(documents, embeddings=doc_vectors)


# --- 5. Analyze and Visualize BERTopic Results ---
print("\nStep 5: Analyzing BERTopic results...")

# Get the main topic information
topic_info = topic_model.get_topic_info()
print("\nMost frequent topics:")
print(topic_info.head(10))

# Get the representative documents for a specific topic (e.g., the first topic, topic 0)
if 0 in topic_info.Topic.values:
    print("\nRepresentative documents for Topic 0:")
    rep_docs = topic_model.get_representative_docs(0)
    for doc in rep_docs:
        print(f"  - {doc}")
else:
    print("\nTopic 0 was not found (it may have been merged or is an outlier topic).")


# --- 6. Visualization ---
print("\nStep 6: Generating visualizations...")

# Visualize the topics (requires a browser or interactive environment)
try:
    fig1 = topic_model.visualize_topics()
    fig1.show()
    fig1.write_html("bertopic_topics.html")
    print("Saved interactive topic visualization to 'bertopic_topics.html'")
except Exception as e:
    print(
        f"Could not show interactive plot: {e}. If in a non-interactive script, this is expected."
    )

# Visualize the document hierarchy
try:
    fig2 = topic_model.visualize_hierarchy()
    fig2.show()
    fig2.write_html("bertopic_hierarchy.html")
    print("Saved interactive hierarchy visualization to 'bertopic_hierarchy.html'")
except Exception as e:
    print(
        f"Could not show interactive plot: {e}. If in a non-interactive script, this is expected."
    )


# Add cluster info back to the original DataFrame
df["bertopic_cluster"] = topics
df.to_csv("microstrategy_reports_bertopic.csv", index=False)
print(
    "\nSaved final results with BERTopic clusters to 'microstrategy_reports_bertopic.csv'"
)

print("\n--- SCRIPT COMPLETE ---")
