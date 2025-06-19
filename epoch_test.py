import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Prepare Data ---
print("Step 1: Loading and preparing data...")
try:
    df = pd.read_csv("microstrategy_reports.csv")
except FileNotFoundError:
    print("Error: 'microstrategy_reports.csv' not found.")
    print("Please run the metadata extraction script first.")
    # As a fallback, create a dummy DataFrame for demonstration
    data = {
        "id": [f"ID{i}" for i in range(5)],
        "name": [
            "Annual Sales Report",
            "Sales Performance Dashboard",
            "Inventory Stock Levels",
            "Product Inventory Details",
            "Quarterly Financial Summary",
        ],
        "description": [
            "A detailed report on annual sales figures.",
            "Dashboard showing key sales performance indicators.",
            "Current stock levels for all inventory items.",
            "Detailed information about product inventory.",
            "A summary of the quarterly financial results.",
        ],
    }
    df = pd.DataFrame(data)

# Combine name and description into a single text field
df["description"] = df["description"].fillna("")
df["text_for_clustering"] = df["name"] + " " + df["description"]


# --- 2. Preprocess Text for Word2Vec ---
print("Step 2: Preprocessing text for Word2Vec...")
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(
        r"[^a-zA-Z\s]", "", text, re.I | re.A
    )  # Remove punctuation and numbers
    tokens = word_tokenize(text)  # Tokenize
    # Remove stopwords and short tokens
    filtered_tokens = [
        word for word in tokens if word not in stop_words and len(word) > 2
    ]
    return filtered_tokens


# Apply preprocessing to create a list of token lists
corpus = [preprocess(text) for text in df["text_for_clustering"]]


# --- 3. Train Word2Vec Model (Modified for experimentation) ---
print("Step 3: Training Word2Vec model with varying epochs...")

# Define a list of epoch values to experiment with
epoch_values = [10, 20, 30, 40, 50]  # Example values

clustering_results = {}

for epochs in epoch_values:
    print(f"Training Word2Vec model for {epochs} epochs...")
    w2v_model = Word2Vec(
        sentences=corpus, vector_size=100, window=5, min_count=2, workers=4
    )

    w2v_model.train(corpus, total_examples=len(corpus), epochs=epochs)
    print(f"Word2Vec model trained successfully for {epochs} epochs.")

    # --- Steps 4-6: Create Document Vectors, UMAP, HDBSCAN ---
    # (These steps remain the same, but you'll re-run them for each epoch count)
    document_vectors = np.array([get_document_vector(doc, w2v_model) for doc in corpus])
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
    )
    reduced_embeddings = reducer.fit_transform(document_vectors)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, metric="euclidean", cluster_selection_method="eom"
    )
    clusterer.fit(reduced_embeddings)
    df[f"cluster_epochs_{epochs}"] = clusterer.labels_

    # Store or print relevant metrics for evaluation
    clustering_results[epochs] = {
        "cluster_counts": df[f"cluster_epochs_{epochs}"].value_counts().to_dict(),
        "sample_items": {
            cluster_id: df[df[f"cluster_epochs_{epochs}"] == cluster_id]["name"]
            .head(5)
            .tolist()
            for cluster_id in sorted(df[f"cluster_epochs_{epochs}"].unique())
        },
    }
    print(f"Clustering completed for {epochs} epochs.")

# --- Step 7: Visualize (Optional - you can choose to visualize for the best epoch count) ---
# You might want to select the best epoch count based on your analysis of the results
# and then generate the visualization for that specific result.

# --- Step 8: Analyze Results (Modified to compare results) ---
print("\nStep 8: Analyzing and comparing cluster results across epochs...")

# You can now iterate through the clustering_results dictionary
# and analyze how the cluster distribution and sample items change
for epochs, results in clustering_results.items():
    print(f"\n--- Results for {epochs} Epochs ---")
    print("Cluster Distribution:")
    print(results["cluster_counts"])
    print("Sample items from each cluster:")
    for cluster_id, sample_items in results["sample_items"].items():
        if cluster_id == -1:
            print("\n--- Noise Points (Not in a cluster) ---")
        else:
            print(f"\n--- Cluster {cluster_id} ---")
        for item in sample_items:
            print(f"  - {item}")

# Optionally, save the DataFrame with all epoch results
df.to_csv("microstrategy_reports_clustered_epochs_comparison.csv", index=False)
print(
    "\nSaved clustered data with epoch comparisons to 'microstrategy_reports_clustered_epochs_comparison.csv'"
)
