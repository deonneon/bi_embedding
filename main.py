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


# --- 3. Train Word2Vec Model ---
print("Step 3: Training Word2Vec model...")
# Parameters:
# - vector_size: The dimensionality of the word vectors.
# - window: The maximum distance between the current and predicted word within a sentence.
# - min_count: Ignores all words with a total frequency lower than this.
# - workers: Number of CPU cores to use.
w2v_model = Word2Vec(
    sentences=corpus, vector_size=100, window=5, min_count=2, workers=4
)

w2v_model.train(corpus, total_examples=len(corpus), epochs=20)
print("Word2Vec model trained successfully.")


# --- 4. Create Document Vectors by Averaging Word Vectors ---
print("Step 4: Creating document vectors...")


def get_document_vector(doc_tokens, model):
    # Get the vector for each word in the document, if it exists in the model's vocabulary
    word_vectors = [model.wv[word] for word in doc_tokens if word in model.wv]

    if not word_vectors:
        # If no words in the document are in the vocab, return a zero vector
        return np.zeros(model.vector_size)

    # Return the mean of the word vectors
    return np.mean(word_vectors, axis=0)


document_vectors = np.array([get_document_vector(doc, w2v_model) for doc in corpus])
print(f"Created document vectors with shape: {document_vectors.shape}")


# --- 5. Reduce Dimensionality with UMAP ---
print("Step 5: Reducing dimensionality with UMAP...")
# UMAP works well with cosine distance for text-based vectors
reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
)

reduced_embeddings = reducer.fit_transform(document_vectors)


# --- 6. Cluster with HDBSCAN ---
print("Step 6: Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5, metric="euclidean", cluster_selection_method="eom"
)  # 'eom' is often a good choice

clusterer.fit(reduced_embeddings)
df["cluster"] = clusterer.labels_


# --- 7. Visualize the Clusters ---
print("Step 7: Visualizing clusters...")
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(14, 9))

# Create a color palette
unique_clusters = np.unique(clusterer.labels_)
n_clusters = len(unique_clusters)
palette = sns.color_palette("hsv", n_clusters)
# Make noise points (-1) black
cluster_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in clusterer.labels_]
cluster_member_colors = np.array(cluster_colors)

# Plot the points
scatter = plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    s=50,
    c=cluster_member_colors,
    alpha=0.7,
)

plt.title("MicroStrategy Metadata Clusters (Gensim Word2Vec)", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)

# Create a legend
legend_elements = []
for i, cluster_id in enumerate(unique_clusters):
    if cluster_id == -1:
        label = "Noise"
        color = (0.0, 0.0, 0.0)
    else:
        label = f"Cluster {cluster_id}"
        color = palette[i - 1 if -1 in unique_clusters else i]
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=10,
        )
    )

plt.legend(handles=legend_elements, title="Clusters")
plt.grid(True)
plt.show()

# --- 8. Analyze Results ---
print("\nStep 8: Analyzing cluster results...")
# Save the clustered data to a CSV file
df.to_csv("microstrategy_reports_clustered_gensim.csv", index=False)
print("Saved clustered data to 'microstrategy_reports_clustered_gensim.csv'")

# Print the number of items in each cluster
print("\nCluster Distribution:")
print(df["cluster"].value_counts())

# Print sample items from each cluster
print("\nSample items from each cluster:")
for cluster_id in sorted(df["cluster"].unique()):
    if cluster_id == -1:
        print("\n--- Noise Points (Not in a cluster) ---")
    else:
        print(f"\n--- Cluster {cluster_id} ---")

    sample = df[df["cluster"] == cluster_id].head(5)
    for index, row in sample.iterrows():
        print(f"  - {row['name']}")
