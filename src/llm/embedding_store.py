import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

# Load data
df = pd.read_csv("data/processed/reviews_with_sentiment.csv")

# Choose embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert review text to vectors
embeddings = model.encode(df["review_text"].tolist(), show_progress_bar=True)

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "models/faiss_index.index")

# Save metadata (like review text and sentiment)
with open("models/review_metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

print("✅ FAISS index and metadata saved.")