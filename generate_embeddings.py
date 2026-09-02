import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KB_JSON = os.path.join(BASE_DIR, "Backend", "cleaned_legal_data.json")
KB_EMBEDDINGS = os.path.join(BASE_DIR, "Backend", "legal_embeddings.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading legal knowledge base...")

with open(KB_JSON, "r", encoding="utf-8") as f:
    records = json.load(f)

texts = [
    f"Question: {record.get('question', '')}\n"
    f"Answer: {record.get('answer', '')}"
    for record in records
]

print(f"Records found: {len(texts)}")
print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
)

embeddings = embeddings.astype("float32")

print(f"Embedding shape: {embeddings.shape}")

np.save(KB_EMBEDDINGS, embeddings)

print(f"Saved embeddings to: {KB_EMBEDDINGS}")
print("Done!")