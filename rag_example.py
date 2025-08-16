"""
Example: Retrieval-Augmented Generation (RAG) with FAISS and a small T5 model.
This script:
  1. Stores text documents in a vector index.
  2. Retrieves the most relevant docs for a query.
  3. Uses the retrieved docs as context for a generative model (Flan-T5).
"""

from sentence_transformers import SentenceTransformer
import faiss                   # Vector index for similarity search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────────────
# 1. Prepare a tiny document store (normally you'd have a large corpus)
# ─────────────────────────────────────────────────────
documents = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "Berlin is the capital of Germany and known for the Brandenburg Gate.",
    "Tokyo is the capital of Japan and famous for its modern skyline.",
    "Ottawa is the capital of Canada and lies on the Ottawa River."
]

# ─────────────────────────────────────────────────────
# 2. Create embeddings for documents and build a FAISS index
# ─────────────────────────────────────────────────────
# Use a small sentence-transformers model to embed the documents
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~22MB, fast and accurate
doc_embeddings = embedder.encode(documents, convert_to_tensor=False)

# Build a FAISS index (L2 distance) and add vectors
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # Flat index, simple but works
index.add(doc_embeddings)

# ─────────────────────────────────────────────────────
# 3. Retrieval helper: fetch top-k relevant docs for a query
# ─────────────────────────────────────────────────────
def retrieve(query: str, k: int = 2):
    """
    Encode the query, search the FAISS index for the nearest vectors,
    and return the corresponding documents.
    """
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, k)  # k nearest neighbors
    return [documents[i] for i in indices[0]]

# ─────────────────────────────────────────────────────
# 4. Load a small generative model (Flan-T5) for answering questions
# ─────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# ─────────────────────────────────────────────────────
# 5. RAG function: retrieve context, build prompt, generate answer
# ─────────────────────────────────────────────────────
def answer(query: str, k: int = 2) -> str:
    """
    Retrieve top-k docs, build a prompt including them, and let the
    generative model produce an answer.
    """
    retrieved_docs = retrieve(query, k=k)
    context = "\n".join(retrieved_docs)

    prompt = (
        "Answer the question based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ─────────────────────────────────────────────────────
# 6. Try it out
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    user_question = "What is the capital of Germany?"
    print("Question:", user_question)
    print("Answer  :", answer(user_question))

    user_question = "Which city is the capital of Canada?"
    print("\nQuestion:", user_question)
    print("Answer  :", answer(user_question))
