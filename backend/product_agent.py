import os
import pandas as pd
from backend.config import DATA_DIR
from backend.vector_store import VectorStore
from backend.openrouter_client import generate_answer

_product_store = VectorStore(name="product_index")


def _load_product_chunks() -> list[dict]:
    df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"), dtype=str)
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Product: {row.get('product_name', '')}. "
            f"Category: {row.get('category', '')}. "
            f"Price: {row.get('price', '')}. "
            f"Rating: {row.get('rating', '')}. "
            f"Stock: {row.get('stock', '')}. "
            f"Description: {row.get('description', '')}"
        )
        chunks.append({
            "text": text,
            "source": "products.csv",
            "product_id": row.get("product_id", ""),
        })
    return chunks


def init_product_agent():
    """Call once at startup to build/load the product vector index."""
    chunks = _load_product_chunks()
    _product_store.build_or_load(chunks)


PRODUCT_SYSTEM_PROMPT = """You are a helpful product recommendation assistant.
Answer the user's question using ONLY the product context below. If nothing
in the context fits what they're asking for, say so honestly instead of
making up a product. Be concise and mention prices/ratings when relevant.

Context:
{context}
"""


def answer(user_message: str) -> str:
    relevant_chunks = _product_store.search(user_message, top_k=5)
    context = "\n\n".join(c["text"] for c in relevant_chunks)
    prompt = PRODUCT_SYSTEM_PROMPT.format(context=context or "No relevant products found.")
    return generate_answer(prompt, user_message)