import chromadb
from backend.config import INDEX_DIR
from backend.openrouter_client import embed_texts


class VectorStore:
    def __init__(self, name: str):
        self.name = name
        self._client = chromadb.PersistentClient(path=INDEX_DIR)
        self._collection = self._client.get_or_create_collection(name=name)

    def build_or_load(self, chunks: list[dict]):
        if self._collection.count() > 0:
            print(f"[{self.name}] Loaded existing Chroma collection: {self._collection.count()} chunks")
            return

        print(f"[{self.name}] No existing collection, embedding {len(chunks)} chunks via OpenRouter...")
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        ids = [f"{self.name}_{i}" for i in range(len(chunks))]
        metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[{self.name}] Built Chroma collection: {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self._collection.count() == 0:
            return []

        query_vec = embed_texts([query])[0]

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        chunks = []
        for text, meta in zip(documents, metadatas):
            chunk = {"text": text}
            chunk.update(meta)
            chunks.append(chunk)
        return chunks