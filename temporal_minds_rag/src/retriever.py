from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, embed_fields=["text"]):
        self.embed_fields = embed_fields
        self.index = None
        self.documents = []
        self.embeddings = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def combine_fields(self, doc):
        return " ".join([str(doc.get(field, "")) for field in self.embed_fields])

    def build_index(self, knowledge_base):
        self.documents = knowledge_base
        corpus = [self.combine_fields(doc) for doc in knowledge_base]
        embeddings = self.embedder.encode(corpus, convert_to_numpy=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def query(self, user_query, top_k=3, persona=None):
        if persona:
            filtered = [doc for doc in self.documents if doc.get("persona") == persona]
            if not filtered:
                return []

            corpus = [self.combine_fields(doc) for doc in filtered]
            embeddings = self.embedder.encode(corpus, convert_to_numpy=True)

            faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings)

            query_embedding = self.embedder.encode([user_query], convert_to_numpy=True)
            distances, indices = faiss_index.search(query_embedding, top_k)
            return [filtered[i] for i in indices[0]]

        query_embedding = self.embedder.encode([user_query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
