from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_facts(knowledge_base, query, top_k=3):
    corpus = [item['text'] for item in knowledge_base]
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]

    relevant_facts = [corpus[hit['corpus_id']] for hit in hits]
    return relevant_facts